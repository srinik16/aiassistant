import os
import textwrap
import time
from datetime import datetime
from io import BytesIO
from typing import Any, List, Optional, Sequence
from dotenv import load_dotenv
from langchain_classic.chains import LLMChain, SequentialChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document

load_dotenv()

try:
    import faiss  #faiss-cpu on Windows
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


# Called at the start of each conversation turn to retrieve or create the memory buffer for context management.
def _get_conversation_memory() -> ConversationBufferMemory:
    """Return (and lazily initialize) the conversation memory buffer."""
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
            output_key="final_answer",
        )
    return st.session_state.conversation_memory


def _get_langchain_llm() -> Optional[ChatOpenAI]:
    """Return a cached ChatOpenAI instance if API key is available."""
    api_key = "your-key-here"
    if not api_key:
        return None
    if "langchain_llm" not in st.session_state:
        st.session_state.langchain_llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key,
        )
    return st.session_state.langchain_llm


# Basic page config
st.set_page_config(
    page_title="Doc Chat Studio",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def build_prompt(question: str, contexts: Sequence[dict]) -> list:
    """Build chat messages using a single template via ChatPromptTemplate.from_template.
    Returns a list of OpenAI-compatible messages dicts (system + user).
    """
    context_block = "\n\n".join(
        [f"[Source: {c['source']}]\n{c['text']}" for c in contexts]
    )

    template = ChatPromptTemplate.from_template(
        """
        1. Never invent or assume facts that are not explicitly provided.
        2. Always check the previous conversation history before responding.
        3. If the user asks a question, first look at:
        - Previous user messages
        - Previous assistant responses
        - The provided context/documents
        4. If the answer exists in the previous messages, reuse that exact information.
        5. If the information is missing, respond with:
        "The requested information is not available in the provided context."
        6. Keep the answer grounded in the given context.
        7. Do not break character or explain the rules unless asked.
        8. if no response is found in the context, respond with: "No sufficient information/question."
        - Cite sources like [Source: filename].

        Context:
        {context}

        Question:
        {question}
        """
    )

    lc_messages = template.format_messages(context=context_block, question=question)
    # Convert LangChain messages to OpenAI dict format
    messages = []
    for m in lc_messages:
        role = "system" if m.type == "system" else ("user" if m.type == "human" else "assistant")
        messages.append({"role": role, "content": m.content})
    return messages


def _create_sequential_chain() -> Optional[SequentialChain]:
    """Build a SequentialChain that performs summarize → analyze → answer."""
    llm = _get_langchain_llm()
    if llm is None:
        return None

    memory = _get_conversation_memory()

    summary_chain = LLMChain(
        llm=llm,
        output_key="summary",
        prompt=PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
            1. Never invent or assume facts that are not explicitly provided.
            2. Always check the previous conversation history before responding.
            3. If the user asks a question, first look at:
            - Previous user messages
            - Previous assistant responses
            - The provided context/documents
            4. If the answer exists in the previous messages, reuse that exact information.
            5. If the information is missing, respond with:
            "The requested information is not available in the provided context."
            6. Keep the answer grounded in the given context.
            7. Do not break character or explain the rules unless asked.
            8. if no response is found in the context, respond with: "No sufficient information/question."
            - Cite sources like [Source: filename].

            Context:
            {context}

            Question for reference:
            {question}

            Prior conversation (for context only):
            {chat_history}

            Summary:""",
        ),
    )

    comparison_chain = LLMChain(
        llm=llm,
        output_key="analysis",
        prompt=PromptTemplate(
            input_variables=["summary", "question", "context"],
            template="""
            You are an intelligent research assistant. Use the context, summary, and analysis to answer the question.
            1. Never invent or assume facts that are not explicitly provided.
            2. Always check the previous conversation history before responding.
            3. If the user asks a question, first look at:
            - Previous user messages
            - Previous assistant responses
            - The provided context/documents
            4. If the answer exists in the previous messages, reuse that exact information.
            5. If the information is missing, respond with:
            "The requested information is not available in the provided context."
            6. Keep the answer grounded in the given context.
            7. Do not break character or explain the rules unless asked.
            8. if no response is found in the context, respond with: "No sufficient information/question."
            - Cite sources like [Source: filename].

            Summary:
            {summary}

            Context (optional reference):
            {context}

            Question:
            {question}

            Analysis:""",
        ),
    )

    final_chain = LLMChain(
        llm=llm,
        output_key="final_answer",
        memory=memory,
        prompt=PromptTemplate(
            input_variables=["context", "summary", "analysis", "question", "chat_history"],
            template="""
            You are an intelligent research assistant. Use the context, summary, and analysis to answer the question.
            1. Never invent or assume facts that are not explicitly provided.
            2. Always check the previous conversation history before responding.
            3. If the user asks a question, first look at:
            - Previous user messages
            - Previous assistant responses
            - The provided context/documents
            4. If the answer exists in the previous messages, reuse that exact information.
            5. If the information is missing, respond with:
            "The requested information is not available in the provided context."
            6. Keep the answer grounded in the given context.
            7. Do not break character or explain the rules unless asked.
            8. if no response is found in the context, respond with: "No sufficient information/question."
            - Cite sources like [Source: filename].

            Context:
            {context}

            Summary:
            {summary}

            Analysis:
            {analysis}

            Prior conversation:
            {chat_history}

            Question: {question}

            Respond with a structured answer, using bullet points where it helps.

            Answer:""",
        ),
    )

    return SequentialChain(
        chains=[summary_chain, comparison_chain, final_chain],
        input_variables=["context", "question", "chat_history"],
        output_variables=["summary", "analysis", "final_answer"],
        verbose=False,
    )

def _inject_styles() -> None:
    """Inject custom CSS for a modern, focused look with gradient background and card styling."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #ffffff;
                --panel: #ffffff;
                --accent: #2563eb;
                --accent-2: #0891b2;
                --text: #0f172a;
                --muted: #334155;
                --border: rgba(2, 6, 23, 0.08);
                --glow: 0 6px 30px rgba(37, 99, 235, 0.15);
            }
            * { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
            body { background: #ffffff; color: var(--text); }
            .block-container { 
                padding: 2rem 1.5rem 2rem 1.5rem; 
                max-width: 880px; 
                margin-left: auto; 
                margin-right: auto; 
            }
            .stButton>button, .stTextInput>div>div>input, .stTextArea textarea {
                border-radius: 12px;
                border: 1px solid var(--border);
                background: #ffffff;
                color: var(--text);
            }
            .stButton>button:hover { border-color: var(--accent); box-shadow: var(--glow); }
            .pill { display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px;
                    background: #ffffff; border: 1px solid var(--border); color: var(--muted); margin: 4px 6px 4px 0; }
            .pill span { color: var(--text); font-weight: 600; }
            .card { background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: var(--glow); }
            .hero-title { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.01em; color: var(--text); }
            .hero-sub { color: var(--muted); max-width: 720px; }
            .chat-bubble { padding: 0.9rem 1rem; border-radius: 12px; margin-bottom: 0.6rem; border: 1px solid var(--border); background: #ffffff; }
            .user { background: #ffffff; }
            .assistant { background: #ffffff; }
            .timestamp { font-size: 0.8rem; color: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    """Initialize session state for documents, chat messages, RAG corpus, FAISS index, and embedder."""
    # Store uploaded document metadata (name, size, content)
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    # Store chat history (user questions + assistant answers)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Store text chunks with source metadata for retrieval
    if "corpus" not in st.session_state:
        st.session_state.corpus = []  # list of dicts with text + source
    # LangChain FAISS vectorstore with retriever
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    # LangChain embeddings model
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    # Sentence transformer model for embedding text (backup)
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    # Fallback embeddings matrix for numpy-based similarity when FAISS isn't available
    if "embeddings_matrix" not in st.session_state:
        st.session_state.embeddings_matrix = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load LangChain HuggingFace embeddings model (all-MiniLM-L6-v2) for vectorstore."""
    if st.session_state.embeddings is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": False}
            )
    return st.session_state.embeddings


def _get_embedder() -> SentenceTransformer:
    """Lazy-load the sentence transformer model (all-MiniLM-L6-v2) for embedding documents and queries."""
    if st.session_state.embedder is None:
        with st.spinner("Loading embedding model..."):
            # Load lightweight 384-dim model suitable for semantic similarity
            # The all-MiniLM-L6-v2 is a pre-trained sentence-transformers model designed to map sentences and paragraphs into a 384-dimensional dense vector space.
            st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return st.session_state.embedder


def _get_retriever(k: int = 4):
    """Get LangChain retriever from vectorstore for top-k similarity search."""
    if st.session_state.vectorstore is None:
        return None
    return st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def _read_file(f: Any) -> str:
    """Extract text from uploaded file. Supports PDF, DOCX, MD, and TXT formats."""
    suffix = str(f.name).lower()
    data = f.read()

    # Extract text from each page of PDF
    if suffix.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # Extract text from Word document paragraphs
    if suffix.endswith(".docx"):
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    # Read plain text and markdown files
    if suffix.endswith(".md") or suffix.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    return ""


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text using RecursiveCharacterTextSplitter for better semantic chunking."""
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    st.success(f"Total number of chunks: {len(chunks)}.")
    return chunks


def _search_index(query: str, top_k: int = 4) -> List[dict]:
    """Retrieve top_k most similar text chunks using LangChain retriever."""
    if st.session_state.vectorstore is None:
        return []

    retriever = _get_retriever(k=top_k)
    if retriever is None:
        return []

    # Use LangChain retriever to invoke and get relevant documents
    docs = retriever.invoke(query)
    
    contexts = []
    for doc in docs:
        contexts.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": 0.0  # LangChain similarity_search_with_score could be used for actual scores
        })
    return contexts


def _call_llm(question: str, contexts: Sequence[dict]) -> str:
    """Generate an answer using multi-step reasoning when possible."""
    api_key = "your-key-here"

    context_block = "\n\n".join([
        f"[Source: {ctx['source']}]\n{ctx['text']}" for ctx in contexts
    ])
    sources = {ctx["source"] for ctx in contexts if ctx.get("source")}

    if api_key:
        chain = _create_sequential_chain()
        if chain is not None:
            memory = _get_conversation_memory()
            
            # Format chat history as a readable string
            chat_history_str = ""
            if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
                history_parts = []
                for msg in memory.chat_memory.messages:
                    role = msg.type if hasattr(msg, 'type') else 'unknown'
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    history_parts.append(f"{role}: {content}")
                chat_history_str = "\n".join(history_parts)
                #st.markdown(f"Chat history for context:\n{chat_history_str}")
            
            with st.spinner("Reasoning over your documents..."):
                status = st.empty()
                try:
                    status.info("Step 1/3 • Summarizing context")
                    time.sleep(0.2)
                    result = chain({
                        "context": context_block or "(no context)",
                        "question": question,
                        "chat_history": chat_history_str or "(no prior conversation)",
                    })
                    status.info("Step 2/3 • Comparing relevant details")
                    time.sleep(0.2)
                    status.info("Step 3/3 • Crafting final answer")
                    time.sleep(0.2)
                finally:
                    status.empty()

            answer_text = result.get("final_answer", "")
            if sources:
                citations = ", ".join(sorted({f"[{src}]" for src in sources}))
                answer_text = f"{answer_text}\n\n**Sources:** {citations}"
            memory.save_context({"question": question}, {"final_answer": answer_text})
            return answer_text.strip() or "No answer generated."

        # Fallback to direct chat completion if chain could not be created
        client = OpenAI(api_key=api_key)
        messages = build_prompt(question, contexts)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        text = completion.choices[0].message.content.strip()
        if sources:
            citations = ", ".join(sorted({f"[{src}]" for src in sources}))
            text = f"{text}\n\n**Sources:** {citations}"
        _get_conversation_memory().save_context({"question": question}, {"final_answer": text})
        return text

    # Fallback: return retrieved snippets if no OpenAI key
    if not contexts:
        return "No documents indexed yet. Upload files to ground answers."

    stitched = "\n\n".join([c["text"] for c in contexts])[:1200]
    offline_answer = textwrap.dedent(
        f"""
        (Offline demo) Using the closest snippets:\n{stitched}\n\nQuestion: {question}
        """
    ).strip()
    _get_conversation_memory().save_context({"question": question}, {"final_answer": offline_answer})
    return offline_answer

def _render_header() -> None:
    """Display the app title and description at the top of the page."""
    st.markdown("<div class='hero-title'>Doc Chat Studio</div>", unsafe_allow_html=True)   


def _render_doc_pills() -> None:
    """Display uploaded documents as pill badges showing filename and size."""
    if not st.session_state.uploaded_docs:
        st.caption("No documents yet.")
        return
    # Build HTML pill badges for each document
    pills = []
    for doc in st.session_state.uploaded_docs:
        pills.append(
            f"<span class='pill'><span>{doc['name']}</span> - {doc['size_kb']} KB</span>"
        )
    st.markdown(" ".join(pills), unsafe_allow_html=True)

# Add Documents
def _add_docs(files: List[Any]) -> None:
    """Process uploaded files: extract text, chunk, embed, and add to LangChain FAISS vectorstore."""
    from langchain_core.documents import Document as LangChainDocument
    
    embeddings = _get_embeddings()
    all_documents = []
    
    for f in files:
        # Step 1: Extract raw text from file
        raw_text = _read_file(f)
        if not raw_text.strip():
            st.warning(f"No text extracted from {f.name}; skipped.")
            continue

        # Step 2: Split text into overlapping chunks
        chunks = _chunk_text(raw_text)
        if not chunks:
            st.warning(f"No textual chunks produced from {f.name}; skipped.")
            continue
        
        # Step 3: Create LangChain Document objects with metadata
        for i, chunk in enumerate(chunks):
            doc = LangChainDocument(
                page_content=chunk,
                metadata={"source": f.name, "chunk": i}
            )
            all_documents.append(doc)
            # Also store in corpus for backward compatibility
            st.session_state.corpus.append({
                "text": chunk,
                "source": f.name,
                "chunk": i,
            })

        # Step 4: Save document metadata for UI display
        st.session_state.uploaded_docs.append({
            "name": f.name,
            "size_kb": round(len(raw_text.encode("utf-8")) / 1024, 2),
            "uploaded_at": datetime.utcnow(),
            "content": raw_text,
        })
    
    # Step 5: Create or update FAISS vectorstore with LangChain wrapper
    if all_documents:
        if st.session_state.vectorstore is None:
            # Create new vectorstore
            st.session_state.vectorstore = FAISS.from_documents(
                documents=all_documents,
                embedding=embeddings
            )
        else:
            # Add documents to existing vectorstore
            st.session_state.vectorstore.add_documents(all_documents)


def _generate_answer(question: str) -> str:
    """Main RAG pipeline: retrieve relevant chunks from FAISS, then generate answer with LLM."""
    # Step 1: Retrieve top 4 most relevant chunks
    contexts = _search_index(question, top_k=4)
    # Step 2: Generate answer using retrieved context
    return _call_llm(question, contexts)


def _render_chat_area() -> None:
    """Display chat history and input box. Process user questions through RAG pipeline."""
    st.markdown("### Chat")
    chat_container = st.container()
    
    # Display all previous messages
    with chat_container:
        for msg in st.session_state.messages:
            role_class = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(msg["role"]):
                st.markdown(
                    f"<div class='chat-bubble {role_class}'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<div class='timestamp'>{msg['time']}</div>", unsafe_allow_html=True)

    # Handle new user input
    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt, "time": timestamp})
        # Generate RAG-based answer
        answer = _generate_answer(prompt)
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer, "time": timestamp})
        st.rerun()


def _render_left_panel() -> None:
    """Display document upload UI and list of uploaded documents."""
    st.markdown("### Documents")
    
    # File upload form
    with st.form("upload-form"):
        files = st.file_uploader(
            "Drop multiple PDFs, text, Word, or markdown files",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "md"],
        )
        submitted = st.form_submit_button("Add to workspace")
        if submitted and files:
            # Process and index uploaded files
            _add_docs(files)
            st.success(f"Added {len(files)} document(s).")
    
    # Show uploaded document badges
    _render_doc_pills()
    st.divider()
    st.markdown(
        "**Note:** Using LangChain `FAISS VectorStore` with `Retriever` for semantic search (in-memory)."
    )


def main() -> None:
    """Main app entry point: initialize state, render UI with two-column layout (docs + chat)."""
    # Apply custom CSS styling
    _inject_styles()
    # Initialize session state variables
    _init_state()
    # Render page header
    _render_header()
    
    # Create two-column layout: left for docs, right for chat
    tabs = st.tabs(["Documents", "Chat"])
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        _render_left_panel()
        st.markdown("</div>", unsafe_allow_html=True)
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        _render_chat_area()
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()