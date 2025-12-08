# Doc Chat Studio

A modern, intelligent document chat application built with Streamlit, LangChain, and FAISS for conversational Q&A over your documents using RAG (Retrieval-Augmented Generation).

## 🎯 Overview

Doc Chat Studio allows users to upload multiple documents (PDFs, DOCX, TXT, MD), automatically indexes them using FAISS vector embeddings, and enables natural language conversations with multi-step reasoning powered by OpenAI's GPT models.

## ✨ Key Features

- **Multi-Document Upload**: Support for PDF, DOCX, TXT, and Markdown files
- **Intelligent Chunking**: Uses `RecursiveCharacterTextSplitter` for semantic text splitting
- **LangChain Retriever**: FAISS-based `VectorStoreRetriever` for efficient similarity search
- **Multi-Step Reasoning**: Sequential chain (Summarize → Analyze → Answer) using LangChain
- **Conversation Memory**: Context-aware responses using `ConversationBufferMemory`
- **Source Citations**: Automatic citation of source documents in answers
- **Modern UI**: Centered, responsive design with custom CSS styling
- **In-Memory Storage**: FAISS vectorstore managed by LangChain

## 🏗️ Architecture

### Tech Stack

- **UI Framework**: Streamlit
- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: HuggingFaceEmbeddings (all-MiniLM-L6-v2) via LangChain
- **Vector Database**: LangChain FAISS VectorStore (in-memory)
- **Retriever**: VectorStoreRetriever with similarity search
- **Orchestration**: LangChain
- **Document Processing**: PyPDF, python-docx

### RAG Pipeline

1. **Document Upload** → Extract text from files
2. **Text Chunking** → Split into semantic chunks (1000 chars, 200 overlap)
3. **Embedding** → Convert chunks to 384-dim vectors using HuggingFaceEmbeddings
4. **Indexing** → Store in LangChain FAISS VectorStore
5. **Query** → Embed user question
6. **Retrieval** → VectorStoreRetriever finds top-4 similar chunks using `.invoke()`
7. **Multi-Step Reasoning**:
   - **Step 1**: Summarize retrieved context
   - **Step 2**: Analyze relevance to question
   - **Step 3**: Generate final answer with citations
8. **Memory** → Save Q&A pairs for context-aware follow-ups

## 📋 Core Functions

### Memory & LLM Management

| Function                     | Description                                                 |
| ---------------------------- | ----------------------------------------------------------- |
| `_get_conversation_memory()` | Initialize/return ConversationBufferMemory for chat history |
| `_get_langchain_llm()`       | Create/cache ChatOpenAI instance with API key               |

### Prompt Engineering

| Function                     | Description                                                            |
| ---------------------------- | ---------------------------------------------------------------------- |
| `build_prompt()`             | Build OpenAI-compatible messages from template and context             |
| `_create_sequential_chain()` | Create 3-step LangChain SequentialChain (summarize → analyze → answer) |

### Document Processing

| Function        | Description                                                                      |
| --------------- | -------------------------------------------------------------------------------- |
| `_read_file()`  | Extract text from PDF/DOCX/TXT/MD files                                          |
| `_chunk_text()` | Split text using RecursiveCharacterTextSplitter (1000/200)                       |
| `_add_docs()`   | Process files: extract → chunk → create LangChain Documents → add to vectorstore |

### Vector Search

| Function            | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| `_get_embeddings()` | Lazy-load LangChain HuggingFaceEmbeddings (all-MiniLM-L6-v2)      |
| `_get_embedder()`   | Lazy-load SentenceTransformer model (backup for compatibility)    |
| `_get_retriever()`  | Create VectorStoreRetriever from FAISS vectorstore with top-k     |
| `_search_index()`   | Retrieve top-k documents using retriever.invoke() and format them |

### Answer Generation

| Function             | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `_call_llm()`        | Generate answer using multi-step reasoning chain with memory |
| `_generate_answer()` | Main RAG pipeline: retrieve contexts → call LLM              |

### UI Components

| Function               | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `_inject_styles()`     | Apply custom CSS for modern, centered design    |
| `_init_state()`        | Initialize Streamlit session state variables    |
| `_render_header()`     | Display app title                               |
| `_render_doc_pills()`  | Show uploaded document badges with size         |
| `_render_left_panel()` | Document upload UI and file list                |
| `_render_chat_area()`  | Chat history display and input box              |
| `main()`               | App entry point: initialize state → render tabs |

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone repository
cd capstone/aiassistant

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
# Windows PowerShell:
$env:OPENAI_API_KEY="your-key-here"

# Or create .env file:
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

```
streamlit>=1.37.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
pypdf>=4.2.0
python-docx>=1.1.0
openai>=1.42.0
langchain>=0.2.0
langchain-openai>=0.2.1
langchain-community>=0.2.0
langchain-classic
python-dotenv
numpy>=1.24.0
```

## 💡 Usage

### Upload Documents

1. Navigate to the **Documents** tab
2. Click "Browse files" or drag-and-drop
3. Select PDF, DOCX, TXT, or MD files
4. Click "Add to workspace"
5. Wait for indexing to complete

### Ask Questions

1. Navigate to the **Chat** tab
2. Type your question in the input box
3. Press Enter
4. The app will:
   - Retrieve relevant document chunks
   - Summarize context (Step 1/3)
   - Analyze relevance (Step 2/3)
   - Generate answer with citations (Step 3/3)

### Follow-up Questions

The app maintains conversation history, so you can ask follow-up questions that reference previous answers:

```
You: What is the leave policy?
Bot: Employees get 15 days annual leave... [Source: policy.pdf]

You: Can I carry forward unused leave?
Bot: Based on our previous discussion about the leave policy...
```

## 🔧 Configuration

### Chunk Size

Modify chunking parameters in `_chunk_text()`:

```python
chunk_size=1000,  # Characters per chunk
overlap=200       # Overlapping characters
```

### Top-K Retrieval

Change number of retrieved chunks in `_generate_answer()`:

```python
contexts = _search_index(question, top_k=4)  # Retrieve 4 chunks
```

### LLM Model

Update model in `_get_langchain_llm()`:

```python
model_name="gpt-3.5-turbo",  # or "gpt-4", "gpt-4-turbo"
temperature=0.3,              # Lower = more deterministic
```

## 🎨 UI Features

- **Centered Layout**: Max-width 880px for optimal readability
- **Light Theme**: Clean white background with subtle borders
- **Card Design**: Elevated surfaces with shadow effects
- **Document Pills**: Visual badges showing uploaded files
- **Chat Bubbles**: Distinct styling for user/assistant messages
- **Progress Indicators**: Real-time feedback during multi-step reasoning
- **Timestamps**: UTC timestamps for each message

## ⚠️ Limitations

### Current Limitations

- **In-Memory Only**: FAISS index is not persisted to disk (resets on app restart)
- **No Persistence**: Uploaded documents and chat history lost on page refresh
- **Session-Based**: Each browser session maintains separate state
- **No Multi-User Support**: Not designed for concurrent users

### Future Enhancements

- [ ] Persistent FAISS storage (save/load from disk)
- [ ] Document metadata persistence (SQLite/JSON)
- [ ] Chat history export/import
- [ ] Multi-user support with authentication
- [ ] Analytics dashboard (document stats, query patterns)
- [ ] Advanced filters (date range, document type)
- [ ] Visualization of embeddings (t-SNE/UMAP)

## 🐛 Troubleshooting

### "No module named 'langchain.chains'"

```bash
pip install langchain>=0.2.0 langchain-classic --upgrade
```

### "One input key expected got [...]"

Ensure `ConversationBufferMemory` has `input_key="question"`:

```python
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",  # Required for multiple inputs
    output_key="final_answer"
)
```

### Context Not Being Used

Check that chat history is being formatted correctly:

```python
# Format messages as string before passing to chain
chat_history_str = "\n".join([f"{msg.type}: {msg.content}"
                               for msg in memory.chat_memory.messages])
```

### FAISS Not Available

The app automatically falls back to NumPy cosine similarity:

```python
# NumPy fallback for similarity search
if FAISS_AVAILABLE:
    # Use FAISS
else:
    # Use NumPy cosine similarity
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ using Streamlit, LangChain, and OpenAI**
