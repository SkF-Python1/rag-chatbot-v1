# Context-Aware RAG Chatbot (Local Setup)

A powerful Retrieval-Augmented Generation (RAG) chatbot system built with LangChain, FAISS, and FastAPI, designed to work with local documents and LM Studio for privacy and cost-effectiveness.

## Features

- 🚀 **High Performance**: Optimized for low latency (<1.2s response time)
- 📚 **Multi-Format Support**: PDF, HTML, Word documents
- 🔍 **Semantic Search**: FAISS vector store for efficient document retrieval
- 🤖 **Local LLM**: Integration with LM Studio and Mistral model
- 🌐 **Web Interface**: FastAPI-based REST API with chat interface
- 🔒 **Privacy First**: Complete local setup, no external API calls
- 📈 **Scalable**: Handles 10K+ documents efficiently

## Prerequisites

- Python 3.8+
- LM Studio installed and running
- mistralai/devstral-small-2507 model downloaded in LM Studio

## Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
python setup.py
```

### Option 2: Manual Setup
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure LM Studio**
   - Start LM Studio
   - Load the `mistralai/devstral-small-2507` model
   - Ensure the server is running on `http://localhost:1234`

3. **Prepare Documents**
   - Place your documents (PDF, HTML, Word) in the `./documents` folder
   - Run the ingestion script to process and index documents:
   ```bash
   python scripts/ingest_documents.py ingest --path ./documents
   ```

4. **Start the Application**
   ```bash
   python main.py
   ```

5. **Access the Chat Interface**
   - Open your browser to `http://localhost:8000`
   - Start chatting with your documents!

## Project Structure

```
RAG - Local/
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── main.py                # FastAPI application
├── src/
│   ├── document_processor.py  # Document processing utilities
│   ├── vector_store.py        # FAISS vector store management
│   ├── llm_client.py          # LM Studio integration
│   └── rag_pipeline.py        # RAG logic
├── scripts/
│   └── ingest_documents.py    # Document ingestion script
├── static/                    # Web interface files
├── documents/                 # Your source documents
├── vector_store/             # FAISS index storage
└── logs/                     # Application logs
```

## Configuration

Edit `config.py` to customize:
- LM Studio connection settings
- Document processing parameters
- Vector store configuration
- API server settings

## Performance Targets

- **Answer Relevance**: 93%+
- **Response Latency**: <1.2s
- **Document Capacity**: 10K+ documents
- **Concurrent Users**: Optimized for multiple simultaneous queries

## Usage

1. **Document Ingestion**: Add documents to the `documents` folder and run ingestion
2. **Chat Interface**: Ask questions about your documents through the web interface
3. **API Access**: Use the REST API endpoints for programmatic access

## API Endpoints

- `POST /chat` - Send a message and get AI response
- `POST /upload` - Upload new documents
- `GET /health` - System health check
- `GET /stats` - Vector store statistics 