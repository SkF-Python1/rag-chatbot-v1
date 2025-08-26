"""FastAPI application for RAG chatbot system."""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from loguru import logger

from config import settings
from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_client import LMStudioClient


# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global rag_pipeline
    
    # Startup
    logger.info("Starting RAG chatbot application")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = create_rag_pipeline()
        
        # Run health check
        health = await rag_pipeline.health_check()
        if health["healthy"]:
            logger.info("RAG pipeline initialized successfully")
        else:
            logger.warning("RAG pipeline health check failed, but continuing")
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Continue anyway - some endpoints might still work
        
    yield
    
    # Shutdown
    logger.info("Shutting down RAG chatbot application")
    if rag_pipeline:
        await rag_pipeline.close()


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Local RAG chatbot with LM Studio and FAISS",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    use_rag: bool = True
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    response_time: float
    retrieval_time: float
    generation_time: float
    confidence_score: float
    context_chunks: int
    use_rag: bool


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int
    processing_time: float
    sources: List[str]


class HealthResponse(BaseModel):
    healthy: bool
    vector_store: Dict[str, Any]
    llm_client: Dict[str, Any]


class StatsResponse(BaseModel):
    vector_store: Dict[str, Any]
    llm_model: str
    llm_base_url: str
    settings: Dict[str, Any]


# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface."""
    html_file = Path("static/index.html")
    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Return a simple HTML page if static file doesn't exist
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Chatbot</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .user { background-color: #e3f2fd; }
                .assistant { background-color: #f3e5f5; }
                input[type="text"] { width: 70%; padding: 10px; }
                button { padding: 10px 20px; margin: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>RAG Chatbot</h1>
                <div id="chat-container"></div>
                <input type="text" id="messageInput" placeholder="Type your message..." />
                <button onclick="sendMessage()">Send</button>
                <button onclick="clearChat()">Clear</button>
            </div>
            <script>
                async function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    if (!message) return;
                    
                    const chatContainer = document.getElementById('chat-container');
                    chatContainer.innerHTML += `<div class="message user">You: ${message}</div>`;
                    
                    input.value = '';
                    input.disabled = true;
                    
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ message: message })
                        });
                        
                        const data = await response.json();
                        chatContainer.innerHTML += `<div class="message assistant">Assistant: ${data.response}</div>`;
                        
                        if (data.sources && data.sources.length > 0) {
                            chatContainer.innerHTML += `<div class="message assistant"><small>Sources: ${data.sources.length} documents</small></div>`;
                        }
                        
                    } catch (error) {
                        chatContainer.innerHTML += `<div class="message assistant">Error: ${error.message}</div>`;
                    }
                    
                    input.disabled = false;
                    input.focus();
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                function clearChat() {
                    document.getElementById('chat-container').innerHTML = '';
                }
                
                document.getElementById('messageInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
            </script>
        </body>
        </html>
        """)


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat endpoint with RAG capabilities."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        # Prepare kwargs for the pipeline
        kwargs = {}
        if message.top_k is not None:
            kwargs["top_k"] = message.top_k
        if message.temperature is not None:
            kwargs["temperature"] = message.temperature
        if message.max_tokens is not None:
            kwargs["max_tokens"] = message.max_tokens
        
        # Get response from RAG pipeline
        response = await rag_pipeline.chat(
            user_message=message.message,
            use_rag=message.use_rag,
            **kwargs
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_immediately: bool = Form(True)
):
    """Upload and process a document."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    # Check file type
    allowed_extensions = {'.pdf', '.html', '.htm', '.docx', '.doc', '.txt'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}"
        )
    
    try:
        # Save uploaded file
        upload_dir = Path(settings.documents_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        # Write file content
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"Uploaded file: {file_path}")
        
        if process_immediately:
            # Process document immediately
            result = await rag_pipeline.add_documents(str(file_path))
            return DocumentUploadResponse(**result)
        else:
            # Add to background processing
            background_tasks.add_task(
                process_document_background, 
                str(file_path)
            )
            
            return DocumentUploadResponse(
                success=True,
                message=f"File uploaded successfully, processing in background",
                chunks_added=0,
                processing_time=0.0,
                sources=[str(file_path)]
            )
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


async def process_document_background(file_path: str):
    """Background task to process uploaded documents."""
    if rag_pipeline:
        try:
            result = await rag_pipeline.add_documents(file_path)
            logger.info(f"Background processing completed: {result}")
        except Exception as e:
            logger.error(f"Background processing failed: {e}")


@app.post("/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    documents_path: str = Form(...),
    process_immediately: bool = Form(True)
):
    """Ingest documents from a directory path."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    documents_path_obj = Path(documents_path)
    if not documents_path_obj.exists():
        raise HTTPException(status_code=400, detail="Documents path does not exist")
    
    try:
        if process_immediately:
            result = await rag_pipeline.add_documents(documents_path)
            return result
        else:
            background_tasks.add_task(
                process_document_background, 
                documents_path
            )
            
            return {
                "success": True,
                "message": "Documents ingestion started in background",
                "chunks_added": 0,
                "processing_time": 0.0,
                "sources": []
            }
            
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.delete("/documents/{source_file:path}")
async def delete_document(source_file: str):
    """Delete a document from the vector store."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        result = await rag_pipeline.delete_documents(source_file)
        return result
        
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        health = await rag_pipeline.health_check()
        return HealthResponse(**health)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        stats = rag_pipeline.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        stats = rag_pipeline.get_stats()
        vector_stats = stats["vector_store"]
        
        # Get unique sources from vector store
        unique_sources = set()
        for chunk in rag_pipeline.vector_store.document_chunks:
            unique_sources.add(chunk.source_file)
        
        return {
            "total_documents": len(unique_sources),
            "total_chunks": vector_stats["total_chunks"],
            "documents": list(unique_sources)
        }
        
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=f"List documents error: {str(e)}")


@app.post("/clear")
async def clear_vector_store():
    """Clear all documents from the vector store."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        rag_pipeline.vector_store.clear()
        rag_pipeline.vector_store.save()
        
        return {
            "success": True,
            "message": "Vector store cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Clear error: {str(e)}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        settings.log_file,
        rotation="10 MB",
        retention="7 days",
        level=settings.log_level
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=settings.log_level,
        colorize=True
    )
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=settings.log_level.lower()
    ) 