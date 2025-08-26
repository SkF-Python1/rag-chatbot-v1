#!/usr/bin/env python3
"""Setup script for RAG Chatbot system."""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🤖 RAG Chatbot Setup")
    print("=" * 60)
    print("Setting up your local RAG chatbot with LM Studio & FAISS")
    print()

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "documents",
        "vector_store", 
        "logs",
        "static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}/")
    
    print("✅ Directories created successfully!")
    return True

def create_env_file():
    """Create .env file with default settings."""
    print("⚙️  Creating .env file...")
    
    env_content = """# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=mistralai/devstral-small-2507
LM_STUDIO_API_KEY=lm-studio

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true

# Document Storage
DOCUMENTS_PATH=./documents
VECTOR_STORE_PATH=./vector_store

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=2048
TEMPERATURE=0.1
TOP_K_RESULTS=5

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(env_content)
        print("✅ .env file created with default settings")
    else:
        print("ℹ️  .env file already exists, skipping...")
    
    return True

def check_lm_studio():
    """Check if LM Studio is running."""
    print("🔍 Checking LM Studio connection...")
    
    try:
        import httpx
        import asyncio
        
        async def test_connection():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:1234/v1/models")
                    return response.status_code == 200
            except:
                return False
        
        is_running = asyncio.run(test_connection())
        
        if is_running:
            print("✅ LM Studio is running and accessible")
            return True
        else:
            print("⚠️  LM Studio is not running or not accessible")
            print("   Please start LM Studio and load the mistralai/devstral-small-2507 model")
            return False
            
    except ImportError:
        print("⚠️  Cannot check LM Studio (httpx not installed yet)")
        return False

def create_sample_document():
    """Create a sample document for testing."""
    print("📄 Creating sample document...")
    
    sample_content = """# RAG Chatbot Documentation

## Overview
This is a Retrieval-Augmented Generation (RAG) chatbot system that combines:
- LM Studio for local language model inference
- FAISS for efficient vector similarity search
- FastAPI for web API endpoints
- Modern web interface for chat interactions

## Features
- 🚀 High Performance: Optimized for low latency (<1.2s response time)
- 📚 Multi-Format Support: PDF, HTML, Word documents
- 🔍 Semantic Search: FAISS vector store for efficient document retrieval
- 🤖 Local LLM: Integration with LM Studio and Mistral model
- 🌐 Web Interface: FastAPI-based REST API with chat interface
- 🔒 Privacy First: Complete local setup, no external API calls
- 📈 Scalable: Handles 10K+ documents efficiently

## Getting Started
1. Install dependencies: pip install -r requirements.txt
2. Start LM Studio and load the mistralai/devstral-small-2507 model
3. Run the application: python main.py
4. Open your browser to http://localhost:8000

## Usage
- Upload documents through the web interface
- Ask questions about your documents
- The system will retrieve relevant information and generate responses
- View sources and confidence scores for each answer

## Configuration
Edit the .env file to customize:
- LM Studio connection settings
- Document processing parameters
- Vector store configuration
- API server settings

This is a sample document to test the RAG system functionality.
"""
    
    sample_path = Path("documents/sample_documentation.txt")
    if not sample_path.exists():
        sample_path.write_text(sample_content)
        print("✅ Sample document created: documents/sample_documentation.txt")
    else:
        print("ℹ️  Sample document already exists, skipping...")
    
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Start LM Studio and load the mistralai/devstral-small-2507 model")
    print("2. Run the application:")
    print("   python main.py")
    print()
    print("3. Open your browser to: http://localhost:8000")
    print()
    print("4. Optional: Ingest documents using the script:")
    print("   python scripts/ingest_documents.py ingest --path ./documents")
    print()
    print("5. Start chatting with your documents!")
    print()
    print("Need help? Check the README.md file for detailed instructions.")
    print()

def main():
    """Main setup function."""
    print_banner()
    
    success = True
    
    # Step 1: Install dependencies
    if not install_dependencies():
        success = False
    
    # Step 2: Create directories
    if not create_directories():
        success = False
    
    # Step 3: Create .env file
    if not create_env_file():
        success = False
    
    # Step 4: Create sample document
    if not create_sample_document():
        success = False
    
    # Step 5: Check LM Studio (optional)
    check_lm_studio()
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup encountered some issues. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 