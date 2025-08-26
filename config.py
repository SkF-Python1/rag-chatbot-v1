"""Configuration settings for the RAG Chatbot system."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # LM Studio Configuration
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "qwen/qwen3-14b"
    lm_studio_api_key: str = "lm-studio"
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_reload: bool = True
    
    # Document Storage
    documents_path: str = "./documents"
    vector_store_path: str = "./vector_store"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 2048
    temperature: float = 0.1
    top_k_results: int = 5
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure required directories exist
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.documents_path,
        settings.vector_store_path,
        Path(settings.log_file).parent,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


# Initialize directories on import
create_directories() 