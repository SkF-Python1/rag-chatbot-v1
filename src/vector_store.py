"""FAISS vector store implementation for document retrieval."""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import settings
from src.document_processor import DocumentChunk


class VectorStore:
    """FAISS-based vector store for document embeddings."""
    
    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.device = settings.embedding_device
        self.vector_store_path = Path(settings.vector_store_path)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=self.device
        )
        
        # Get embedding dimension
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Initialize with proper index
        self.document_chunks = []
        self.chunk_metadata = {}
        
        # File paths for persistence
        self.index_file = self.vector_store_path / "faiss_index.bin"
        self.chunks_file = self.vector_store_path / "chunks.pkl"
        self.metadata_file = self.vector_store_path / "metadata.json"
        
        # Load existing data if available
        self._load_existing_data()
    
    def _load_existing_data(self) -> None:
        """Load existing vector store data from disk."""
        try:
            if (self.index_file.exists() and 
                self.chunks_file.exists() and 
                self.metadata_file.exists()):
                
                logger.info("Loading existing vector store data...")
                
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_file))
                
                # Load document chunks
                with open(self.chunks_file, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                
                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    self.chunk_metadata = json.load(f)
                
                logger.info(f"Loaded {len(self.document_chunks)} document chunks")
                
            else:
                logger.info("No existing vector store found, creating new one")
                self._initialize_new_index()
                
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            logger.info("Creating new vector store")
            self._initialize_new_index()
    
    def _initialize_new_index(self) -> None:
        """Initialize a new FAISS index."""
        # Create a flat (brute force) index for exact search
        # For large datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.document_chunks = []
        self.chunk_metadata = {}
        logger.info(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to add
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        logger.info(f"Adding {len(chunks)} document chunks to vector store")
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store document chunks and metadata
        for chunk in chunks:
            chunk_id = str(uuid.uuid4()) if not hasattr(chunk, 'chunk_id') else chunk.chunk_id
            self.document_chunks.append(chunk)
            self.chunk_metadata[chunk_id] = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index
            }
        
        logger.info(f"Added {len(chunks)} chunks. Total chunks: {len(self.document_chunks)}")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of tuples (DocumentChunk, similarity_score)
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        if not self.document_chunks:
            logger.warning("No documents in vector store")
            return []
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS index
        similarities, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(top_k, len(self.document_chunks))
        )
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # Valid index
                chunk = self.document_chunks[idx]
                results.append((chunk, float(similarity)))
        
        logger.info(f"Found {len(results)} similar documents for query")
        return results
    
    def search_by_filter(self, 
                        query: str, 
                        filters: Optional[Dict[str, Any]] = None, 
                        top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search with metadata filters.
        
        Args:
            query: Search query text
            filters: Dictionary of metadata filters to apply
            top_k: Number of top results to return
            
        Returns:
            List of tuples (DocumentChunk, similarity_score)
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        # Get all search results first
        all_results = self.search(query, len(self.document_chunks))
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for chunk, score in all_results:
                if self._matches_filters(chunk, filters):
                    filtered_results.append((chunk, score))
            return filtered_results[:top_k]
        
        return all_results[:top_k]
    
    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """
        Check if a chunk matches the given filters.
        
        Args:
            chunk: DocumentChunk to check
            filters: Filters to apply
            
        Returns:
            True if chunk matches all filters
        """
        for key, value in filters.items():
            if key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
            else:
                return False
        return True
    
    def save(self) -> None:
        """Save the vector store to disk."""
        try:
            # Ensure directory exists
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save document chunks
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2)
            
            logger.info(f"Vector store saved to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def delete_documents_by_source(self, source_file: str) -> int:
        """
        Delete all documents from a specific source file.
        
        Args:
            source_file: Path to the source file
            
        Returns:
            Number of deleted documents
        """
        indices_to_remove = []
        
        for i, chunk in enumerate(self.document_chunks):
            if chunk.source_file == source_file:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            logger.info(f"No documents found for source: {source_file}")
            return 0
        
        # Remove from document chunks (in reverse order to maintain indices)
        for i in reversed(indices_to_remove):
            removed_chunk = self.document_chunks.pop(i)
            # Remove from metadata
            if removed_chunk.chunk_id in self.chunk_metadata:
                del self.chunk_metadata[removed_chunk.chunk_id]
        
        # Rebuild FAISS index (expensive operation)
        self._rebuild_index()
        
        logger.info(f"Deleted {len(indices_to_remove)} documents from {source_file}")
        return len(indices_to_remove)
    
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from existing chunks."""
        if not self.document_chunks:
            self._initialize_new_index()
            return
        
        logger.info("Rebuilding FAISS index...")
        
        # Create new index
        self._initialize_new_index()
        
        # Re-add all chunks
        texts = [chunk.content for chunk in self.document_chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        self.index.add(embeddings.astype(np.float32))
        logger.info("FAISS index rebuilt successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        total_chunks = len(self.document_chunks)
        
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_sources": 0,
                "embedding_dimension": self.embedding_dim,
                "index_size": 0
            }
        
        # Count unique sources
        sources = set(chunk.source_file for chunk in self.document_chunks)
        
        # Calculate index size
        index_size = self.index.ntotal if self.index else 0
        
        return {
            "total_chunks": total_chunks,
            "total_sources": len(sources),
            "embedding_dimension": self.embedding_dim,
            "index_size": index_size,
            "embedding_model": self.embedding_model_name,
            "device": self.device
        }
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self._initialize_new_index()
        self.chunk_metadata = {}
        logger.info("Vector store cleared")


# Utility function for easy import
def create_vector_store(embedding_model: Optional[str] = None) -> VectorStore:
    """Create and return a VectorStore instance."""
    return VectorStore(embedding_model) 