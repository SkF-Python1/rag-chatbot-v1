"""RAG pipeline implementation combining retrieval and generation."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from loguru import logger

from config import settings
from src.document_processor import DocumentProcessor, DocumentChunk
from src.vector_store import VectorStore
from src.llm_client import LMStudioClient


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    response_time: float
    retrieval_time: float
    generation_time: float
    context_chunks: List[str]
    confidence_score: float


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval and generation."""
    
    def __init__(self, 
                 vector_store: Optional[VectorStore] = None,
                 llm_client: Optional[LMStudioClient] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store for document retrieval
            llm_client: LM Studio client for text generation
        """
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or LMStudioClient()
        
        logger.info("RAG pipeline initialized")
    
    async def query(self, 
                   question: str, 
                   top_k: Optional[int] = None,
                   filters: Optional[Dict[str, Any]] = None,
                   system_prompt: Optional[str] = None,
                   **llm_kwargs) -> RAGResponse:
        """
        Execute a complete RAG query.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            filters: Optional filters for retrieval
            system_prompt: Optional system prompt override
            **llm_kwargs: Additional arguments for LLM generation
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        logger.info(f"Retrieving documents for question: {question}")
        retrieval_start = time.time()
        
        if filters:
            retrieved_docs = self.vector_store.search_by_filter(
                question, filters, top_k or settings.top_k_results
            )
        else:
            retrieved_docs = self.vector_store.search(
                question, top_k or settings.top_k_results
            )
        
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                response_time=time.time() - start_time,
                retrieval_time=retrieval_time,
                generation_time=0.0,
                context_chunks=[],
                confidence_score=0.0
            )
        
        # Step 2: Prepare context chunks
        context_chunks = []
        sources = []
        
        for chunk, similarity_score in retrieved_docs:
            context_chunks.append(chunk.content)
            sources.append({
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "similarity_score": float(similarity_score),
                "metadata": chunk.metadata
            })
        
        logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
        
        # Step 3: Generate response using LLM
        generation_start = time.time()
        
        try:
            answer = await self.llm_client.rag_chat(
                question=question,
                context_chunks=context_chunks,
                system_prompt=system_prompt,
                **llm_kwargs
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"I encountered an error while generating the response: {str(e)}"
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Calculate confidence score based on similarity scores
        confidence_score = self._calculate_confidence_score(retrieved_docs)
        
        logger.info(f"RAG query completed in {total_time:.2f}s")
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            response_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            context_chunks=context_chunks,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence_score(self, retrieved_docs: List[Tuple[DocumentChunk, float]]) -> float:
        """
        Calculate confidence score based on retrieval results.
        
        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        # Use the highest similarity score as primary indicator
        max_similarity = max(score for _, score in retrieved_docs)
        
        # Consider the number of relevant documents
        num_docs = len(retrieved_docs)
        
        # Weight the confidence based on both similarity and number of docs
        confidence = (max_similarity * 0.7) + (min(num_docs / 5, 1.0) * 0.3)
        
        return min(confidence, 1.0)
    
    async def chat(self, 
                  user_message: str,
                  conversation_history: Optional[List[Dict[str, str]]] = None,
                  use_rag: bool = True,
                  **kwargs) -> Dict[str, Any]:
        """
        Chat interface with optional RAG.
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation messages
            use_rag: Whether to use RAG for this query
            **kwargs: Additional arguments for query/generation
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        if use_rag:
            rag_response = await self.query(user_message, **kwargs)
            return {
                "response": rag_response.answer,
                "sources": rag_response.sources,
                "response_time": rag_response.response_time,
                "retrieval_time": rag_response.retrieval_time,
                "generation_time": rag_response.generation_time,
                "confidence_score": rag_response.confidence_score,
                "context_chunks": len(rag_response.context_chunks),
                "use_rag": True
            }
        else:
            # Direct chat without RAG
            try:
                response = await self.llm_client.chat(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    **kwargs
                )
                
                return {
                    "response": response,
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "retrieval_time": 0.0,
                    "generation_time": time.time() - start_time,
                    "confidence_score": 0.5,  # Default for non-RAG responses
                    "context_chunks": 0,
                    "use_rag": False
                }
            except Exception as e:
                logger.error(f"Chat generation failed: {e}")
                return {
                    "response": f"I encountered an error: {str(e)}",
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "retrieval_time": 0.0,
                    "generation_time": time.time() - start_time,
                    "confidence_score": 0.0,
                    "context_chunks": 0,
                    "use_rag": False
                }
    
    async def add_documents(self, documents_path: str) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents_path: Path to documents directory or single file
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Adding documents from: {documents_path}")
        start_time = time.time()
        
        # Process documents
        processor = DocumentProcessor()
        
        try:
            if documents_path.endswith(('.pdf', '.html', '.htm', '.docx', '.doc', '.txt')):
                # Single file
                chunks = processor.process_file(documents_path)
            else:
                # Directory
                chunks = processor.process_directory(documents_path)
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No documents processed",
                    "chunks_added": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            # Save vector store
            self.vector_store.save()
            
            processing_time = time.time() - start_time
            
            logger.info(f"Added {len(chunks)} chunks in {processing_time:.2f}s")
            
            return {
                "success": True,
                "message": f"Successfully processed {len(chunks)} document chunks",
                "chunks_added": len(chunks),
                "processing_time": processing_time,
                "sources": list(set(chunk.source_file for chunk in chunks))
            }
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "success": False,
                "message": f"Error processing documents: {str(e)}",
                "chunks_added": 0,
                "processing_time": time.time() - start_time
            }
    
    async def delete_documents(self, source_file: str) -> Dict[str, Any]:
        """
        Delete documents from the vector store.
        
        Args:
            source_file: Path to the source file to delete
            
        Returns:
            Dictionary with deletion results
        """
        logger.info(f"Deleting documents from: {source_file}")
        start_time = time.time()
        
        try:
            deleted_count = self.vector_store.delete_documents_by_source(source_file)
            
            if deleted_count > 0:
                # Save updated vector store
                self.vector_store.save()
                
                return {
                    "success": True,
                    "message": f"Deleted {deleted_count} document chunks",
                    "deleted_count": deleted_count,
                    "processing_time": time.time() - start_time
                }
            else:
                return {
                    "success": False,
                    "message": "No documents found to delete",
                    "deleted_count": 0,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return {
                "success": False,
                "message": f"Error deleting documents: {str(e)}",
                "deleted_count": 0,
                "processing_time": time.time() - start_time
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        vector_stats = self.vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "llm_model": self.llm_client.model_name,
            "llm_base_url": self.llm_client.base_url,
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k_results": settings.top_k_results,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of all pipeline components.
        
        Returns:
            Dictionary with health status
        """
        logger.info("Running RAG pipeline health check")
        
        # Check vector store
        vector_stats = self.vector_store.get_stats()
        vector_healthy = vector_stats["total_chunks"] >= 0
        
        # Check LLM client
        llm_healthy = await self.llm_client.health_check()
        
        overall_healthy = vector_healthy and llm_healthy
        
        return {
            "healthy": overall_healthy,
            "vector_store": {
                "healthy": vector_healthy,
                "total_chunks": vector_stats["total_chunks"],
                "total_sources": vector_stats["total_sources"]
            },
            "llm_client": {
                "healthy": llm_healthy,
                "model": self.llm_client.model_name,
                "base_url": self.llm_client.base_url
            }
        }
    
    async def close(self):
        """Close the RAG pipeline and cleanup resources."""
        await self.llm_client.close()
        logger.info("RAG pipeline closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Utility functions
def create_rag_pipeline(vector_store: Optional[VectorStore] = None,
                       llm_client: Optional[LMStudioClient] = None) -> RAGPipeline:
    """Create and return a RAGPipeline instance."""
    return RAGPipeline(vector_store, llm_client)


async def test_rag_pipeline(pipeline: Optional[RAGPipeline] = None) -> bool:
    """
    Test the RAG pipeline functionality.
    
    Args:
        pipeline: Optional existing pipeline to test
        
    Returns:
        True if pipeline is working correctly
    """
    if pipeline is None:
        pipeline = create_rag_pipeline()
    
    try:
        # Test health check
        health = await pipeline.health_check()
        if not health["healthy"]:
            logger.error("RAG pipeline health check failed")
            return False
        
        # Test basic query (will work even without documents)
        response = await pipeline.chat("Hello, how are you?", use_rag=False)
        if not response["response"]:
            logger.error("RAG pipeline chat test failed")
            return False
        
        logger.info("RAG pipeline test successful")
        return True
        
    except Exception as e:
        logger.error(f"RAG pipeline test failed: {e}")
        return False
    finally:
        if pipeline:
            await pipeline.close() 