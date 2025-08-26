#!/usr/bin/env python3
"""Document ingestion script for RAG system."""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import time

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from config import settings
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore, create_vector_store
from src.llm_client import LMStudioClient, test_lm_studio_connection


class DocumentIngester:
    """Main document ingestion class."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize the document ingester."""
        self.vector_store = vector_store or create_vector_store()
        self.processor = DocumentProcessor()
        
    async def ingest_documents(self, 
                             documents_path: str, 
                             recursive: bool = True,
                             file_patterns: Optional[List[str]] = None,
                             clear_existing: bool = False) -> dict:
        """
        Ingest documents from a path.
        
        Args:
            documents_path: Path to documents (file or directory)
            recursive: Search recursively in subdirectories
            file_patterns: List of file patterns to include
            clear_existing: Clear existing vector store before ingesting
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        
        logger.info(f"Starting document ingestion from: {documents_path}")
        
        # Clear existing documents if requested
        if clear_existing:
            logger.info("Clearing existing vector store...")
            self.vector_store.clear()
        
        # Process documents
        documents_path_obj = Path(documents_path)
        
        if not documents_path_obj.exists():
            error_msg = f"Path does not exist: {documents_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "total_chunks": 0,
                "total_files": 0,
                "processing_time": 0.0,
                "files_processed": []
            }
        
        try:
            if documents_path_obj.is_file():
                # Single file processing
                chunks = self.processor.process_file(str(documents_path_obj))
                files_processed = [str(documents_path_obj)]
            else:
                # Directory processing
                chunks = self.processor.process_directory(str(documents_path_obj))
                
                # Get list of processed files
                files_processed = list(set(chunk.source_file for chunk in chunks))
            
            if not chunks:
                logger.warning("No documents were processed")
                return {
                    "success": False,
                    "message": "No documents were processed",
                    "total_chunks": 0,
                    "total_files": 0,
                    "processing_time": time.time() - start_time,
                    "files_processed": []
                }
            
            logger.info(f"Processed {len(chunks)} chunks from {len(files_processed)} files")
            
            # Add to vector store
            logger.info("Adding documents to vector store...")
            self.vector_store.add_documents(chunks)
            
            # Save vector store
            logger.info("Saving vector store...")
            self.vector_store.save()
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document ingestion completed in {processing_time:.2f} seconds")
            
            return {
                "success": True,
                "message": f"Successfully ingested {len(chunks)} chunks from {len(files_processed)} files",
                "total_chunks": len(chunks),
                "total_files": len(files_processed),
                "processing_time": processing_time,
                "files_processed": files_processed
            }
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            return {
                "success": False,
                "message": f"Error during ingestion: {str(e)}",
                "total_chunks": 0,
                "total_files": 0,
                "processing_time": time.time() - start_time,
                "files_processed": []
            }
    
    def get_stats(self) -> dict:
        """Get current vector store statistics."""
        return self.vector_store.get_stats()
    
    def list_documents(self) -> List[str]:
        """List all documents in the vector store."""
        unique_sources = set()
        for chunk in self.vector_store.document_chunks:
            unique_sources.add(chunk.source_file)
        return list(unique_sources)
    
    async def delete_document(self, source_file: str) -> dict:
        """Delete a document from the vector store."""
        logger.info(f"Deleting document: {source_file}")
        
        try:
            deleted_count = self.vector_store.delete_documents_by_source(source_file)
            
            if deleted_count > 0:
                self.vector_store.save()
                logger.info(f"Deleted {deleted_count} chunks from {source_file}")
                return {
                    "success": True,
                    "message": f"Deleted {deleted_count} chunks",
                    "deleted_count": deleted_count
                }
            else:
                logger.warning(f"No chunks found for {source_file}")
                return {
                    "success": False,
                    "message": "No chunks found to delete",
                    "deleted_count": 0
                }
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {
                "success": False,
                "message": f"Error deleting document: {str(e)}",
                "deleted_count": 0
            }
    
    async def clear_all(self) -> dict:
        """Clear all documents from the vector store."""
        logger.info("Clearing all documents from vector store...")
        
        try:
            self.vector_store.clear()
            self.vector_store.save()
            
            logger.info("All documents cleared successfully")
            return {
                "success": True,
                "message": "All documents cleared successfully"
            }
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return {
                "success": False,
                "message": f"Error clearing documents: {str(e)}"
            }


async def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Document ingestion script for RAG system")
    parser.add_argument("command", choices=["ingest", "list", "delete", "clear", "stats", "test"], 
                       help="Command to execute")
    parser.add_argument("--path", "-p", type=str, help="Path to documents or specific file")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, 
                       help="Search recursively in subdirectories")
    parser.add_argument("--clear-existing", "-c", action="store_true", 
                       help="Clear existing vector store before ingesting")
    parser.add_argument("--file", "-f", type=str, help="Specific file to delete")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        colorize=True
    )
    
    # Create ingester
    ingester = DocumentIngester()
    
    if args.command == "ingest":
        if not args.path:
            logger.error("Path is required for ingest command")
            sys.exit(1)
        
        result = await ingester.ingest_documents(
            documents_path=args.path,
            recursive=args.recursive,
            clear_existing=args.clear_existing
        )
        
        if result["success"]:
            logger.info(f"✅ {result['message']}")
            logger.info(f"📊 Total chunks: {result['total_chunks']}")
            logger.info(f"📁 Total files: {result['total_files']}")
            logger.info(f"⏱️  Processing time: {result['processing_time']:.2f}s")
        else:
            logger.error(f"❌ {result['message']}")
            sys.exit(1)
    
    elif args.command == "list":
        documents = ingester.list_documents()
        stats = ingester.get_stats()
        
        logger.info(f"📁 Total documents: {len(documents)}")
        logger.info(f"📊 Total chunks: {stats['total_chunks']}")
        logger.info(f"📏 Embedding dimension: {stats['embedding_dimension']}")
        
        if documents:
            logger.info("\n📝 Documents:")
            for doc in sorted(documents):
                logger.info(f"  • {doc}")
        else:
            logger.info("No documents found")
    
    elif args.command == "delete":
        if not args.file:
            logger.error("File path is required for delete command")
            sys.exit(1)
        
        result = await ingester.delete_document(args.file)
        
        if result["success"]:
            logger.info(f"✅ {result['message']}")
        else:
            logger.error(f"❌ {result['message']}")
            sys.exit(1)
    
    elif args.command == "clear":
        confirm = input("Are you sure you want to clear all documents? (y/N): ")
        if confirm.lower() == 'y':
            result = await ingester.clear_all()
            
            if result["success"]:
                logger.info(f"✅ {result['message']}")
            else:
                logger.error(f"❌ {result['message']}")
                sys.exit(1)
        else:
            logger.info("Operation cancelled")
    
    elif args.command == "stats":
        stats = ingester.get_stats()
        
        logger.info("📊 Vector Store Statistics:")
        logger.info(f"  • Total chunks: {stats['total_chunks']}")
        logger.info(f"  • Total sources: {stats['total_sources']}")
        logger.info(f"  • Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"  • Index size: {stats['index_size']}")
        logger.info(f"  • Embedding model: {stats['embedding_model']}")
        logger.info(f"  • Device: {stats['device']}")
    
    elif args.command == "test":
        logger.info("🔧 Testing system components...")
        
        # Test vector store
        stats = ingester.get_stats()
        logger.info(f"✅ Vector store: {stats['total_chunks']} chunks loaded")
        
        # Test LM Studio connection
        try:
            connection_ok = await test_lm_studio_connection()
            if connection_ok:
                logger.info("✅ LM Studio connection: OK")
            else:
                logger.error("❌ LM Studio connection: FAILED")
        except Exception as e:
            logger.error(f"❌ LM Studio connection error: {e}")
        
        # Test document processing
        test_text = "This is a test document for processing."
        try:
            processor = DocumentProcessor()
            # Create a temporary test file
            test_file = Path("test_doc.txt")
            test_file.write_text(test_text)
            
            chunks = processor.process_file(str(test_file))
            test_file.unlink()  # Clean up
            
            if chunks:
                logger.info(f"✅ Document processing: {len(chunks)} chunks created")
            else:
                logger.error("❌ Document processing: No chunks created")
        except Exception as e:
            logger.error(f"❌ Document processing error: {e}")
        
        logger.info("🔧 System test completed")


if __name__ == "__main__":
    asyncio.run(main()) 