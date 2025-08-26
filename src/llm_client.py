"""LM Studio client integration for local LLM inference."""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncIterator
import time

import httpx
from loguru import logger

from config import settings


class LMStudioClient:
    """Client for interacting with LM Studio's local API server."""
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize the LM Studio client.
        
        Args:
            base_url: Base URL for LM Studio API server
            model_name: Name of the model to use
            api_key: API key for authentication
        """
        self.base_url = base_url or settings.lm_studio_base_url
        self.model_name = model_name or settings.lm_studio_model
        self.api_key = api_key or settings.lm_studio_api_key
        
        # Remove trailing slash from base URL
        self.base_url = self.base_url.rstrip('/')
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),  # 2 minute timeout
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info(f"Initialized LM Studio client for {self.model_name} at {self.base_url}")
    
    async def health_check(self) -> bool:
        """
        Check if LM Studio server is healthy and accessible.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on the LM Studio server.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def generate_response(self, 
                              messages: List[Dict[str, str]], 
                              temperature: Optional[float] = None,
                              max_tokens: Optional[int] = None,
                              stream: bool = False) -> Dict[str, Any]:
        """
        Generate a response using the LM Studio model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Response dictionary with generated text
        """
        # Set default parameters
        if temperature is None:
            temperature = settings.temperature
        if max_tokens is None:
            max_tokens = settings.max_tokens
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        try:
            start_time = time.time()
            
            if stream:
                # For streaming, we'll collect all chunks and return the full response
                full_content = ""
                async for chunk in self._generate_streaming_response(payload):
                    if "content" in chunk:
                        full_content += chunk["content"]
                
                end_time = time.time()
                response_time = end_time - start_time
                
                return {
                    "content": full_content,
                    "response_time": response_time,
                    "model": self.model_name,
                    "usage": {},
                    "finish_reason": "stop"
                }
            else:
                response = await self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Calculate timing
                end_time = time.time()
                response_time = end_time - start_time
                
                # Extract the response text
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    return {
                        "content": content,
                        "response_time": response_time,
                        "model": self.model_name,
                        "usage": result.get("usage", {}),
                        "finish_reason": result["choices"][0].get("finish_reason")
                    }
                else:
                    raise ValueError("No valid response generated")
                    
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    async def _generate_streaming_response(self, payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a streaming response.
        
        Args:
            payload: Request payload
            
        Yields:
            Streaming response chunks
        """
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]  # Remove "data: " prefix
                        
                        if chunk_data.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(chunk_data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield {
                                        "content": delta["content"],
                                        "finish_reason": chunk["choices"][0].get("finish_reason")
                                    }
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            raise
    
    async def chat(self, 
                   user_message: str, 
                   system_prompt: Optional[str] = None,
                   conversation_history: Optional[List[Dict[str, str]]] = None,
                   **kwargs) -> str:
        """
        Simple chat interface.
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
            conversation_history: Previous conversation messages
            **kwargs: Additional parameters for generate_response
            
        Returns:
            AI response text
        """
        # Build messages list
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = await self.generate_response(messages, **kwargs)
        return response["content"]
    
    async def rag_chat(self, 
                      question: str,
                      context_chunks: List[str],
                      system_prompt: Optional[str] = None,
                      **kwargs) -> str:
        """
        Chat with RAG context integration.
        
        Args:
            question: User's question
            context_chunks: List of relevant document chunks
            system_prompt: Optional system prompt override
            **kwargs: Additional parameters for generate_response
            
        Returns:
            AI response text with context
        """
        # Default RAG system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
Use the context information to provide accurate and relevant answers. If the context doesn't contain 
enough information to answer the question, say so clearly. Always be concise and factual."""
        
        # Build context string
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create the full prompt
        full_prompt = f"""Context Information:
{context_text}

Question: {question}

Please provide a helpful answer based on the context above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        # Generate response
        response = await self.generate_response(messages, **kwargs)
        return response["content"]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class LMStudioError(Exception):
    """Custom exception for LM Studio related errors."""
    pass


# Utility functions
def create_lm_studio_client(base_url: Optional[str] = None,
                           model_name: Optional[str] = None,
                           api_key: Optional[str] = None) -> LMStudioClient:
    """Create and return an LMStudioClient instance."""
    return LMStudioClient(base_url, model_name, api_key)


async def test_lm_studio_connection(client: Optional[LMStudioClient] = None) -> bool:
    """
    Test the connection to LM Studio server.
    
    Args:
        client: Optional existing client to test
        
    Returns:
        True if connection is successful
    """
    if client is None:
        client = create_lm_studio_client()
    
    try:
        # Test health check
        is_healthy = await client.health_check()
        if not is_healthy:
            logger.error("LM Studio server health check failed")
            return False
        
        # Test model availability
        models = await client.list_models()
        model_names = [model.get("id", "") for model in models]
        
        if client.model_name not in model_names:
            logger.warning(f"Model {client.model_name} not found in available models: {model_names}")
            # Don't return False here as model might still work
        
        # Test simple generation
        test_response = await client.chat("Hello, please respond with 'Connection successful'")
        logger.info(f"Test response: {test_response}")
        
        return True
        
    except Exception as e:
        logger.error(f"LM Studio connection test failed: {e}")
        return False
    finally:
        if client:
            await client.close() 