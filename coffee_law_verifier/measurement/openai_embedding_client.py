"""
OpenAI Embedding Client for real embedding generation
"""
import os
import asyncio
from typing import List, Optional
import numpy as np
import openai
from openai import AsyncOpenAI
import logging

class OpenAIEmbeddingClient:
    """Client for generating embeddings using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding client
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return zero vector on error to avoid breaking the pipeline
            return np.zeros(self.dimensions.get(self.model, 1536))
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Error in batch embedding: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([np.zeros(self.dimensions.get(self.model, 1536)) for _ in batch])
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        return self.dimensions.get(self.model, 1536)


class OpenAILLMClient:
    """Client for generating LLM responses using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI LLM client
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-3.5-turbo)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI's models
        
        Args:
            prompt: The prompt to send to OpenAI
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response text
        """
        try:
            # Extract parameters with defaults
            temperature = kwargs.get('temperature', 0.7)
            max_tokens = kwargs.get('max_tokens', 1000)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating OpenAI response: {e}")
            return f"Error: Failed to generate response - {str(e)}"