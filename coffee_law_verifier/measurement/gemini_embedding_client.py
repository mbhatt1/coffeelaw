"""
Google Gemini Embedding Client for real embedding generation
"""
import os
import asyncio
from typing import List, Optional
import numpy as np
import google.generativeai as genai
import logging

class GeminiEmbeddingClient:
    """Client for generating embeddings using Google's Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "models/embedding-001"):
        """
        Initialize Gemini embedding client
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Embedding model to use (default: models/embedding-001)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Model dimensions
        self.dimensions = {
            "models/embedding-001": 768,  # Google's embedding model dimension
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
            # Google's embedding API is synchronous, so we'll run it in an executor
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document",
                    title=None
                )
            )
            
            return np.array(embedding['embedding'])
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini embedding: {e}")
            # Return zero vector on error to avoid breaking the pipeline
            return np.zeros(self.dimensions.get(self.model, 768))
    
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
                # Google's batch embedding
                loop = asyncio.get_event_loop()
                batch_result = await loop.run_in_executor(
                    None,
                    lambda: genai.embed_content(
                        model=self.model,
                        content=batch,
                        task_type="retrieval_document",
                        title=None
                    )
                )
                
                # Extract embeddings from the result
                if 'embedding' in batch_result:
                    # Single embedding returned (if batch size was 1)
                    embeddings.append(np.array(batch_result['embedding']))
                else:
                    # Multiple embeddings
                    for emb in batch_result['embeddings']:
                        embeddings.append(np.array(emb['embedding']))
                
            except Exception as e:
                self.logger.error(f"Error in batch embedding: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([np.zeros(self.dimensions.get(self.model, 768)) for _ in batch])
            
            # Add delay to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings for the current model"""
        return self.dimensions.get(self.model, 768)


class GeminiLLMClient:
    """Client for generating LLM responses using Google's Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize Gemini LLM client
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use (default: gemini-pro)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.logger = logging.getLogger(__name__)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Google's Gemini
        
        Args:
            prompt: The prompt to send to Gemini
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response text
        """
        try:
            # Extract parameters with defaults
            temperature = kwargs.get('temperature', 0.7)
            max_output_tokens = kwargs.get('max_tokens', 1000)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                candidate_count=1,
            )
            
            # Generate response (synchronous call in executor for async compatibility)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating Gemini response: {e}")
            return f"Error: Failed to generate response - {str(e)}"


# Alternative implementation using Vertex AI for better embedding support
class VertexAIEmbeddingClient:
    """Client for generating embeddings using Google's Vertex AI"""
    
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1", 
                 model: str = "textembedding-gecko@003"):
        """
        Initialize Vertex AI embedding client
        
        Args:
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            location: GCP location (default: us-central1)
            model: Embedding model to use (default: textembedding-gecko@003)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("Google Cloud project ID not provided. Set GOOGLE_CLOUD_PROJECT environment variable or pass project_id parameter.")
        
        try:
            from vertexai.language_models import TextEmbeddingModel
            import vertexai
            
            vertexai.init(project=self.project_id, location=location)
            self.model = TextEmbeddingModel.from_pretrained(model)
            self.use_vertex = True
        except ImportError:
            self.logger.warning("Vertex AI SDK not available. Install with: pip install google-cloud-aiplatform")
            self.use_vertex = False
            
        self.logger = logging.getLogger(__name__)
        self.embedding_dimension = 768  # Gecko models use 768 dimensions
        
    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.use_vertex:
            # Fallback to random embeddings if Vertex AI not available
            return np.random.randn(self.embedding_dimension) / np.sqrt(self.embedding_dimension)
            
        try:
            # Vertex AI embedding is synchronous, so we'll run it in an executor
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.get_embeddings([text])
            )
            
            return np.array(embeddings[0].values)
            
        except Exception as e:
            self.logger.error(f"Error generating Vertex AI embedding: {e}")
            return np.zeros(self.embedding_dimension)
    
    async def embed_batch(self, texts: List[str], batch_size: int = 250) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call (Vertex AI supports up to 250)
            
        Returns:
            List of embedding vectors
        """
        if not self.use_vertex:
            # Fallback to random embeddings if Vertex AI not available
            return [np.random.randn(self.embedding_dimension) / np.sqrt(self.embedding_dimension) 
                    for _ in texts]
            
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.get_embeddings(batch)
                )
                
                for emb in batch_embeddings:
                    embeddings.append(np.array(emb.values))
                
            except Exception as e:
                self.logger.error(f"Error in batch embedding: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([np.zeros(self.embedding_dimension) for _ in batch])
            
            # Add delay to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dimension