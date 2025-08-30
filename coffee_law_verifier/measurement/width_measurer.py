"""
Width measurement for ambiguity quantification
"""
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, cdist

class WidthMeasurer:
    """
    Measure ambiguity width W as specified in Coffee Law
    
    W represents the spread of model responses in embedding space
    """
    
    def __init__(self, llm_client: any, embedding_client: any):
        self.llm = llm_client
        self.embedder = embedding_client
        
    def calculate_from_embeddings(self, embeddings: np.ndarray) -> float:
        """
        Calculate ambiguity width from embeddings
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Width measure W
        """
        if len(embeddings) < 2:
            return 0.0
        
        # Method 1: Standard deviation from centroid (as mentioned in README)
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        width_std = distances.std()
        
        return width_std
    
    def calculate_width_advanced(self, embeddings: np.ndarray) -> dict:
        """
        Calculate multiple width measures for robustness
        
        Returns dictionary with different width calculations
        """
        if len(embeddings) < 2:
            return {'std': 0.0, 'mad': 0.0, 'iqr': 0.0, 'pairwise': 0.0}
        
        # Centroid-based standard deviation
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        width_std = distances.std()
        
        # Median absolute deviation (more robust)
        width_mad = np.median(np.abs(distances - np.median(distances)))
        
        # Interquartile range
        width_iqr = np.percentile(distances, 75) - np.percentile(distances, 25)
        
        # Average pairwise distance
        if len(embeddings) > 2:
            pairwise_distances = pdist(embeddings)
            width_pairwise = pairwise_distances.mean()
        else:
            width_pairwise = np.linalg.norm(embeddings[0] - embeddings[1])
        
        # 1-NN distance to gold (if we have a reference)
        # This would be used when we have a known correct answer
        
        return {
            'std': width_std,
            'mad': width_mad,
            'iqr': width_iqr,
            'pairwise': width_pairwise,
            'primary': width_std  # Use std as primary measure
        }
    
    def measure_width_trajectory(self,
                               prompt: str,
                               pe_ctx_values: List[float],
                               n_samples_per_pe: int = 16) -> List[Tuple[float, float]]:
        """
        Measure how width changes with Pe_ctx
        
        Returns list of (Pe_ctx, width) tuples
        """
        trajectory = []
        
        for pe_ctx in pe_ctx_values:
            # Modify prompt based on Pe_ctx (simplified)
            # In practice, this would use ContextVariator
            modified_prompt = self._modify_prompt_for_pe(prompt, pe_ctx)
            
            # Generate samples and measure width
            embeddings = self._generate_embedding_samples(
                modified_prompt, n_samples_per_pe
            )
            
            width = self.calculate_from_embeddings(embeddings)
            trajectory.append((pe_ctx, width))
        
        return trajectory
    
    def _modify_prompt_for_pe(self, prompt: str, pe_ctx: float) -> str:
        """
        Placeholder for prompt modification based on Pe_ctx
        In practice, this would use the ContextVariator
        """
        # Higher Pe_ctx = more structure, less noise
        if pe_ctx > 5:
            return f"[STRUCTURED]\n{prompt}\n[END STRUCTURED]"
        elif pe_ctx > 1:
            return f"{prompt}"
        else:
            # Add noise for low Pe_ctx
            import random
            noise_words = ['perhaps', 'maybe', 'possibly', 'uncertain']
            noisy_prompt = prompt
            for _ in range(int(5 / pe_ctx)):
                word = random.choice(noise_words)
                noisy_prompt += f" {word}"
            return noisy_prompt
    
    def _generate_embedding_samples(self, prompt: str, n_samples: int) -> np.ndarray:
        """
        Generate multiple response embeddings for a prompt
        
        In production, this would call the actual LLM and embedding APIs
        """
        # Placeholder implementation
        embedding_dim = 384
        
        # Simulate variation based on prompt characteristics
        base_embedding = np.random.randn(embedding_dim)
        
        # Add noise to simulate response variation
        # Less structured prompts = more variation
        noise_scale = 0.1 if '[STRUCTURED]' in prompt else 0.3
        
        embeddings = []
        for _ in range(n_samples):
            noise = np.random.randn(embedding_dim) * noise_scale
            embeddings.append(base_embedding + noise)
        
        return np.array(embeddings)
    
    def calculate_relative_width(self,
                               test_embeddings: np.ndarray,
                               reference_embeddings: np.ndarray) -> float:
        """
        Calculate width relative to a reference set
        
        Useful for comparing across different tasks
        """
        test_width = self.calculate_from_embeddings(test_embeddings)
        ref_width = self.calculate_from_embeddings(reference_embeddings)
        
        if ref_width > 0:
            return test_width / ref_width
        return float('inf')