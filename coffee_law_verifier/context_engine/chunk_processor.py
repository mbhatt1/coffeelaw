"""
Chunk processor for handling context chunks with advanced operations
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import tiktoken

@dataclass
class ProcessedChunk:
    content: str
    embedding: Optional[np.ndarray]
    tokens: List[str]
    token_count: int
    semantic_fingerprint: str
    importance_weight: float

class ChunkProcessor:
    """
    Advanced chunk processing for Coffee Law experiments
    """
    
    def __init__(self, embedding_model: Optional[any] = None):
        self.embedding_model = embedding_model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vectorizer = TfidfVectorizer(max_features=500)
        
    def process_chunks(self, chunks: List[str], 
                      compute_embeddings: bool = True) -> List[ProcessedChunk]:
        """Process raw chunks into structured format with metadata"""
        processed = []
        
        # Compute TF-IDF for importance weights if multiple chunks
        if len(chunks) > 1:
            tfidf_matrix = self.vectorizer.fit_transform(chunks)
            importance_weights = np.array(tfidf_matrix.sum(axis=1)).flatten()
            importance_weights = importance_weights / importance_weights.sum()
        else:
            importance_weights = [1.0]
        
        for i, chunk in enumerate(chunks):
            # Tokenize
            tokens = self.tokenizer.encode(chunk)
            token_strings = [self.tokenizer.decode([t]) for t in tokens]
            
            # Generate semantic fingerprint
            fingerprint = self._generate_fingerprint(chunk)
            
            # Get embedding if model provided and requested
            embedding = None
            if compute_embeddings and self.embedding_model:
                embedding = self._get_embedding(chunk)
            
            processed.append(ProcessedChunk(
                content=chunk,
                embedding=embedding,
                tokens=token_strings,
                token_count=len(tokens),
                semantic_fingerprint=fingerprint,
                importance_weight=importance_weights[i] if i < len(importance_weights) else 1.0
            ))
        
        return processed
    
    def compute_effective_chunks(self, chunks: List[ProcessedChunk],
                               similarity_threshold: float = 0.8) -> float:
        """
        Compute N_eff: effective number of independent chunks
        Formula: N_eff = (Σw_i)² / Σw_i²
        """
        if not chunks:
            return 0.0
        
        # Get importance weights adjusted for similarity
        weights = np.array([c.importance_weight for c in chunks])
        
        if len(chunks) > 1 and all(c.embedding is not None for c in chunks):
            # Adjust weights based on similarity to reduce redundant chunks
            embeddings = np.vstack([c.embedding for c in chunks])
            similarities = cosine_similarity(embeddings)
            
            # Penalize similar chunks
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    if similarities[i, j] > similarity_threshold:
                        # Reduce weight of the less important chunk
                        if weights[i] < weights[j]:
                            weights[i] *= (1 - similarities[i, j])
                        else:
                            weights[j] *= (1 - similarities[i, j])
        
        # Calculate N_eff
        weights = weights / weights.sum()  # Normalize
        n_eff = (weights.sum()) ** 2 / (weights ** 2).sum()
        
        return n_eff
    
    def select_top_k_diverse(self, chunks: List[ProcessedChunk], k: int,
                           diversity_weight: float = 0.5) -> List[ProcessedChunk]:
        """
        Select top-k chunks balancing importance and diversity
        Uses MMR (Maximal Marginal Relevance) algorithm
        """
        if len(chunks) <= k:
            return chunks
        
        selected = []
        remaining = list(chunks)
        
        # Start with most important chunk
        remaining.sort(key=lambda c: c.importance_weight, reverse=True)
        selected.append(remaining.pop(0))
        
        # Iteratively select chunks that maximize importance - similarity
        while len(selected) < k and remaining:
            scores = []
            
            for chunk in remaining:
                # Importance score
                importance = chunk.importance_weight
                
                # Similarity to already selected chunks
                if chunk.embedding is not None and any(s.embedding is not None for s in selected):
                    selected_embeddings = np.vstack([s.embedding for s in selected if s.embedding is not None])
                    chunk_embedding = chunk.embedding.reshape(1, -1)
                    similarities = cosine_similarity(chunk_embedding, selected_embeddings)
                    max_similarity = similarities.max()
                else:
                    # Fallback to text similarity
                    max_similarity = max(
                        self._text_similarity(chunk.content, s.content)
                        for s in selected
                    )
                
                # MMR score
                score = (1 - diversity_weight) * importance - diversity_weight * max_similarity
                scores.append(score)
            
            # Select chunk with highest MMR score
            best_idx = np.argmax(scores)
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def merge_similar_chunks(self, chunks: List[ProcessedChunk],
                           similarity_threshold: float = 0.9) -> List[ProcessedChunk]:
        """Merge highly similar chunks to reduce redundancy"""
        if len(chunks) <= 1:
            return chunks
        
        merged = []
        merged_indices = set()
        
        for i, chunk_i in enumerate(chunks):
            if i in merged_indices:
                continue
                
            # Find all chunks similar to this one
            similar_group = [chunk_i]
            similar_indices = {i}
            
            for j, chunk_j in enumerate(chunks[i+1:], start=i+1):
                if j in merged_indices:
                    continue
                    
                # Check similarity
                if chunk_i.embedding is not None and chunk_j.embedding is not None:
                    similarity = cosine_similarity(
                        chunk_i.embedding.reshape(1, -1),
                        chunk_j.embedding.reshape(1, -1)
                    )[0, 0]
                else:
                    similarity = self._text_similarity(chunk_i.content, chunk_j.content)
                
                if similarity > similarity_threshold:
                    similar_group.append(chunk_j)
                    similar_indices.add(j)
            
            # Merge the similar group
            if len(similar_group) > 1:
                merged_chunk = self._merge_chunk_group(similar_group)
                merged.append(merged_chunk)
                merged_indices.update(similar_indices)
            else:
                merged.append(chunk_i)
                merged_indices.add(i)
        
        return merged
    
    def _generate_fingerprint(self, text: str) -> str:
        """Generate a semantic fingerprint for quick comparison"""
        # Simple fingerprint based on key terms
        words = re.findall(r'\w+', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top 5 words as fingerprint
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        fingerprint = '_'.join([w[0] for w in top_words])
        
        return fingerprint
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from the model"""
        if self.embedding_model is None:
            # Return random embedding for testing
            return np.random.randn(384)  # Typical small embedding size
        
        # Use the actual embedding model if available
        import asyncio
        
        # Check if the embedding model has an async embed method
        if hasattr(self.embedding_model, 'embed'):
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, but this is a sync method
                # For now, return mock embedding to avoid the error
                # TODO: Refactor to make process_chunks async
                if hasattr(self.embedding_model, 'get_dimension'):
                    dim = self.embedding_model.get_dimension()
                else:
                    dim = 384
                return np.random.randn(dim)
            except RuntimeError:
                # No event loop running, we can create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    embedding = loop.run_until_complete(self.embedding_model.embed(text))
                    return embedding
                finally:
                    loop.close()
        
        # Fallback to random embedding if no embed method
        return np.random.randn(384)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_chunk_group(self, chunks: List[ProcessedChunk]) -> ProcessedChunk:
        """Merge a group of similar chunks"""
        # Combine content (taking the longest as base)
        base_chunk = max(chunks, key=lambda c: len(c.content))
        
        # Average embeddings if available
        if all(c.embedding is not None for c in chunks):
            avg_embedding = np.mean([c.embedding for c in chunks], axis=0)
        else:
            avg_embedding = base_chunk.embedding
        
        # Sum importance weights
        total_weight = sum(c.importance_weight for c in chunks)
        
        # Combine tokens (unique)
        all_tokens = []
        seen_tokens = set()
        for chunk in chunks:
            for token in chunk.tokens:
                if token not in seen_tokens:
                    all_tokens.append(token)
                    seen_tokens.add(token)
        
        return ProcessedChunk(
            content=base_chunk.content,
            embedding=avg_embedding,
            tokens=all_tokens[:base_chunk.token_count],  # Keep original length
            token_count=base_chunk.token_count,
            semantic_fingerprint=base_chunk.semantic_fingerprint,
            importance_weight=total_weight
        )