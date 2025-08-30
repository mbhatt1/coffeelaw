"""
Context Variator for controlling Pe_ctx through stretch and diffusion factors
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .pe_calculator import PeContextCalculator

@dataclass
class ContextChunk:
    content: str
    relevance_score: float
    source: str
    metadata: Dict[str, any]

class ContextVariator:
    """
    Controls Pe_ctx by manipulating:
    - Stretch factors: alignment × schema × front-loading
    - Diffusion factors: redundancy + conflict + style drift + decoding noise
    """
    
    def __init__(self, tasks: List[Dict], glossary: Optional[Dict[str, str]] = None):
        self.tasks = tasks
        self.glossary = glossary or {}
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.pe_calculator = PeContextCalculator()
        
    def create_variant(self,
                      template_strength: float = 1.0,
                      front_loading: float = 1.0,
                      deduplication: float = 1.0,
                      style_consistency: float = 1.0,
                      conflict_resolution: float = 1.0,
                      temperature: float = 0.3) -> Tuple[str, float]:
        """
        Generate a context variant with controlled Pe_ctx
        
        Returns:
            Tuple of (formatted_context, pe_ctx_value)
        """
        # Extract chunks from documents
        chunks = self._extract_chunks()
        
        # Apply stretch operations
        if template_strength > 0:
            chunks = self._apply_template(chunks, template_strength)
        
        if front_loading > 0:
            chunks = self._reorder_by_relevance(chunks, front_loading)
            
        # Apply diffusion reduction
        if deduplication > 0:
            chunks = self._deduplicate_chunks(chunks, deduplication)
            
        if conflict_resolution > 0:
            chunks = self._resolve_conflicts(chunks, conflict_resolution)
            
        if style_consistency > 0:
            chunks = self._canonicalize_style(chunks, style_consistency)
            
        # Format final context
        context = self._format_context(chunks)
        
        # Apply glossary enforcement
        if self.glossary:
            context = self._enforce_glossary(context)
        
        # Calculate Pe_ctx using the PeCalculator
        pe_ctx, _ = self.pe_calculator.calculate_pe_ctx(
            alignment=1.0,  # Baseline alignment
            schema=template_strength,
            front_loading=front_loading,
            redundancy=1.0 - deduplication,
            conflict=1.0 - conflict_resolution,
            style_drift=1.0 - style_consistency,
            temperature=temperature
        )
        
        return context, pe_ctx
    
    def _extract_chunks(self) -> List[ContextChunk]:
        """Extract chunks from tasks"""
        chunks = []
        
        # If we have a single task passed, extract its chunks
        if self.tasks and isinstance(self.tasks[0], dict) and 'chunks' in self.tasks[0]:
            # We're working with a single task
            task = self.tasks[0]
            task_chunks = task.get('chunks', [])
            
            for i, chunk_content in enumerate(task_chunks):
                chunks.append(ContextChunk(
                    content=chunk_content,
                    relevance_score=0.5 + (0.5 / (i + 1)),  # Higher relevance for earlier chunks
                    source=task.get('type', 'unknown'),
                    metadata={'position': i, 'task_id': task.get('id', '')}
                ))
        else:
            # Fallback: treat tasks as documents with content field
            for doc in self.tasks:
                if 'content' in doc:
                    sentences = re.split(r'[.!?]+', doc['content'])
                    for i, sent in enumerate(sentences):
                        if len(sent.strip()) > 10:
                            chunks.append(ContextChunk(
                                content=sent.strip(),
                                relevance_score=doc.get('relevance', 0.5),
                                source=doc.get('source', 'unknown'),
                                metadata={'position': i, 'doc_id': doc.get('id', '')}
                            ))
                elif 'chunks' in doc:
                    # Handle task-like structure
                    for i, chunk in enumerate(doc.get('chunks', [])):
                        chunks.append(ContextChunk(
                            content=chunk,
                            relevance_score=0.5 + (0.5 / (i + 1)),
                            source=doc.get('type', 'unknown'),
                            metadata={'position': i, 'task_id': doc.get('id', '')}
                        ))
        
        return chunks
    
    def _apply_template(self, chunks: List[ContextChunk], strength: float) -> List[ContextChunk]:
        """Apply structural template based on README guidelines"""
        if strength < 0.1:
            return chunks
            
        # Template: Goal → Definitions → Constraints → Evidence → Task → Output
        categorized = {
            'goal': [],
            'definitions': [],
            'constraints': [],
            'evidence': [],
            'task': [],
            'output': []
        }
        
        # Simple keyword-based categorization
        for chunk in chunks:
            content_lower = chunk.content.lower()
            if any(word in content_lower for word in ['goal', 'objective', 'purpose']):
                categorized['goal'].append(chunk)
            elif any(word in content_lower for word in ['define', 'definition', 'meaning']):
                categorized['definitions'].append(chunk)
            elif any(word in content_lower for word in ['constraint', 'limit', 'restriction']):
                categorized['constraints'].append(chunk)
            elif any(word in content_lower for word in ['evidence', 'data', 'result', 'finding']):
                categorized['evidence'].append(chunk)
            elif any(word in content_lower for word in ['task', 'step', 'action']):
                categorized['task'].append(chunk)
            elif any(word in content_lower for word in ['output', 'format', 'result']):
                categorized['output'].append(chunk)
            else:
                categorized['evidence'].append(chunk)  # Default category
        
        # Reconstruct chunks in template order
        ordered_chunks = []
        for category in ['goal', 'definitions', 'constraints', 'evidence', 'task', 'output']:
            ordered_chunks.extend(categorized[category][:int(len(categorized[category]) * strength)])
        
        return ordered_chunks
    
    def _reorder_by_relevance(self, chunks: List[ContextChunk], strength: float) -> List[ContextChunk]:
        """Reorder chunks by relevance score with front-loading strength"""
        if strength < 0.1:
            return chunks
            
        # Sort by relevance score
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        
        # Apply front-loading strength (mix sorted and original order)
        n_frontloaded = int(len(chunks) * strength)
        frontloaded = sorted_chunks[:n_frontloaded]
        remaining = chunks[n_frontloaded:]
        
        return frontloaded + remaining
    
    def _deduplicate_chunks(self, chunks: List[ContextChunk], strength: float) -> List[ContextChunk]:
        """Remove duplicate or near-duplicate chunks"""
        if strength < 0.1 or len(chunks) < 2:
            return chunks
            
        # Vectorize chunks
        texts = [c.content for c in chunks]
        try:
            vectors = self.vectorizer.fit_transform(texts)
            similarities = cosine_similarity(vectors)
            
            # Mark duplicates based on similarity threshold
            threshold = 0.9 * strength  # Higher strength = stricter deduplication
            keep_indices = []
            
            for i in range(len(chunks)):
                is_duplicate = False
                for j in keep_indices:
                    if similarities[i, j] > threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep_indices.append(i)
            
            return [chunks[i] for i in keep_indices]
        except:
            # Fallback to simple exact matching
            seen_content = set()
            unique_chunks = []
            for chunk in chunks:
                content_key = chunk.content.lower().strip()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_chunks.append(chunk)
            return unique_chunks
    
    def _resolve_conflicts(self, chunks: List[ContextChunk], strength: float) -> List[ContextChunk]:
        """Resolve conflicting information in chunks"""
        if strength < 0.1:
            return chunks
            
        # Simple conflict detection based on negation patterns
        conflict_patterns = [
            (r'is\s+not', r'is'),
            (r'cannot', r'can'),
            (r'never', r'always'),
            (r'false', r'true'),
        ]
        
        resolved_chunks = []
        seen_facts = {}
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            
            # Check for potential conflicts
            is_conflicting = False
            for neg_pattern, pos_pattern in conflict_patterns:
                if re.search(neg_pattern, content_lower):
                    base_fact = re.sub(neg_pattern, pos_pattern, content_lower)
                    if base_fact in seen_facts:
                        # Conflict detected - keep higher relevance
                        if chunk.relevance_score > seen_facts[base_fact].relevance_score:
                            # Replace previous chunk
                            resolved_chunks = [c for c in resolved_chunks if c != seen_facts[base_fact]]
                            resolved_chunks.append(chunk)
                            seen_facts[base_fact] = chunk
                        is_conflicting = True
                        break
                elif re.search(pos_pattern, content_lower):
                    seen_facts[content_lower] = chunk
            
            if not is_conflicting:
                resolved_chunks.append(chunk)
        
        return resolved_chunks
    
    def _canonicalize_style(self, chunks: List[ContextChunk], strength: float) -> List[ContextChunk]:
        """Canonicalize units, dates, and writing style"""
        if strength < 0.1:
            return chunks
            
        canonicalized = []
        for chunk in chunks:
            content = chunk.content
            
            # Standardize units (simple examples)
            if strength > 0.5:
                content = re.sub(r'(\d+)\s*km', r'\1 kilometers', content)
                content = re.sub(r'(\d+)\s*m', r'\1 meters', content)
                content = re.sub(r'(\d+)\s*kg', r'\1 kilograms', content)
                
            # Standardize date formats
            if strength > 0.7:
                content = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\2-\1', content)
                
            # Standardize number formats
            if strength > 0.3:
                content = re.sub(r'(\d),(\d{3})', r'\1\2', content)
            
            canonicalized.append(ContextChunk(
                content=content,
                relevance_score=chunk.relevance_score,
                source=chunk.source,
                metadata=chunk.metadata
            ))
        
        return canonicalized
    
    def _enforce_glossary(self, context: str) -> str:
        """Enforce consistent terminology from glossary"""
        for term, canonical in self.glossary.items():
            # Case-insensitive replacement while preserving original case
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            context = pattern.sub(canonical, context)
        return context
    
    def _format_context(self, chunks: List[ContextChunk]) -> str:
        """Format chunks into final context string"""
        sections = []
        
        # Group by rough categories (simplified)
        current_section = []
        for chunk in chunks:
            current_section.append(f"• {chunk.content}")
            
            # Add section breaks every 3-5 chunks for readability
            if len(current_section) >= 4:
                sections.append('\n'.join(current_section))
                current_section = []
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return '\n\n'.join(sections)
    