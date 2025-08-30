"""
Pe_ctx calculator with different estimation methods
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PeComponents:
    """Components that make up Pe_ctx"""
    # Stretch factors
    alignment_score: float
    schema_score: float
    front_loading_score: float
    
    # Diffusion factors
    redundancy: float
    conflict: float
    style_drift: float
    decoding_noise: float
    
    # Computed values
    stretch_total: float
    diffusion_total: float
    pe_ctx: float

class PeContextCalculator:
    """
    Calculate and track Pe_ctx values for Coffee Law experiments
    
    From README: Pe_ctx ≈ (alignment × schema × front-loading) / 
                          (redundancy + conflict + style drift + decoding noise)
    """
    
    def __init__(self, baseline_temperature: float = 0.3):
        self.baseline_temp = baseline_temperature
        self.pe_history = []
        
    def calculate_pe_ctx(self,
                        alignment: float = 1.0,
                        schema: float = 1.0,
                        front_loading: float = 1.0,
                        redundancy: float = 0.0,
                        conflict: float = 0.0,
                        style_drift: float = 0.0,
                        temperature: float = 0.3,
                        return_components: bool = False) -> Tuple[float, Optional[PeComponents]]:
        """
        Calculate Pe_ctx from individual components
        
        Args:
            alignment: How well chunks align with task (0-1)
            schema: Template/structure strength (0-1)
            front_loading: Relevance-based ordering strength (0-1)
            redundancy: Amount of redundant information (0-1)
            conflict: Amount of conflicting information (0-1)
            style_drift: Inconsistency in style/units (0-1)
            temperature: Decoding temperature
            return_components: Whether to return detailed breakdown
            
        Returns:
            Tuple of (pe_ctx_value, optional_components)
        """
        # Normalize temperature to baseline
        decoding_noise = temperature / self.baseline_temp
        
        # Calculate stretch (multiplicative factors)
        stretch = alignment * schema * front_loading
        
        # Calculate diffusion (additive factors)
        diffusion = redundancy + conflict + style_drift + decoding_noise
        
        # Prevent division by zero with smaller floor for wider range
        diffusion = max(diffusion, 0.01)
        
        # Calculate Pe_ctx
        pe_ctx = stretch / diffusion
        
        # Store in history
        self.pe_history.append(pe_ctx)
        
        if return_components:
            components = PeComponents(
                alignment_score=alignment,
                schema_score=schema,
                front_loading_score=front_loading,
                redundancy=redundancy,
                conflict=conflict,
                style_drift=style_drift,
                decoding_noise=decoding_noise,
                stretch_total=stretch,
                diffusion_total=diffusion,
                pe_ctx=pe_ctx
            )
            return pe_ctx, components
        
        return pe_ctx, None
    
    def calculate_pe_ctx_from_metrics(self,
                                    chunk_metrics: Dict[str, float],
                                    context_metrics: Dict[str, float]) -> float:
        """
        Calculate Pe_ctx from measured chunk and context metrics
        
        This is a more sophisticated calculation based on actual measurements
        rather than control parameters
        """
        # Extract relevant metrics
        n_chunks = chunk_metrics.get('n_chunks', 1)
        n_eff = chunk_metrics.get('n_effective', n_chunks)
        avg_similarity = chunk_metrics.get('avg_similarity', 0.5)
        
        reranker_scores = context_metrics.get('reranker_scores', [0.5])
        template_adherence = context_metrics.get('template_adherence', 0.5)
        glossary_consistency = context_metrics.get('glossary_consistency', 1.0)
        
        # Estimate stretch factors
        alignment = np.mean(reranker_scores) if reranker_scores else 0.5
        schema = template_adherence
        front_loading = self._calculate_front_loading_score(reranker_scores)
        
        # Estimate diffusion factors
        redundancy = 1.0 - (n_eff / n_chunks) if n_chunks > 0 else 0.0
        conflict = context_metrics.get('conflict_score', 0.0)
        style_drift = 1.0 - glossary_consistency
        temperature = context_metrics.get('temperature', self.baseline_temp)
        
        pe_ctx, _ = self.calculate_pe_ctx(
            alignment=alignment,
            schema=schema,
            front_loading=front_loading,
            redundancy=redundancy,
            conflict=conflict,
            style_drift=style_drift,
            temperature=temperature
        )
        
        return pe_ctx
    
    def _calculate_front_loading_score(self, relevance_scores: List[float]) -> float:
        """
        Calculate how well chunks are front-loaded by relevance
        Returns 1.0 for perfect descending order, 0.5 for random, 0.0 for worst
        Uses Kendall's tau mapped to [0,1] range
        """
        if len(relevance_scores) <= 1:
            return 1.0
        
        n = len(relevance_scores)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if relevance_scores[i] > relevance_scores[j]:
                    concordant += 1
                elif relevance_scores[i] < relevance_scores[j]:
                    discordant += 1
                # ties are neither concordant nor discordant
        
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 1.0
        
        # Calculate Kendall's tau
        tau = (concordant - discordant) / total_pairs
        
        # Map from [-1, 1] to [0, 1]
        return (tau + 1) / 2
    
    def get_pe_range_for_sweep(self, n_points: int = 6,
                             min_pe: float = 0.1,
                             max_pe: float = 10.0) -> List[float]:
        """
        Generate Pe_ctx values for experimental sweep
        Uses log spacing as suggested in README (>1 decade)
        """
        return np.logspace(np.log10(min_pe), np.log10(max_pe), n_points).tolist()
    
    def create_variant_parameters(self, target_pe: float) -> Dict[str, float]:
        """
        Create parameter settings to achieve target Pe_ctx
        
        Uses direct parameter mapping based on Pe_ctx range [0.1, 10.0]
        """
        # Map target_pe to parameter ranges using log scale
        # Pe_ctx = 0.1 -> log10(0.1) = -1
        # Pe_ctx = 10.0 -> log10(10.0) = 1
        log_pe = np.log10(np.clip(target_pe, 0.1, 10.0))
        # Normalize to [0, 1] range: (log_pe + 1) / 2
        norm_pe = (log_pe + 1) / 2
        
        # For high Pe_ctx, we want:
        # - High stretch factors (template, front_loading)
        # - Low diffusion (high dedup, consistency, conflict_res, low temp)
        
        # Direct mapping for extreme Pe_ctx values
        if target_pe >= 10.0:
            params = {
                'template_strength': 0.99,
                'front_loading': 0.99,
                'deduplication': 0.99,
                'style_consistency': 0.99,
                'conflict_resolution': 0.99,
                'temperature': 0.01
            }
        elif target_pe <= 0.1:
            params = {
                'template_strength': 0.01,
                'front_loading': 0.01,
                'deduplication': 0.01,
                'style_consistency': 0.01,
                'conflict_resolution': 0.01,
                'temperature': 2.0
            }
        else:
            # Interpolate parameters based on normalized Pe
            # Stretch factors increase with Pe
            params = {
                'template_strength': 0.01 + 0.98 * norm_pe,
                'front_loading': 0.01 + 0.98 * norm_pe,
                'deduplication': 0.01 + 0.98 * norm_pe,
                'style_consistency': 0.01 + 0.98 * norm_pe,
                'conflict_resolution': 0.01 + 0.98 * norm_pe,
                'temperature': 2.0 * (1 - norm_pe) + 0.01 * norm_pe
            }
        
        return params
    
    def estimate_diffusion_floor(self, measured_b: float) -> float:
        """
        Estimate diffusion floor parameter γ from measured entropy slope
        
        From README: b = (2/3) / (1 + γ*Pe_ctx) => γ*Pe_ctx = (2/3)/b - 1
        """
        expected_b = 2/3
        
        if measured_b <= 0 or measured_b >= expected_b:
            return 0.0
        
        # Average Pe_ctx from history
        if not self.pe_history:
            return 0.0
            
        avg_pe = np.mean(self.pe_history)
        
        # Calculate γ
        gamma_pe = (expected_b / measured_b) - 1
        gamma = gamma_pe / avg_pe if avg_pe > 0 else 0.0
        
        return gamma