"""
Embedding analysis for D_eff and other dimensionality metrics
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

class EmbeddingAnalyzer:
    """
    Analyze embeddings to calculate effective dimension and other metrics
    
    Key metric: D_eff (participation ratio) = (tr C)² / tr(C²)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_d_effective(self, embeddings: np.ndarray) -> float:
        """
        Calculate effective dimension using participation ratio
        
        From README: D_eff = (tr C)² / tr(C²)
        where C is the covariance matrix
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            D_eff value
        """
        if len(embeddings) < 2:
            return 1.0
        
        # Center the embeddings
        centered = embeddings - embeddings.mean(axis=0)
        
        # Calculate covariance matrix
        n_samples = len(embeddings)
        cov = (centered.T @ centered) / (n_samples - 1)
        
        # Calculate traces
        trace_c = np.trace(cov)
        trace_c2 = np.trace(cov @ cov)
        
        # Avoid division by zero
        if trace_c2 < 1e-10:
            return 1.0
        
        # Participation ratio
        d_eff = (trace_c ** 2) / trace_c2
        
        return d_eff
    
    def analyze_embedding_structure(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive analysis of embedding structure
        
        Returns dictionary with various dimensionality measures
        """
        if len(embeddings) < 2:
            return {
                'd_eff': 1.0,
                'intrinsic_dim': 1.0,
                'pca_95': 1,
                'pca_99': 1,
                'condition_number': 1.0
            }
        
        results = {}
        
        # Effective dimension (participation ratio)
        results['d_eff'] = self.calculate_d_effective(embeddings)
        
        # Intrinsic dimension estimation
        results['intrinsic_dim'] = self._estimate_intrinsic_dimension(embeddings)
        
        # PCA-based dimensions
        pca_dims = self._pca_dimension_analysis(embeddings)
        results.update(pca_dims)
        
        # Condition number (spread of eigenvalues)
        results['condition_number'] = self._calculate_condition_number(embeddings)
        
        return results
    
    def _estimate_intrinsic_dimension(self, embeddings: np.ndarray) -> float:
        """
        Estimate intrinsic dimension using MLE method
        
        Based on: Levina & Bickel (2005) - Maximum Likelihood Estimation of Intrinsic Dimension
        """
        n_samples = len(embeddings)
        if n_samples < 10:
            return 1.0
        
        # Use k nearest neighbors
        k = min(10, n_samples - 1)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, embeddings)
        
        # For each point, get k nearest neighbors
        dim_estimates = []
        
        for i in range(n_samples):
            # Get distances to other points (excluding self)
            dists = distances[i]
            dists = dists[dists > 0]
            
            if len(dists) < k:
                continue
            
            # Get k nearest distances
            nearest_dists = np.sort(dists)[:k]
            
            # MLE estimate
            if nearest_dists[-1] > 0:
                log_ratios = np.log(nearest_dists[-1] / nearest_dists[:-1])
                local_dim = 1.0 / np.mean(log_ratios)
                dim_estimates.append(local_dim)
        
        if dim_estimates:
            return np.median(dim_estimates)
        return 1.0
    
    def _pca_dimension_analysis(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Analyze dimensions needed to capture variance thresholds
        """
        # Standardize
        standardized = self.scaler.fit_transform(embeddings)
        
        # PCA
        n_components = min(len(embeddings) - 1, embeddings.shape[1])
        pca = PCA(n_components=n_components)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca.fit(standardized)
        
        # Cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        # Dimensions needed for different thresholds
        pca_95 = np.argmax(cumsum_var >= 0.95) + 1
        pca_99 = np.argmax(cumsum_var >= 0.99) + 1
        pca_90 = np.argmax(cumsum_var >= 0.90) + 1
        
        return {
            'pca_90': int(pca_90),
            'pca_95': int(pca_95),
            'pca_99': int(pca_99),
            'explained_var_ratio': pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 1.0
        }
    
    def _calculate_condition_number(self, embeddings: np.ndarray) -> float:
        """
        Calculate condition number of covariance matrix
        
        High condition number indicates ill-conditioning (some directions dominate)
        """
        # Center embeddings
        centered = embeddings - embeddings.mean(axis=0)
        
        # Covariance
        cov = np.cov(centered.T)
        
        # Eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) > 0:
                condition_number = eigenvalues.max() / eigenvalues.min()
                return condition_number
        except np.linalg.LinAlgError:
            pass
        
        return float('inf')
    
    def track_d_effective_trajectory(self,
                                   embedding_sets: List[np.ndarray]) -> List[float]:
        """
        Track how D_eff changes across multiple embedding sets
        
        Useful for seeing how effective dimension changes with Pe_ctx
        """
        d_eff_values = []
        
        for embeddings in embedding_sets:
            d_eff = self.calculate_d_effective(embeddings)
            d_eff_values.append(d_eff)
        
        return d_eff_values
    
    def calculate_embedding_diversity(self, embeddings: np.ndarray) -> float:
        """
        Calculate diversity score of embeddings
        
        Higher score = more diverse responses
        """
        if len(embeddings) < 2:
            return 0.0
        
        # Average pairwise distance
        from scipy.spatial.distance import pdist
        pairwise_dists = pdist(embeddings)
        
        if len(pairwise_dists) > 0:
            return pairwise_dists.mean()
        return 0.0
    
    def calculate_anisotropy(self, embeddings: np.ndarray) -> float:
        """
        Calculate anisotropy (how much embeddings vary by direction)
        
        0 = isotropic (equal variance in all directions)
        1 = highly anisotropic (variance concentrated in few directions)
        """
        if len(embeddings) < 2:
            return 0.0
        
        # Get eigenvalues of covariance
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 0]
            
            if len(eigenvalues) < 2:
                return 0.0
            
            # Normalize
            eigenvalues = eigenvalues / eigenvalues.sum()
            
            # Calculate anisotropy as 1 - entropy of eigenvalue distribution
            # Maximum entropy = log(n) for uniform distribution
            entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
            max_entropy = np.log(len(eigenvalues))
            
            anisotropy = 1.0 - (entropy / max_entropy)
            
            return anisotropy
            
        except np.linalg.LinAlgError:
            return 0.0