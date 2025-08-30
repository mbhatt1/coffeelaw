"""
Entropy measurement for Coffee Law verification
"""
import numpy as np
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as scipy_entropy
import warnings

class EntropyMeasurer:
    """
    Measure coarse entropy H of response embeddings
    
    From README: H = coarse entropy of PCA-whitened embeddings (fixed bins)
    """
    
    def __init__(self, n_bins: Optional[int] = None):
        self.n_bins = n_bins
        self.pca = None
        self.scaler = StandardScaler()
        
    def calculate_entropy(self, embeddings: np.ndarray, n_bins: Optional[int] = None) -> float:
        """
        Calculate coarse entropy H from embeddings
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            n_bins: Number of bins for discretization (default: sqrt(n_samples))
            
        Returns:
            Entropy value H
        """
        if len(embeddings) < 2:
            return 0.0
        
        # PCA whitening as specified in README
        whitened = self._pca_whiten(embeddings)
        
        # Determine number of bins
        if n_bins is None:
            n_bins = self.n_bins or int(np.sqrt(len(embeddings)))
        
        # Calculate entropy
        H = self._calculate_discrete_entropy(whitened, n_bins)
        
        return H
    
    def _pca_whiten(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply PCA whitening to embeddings
        
        Whitening ensures:
        1. Zero mean
        2. Unit variance in all directions
        3. Uncorrelated components
        """
        # Standardize first
        standardized = self.scaler.fit_transform(embeddings)
        
        # Apply PCA
        n_components = min(len(embeddings) - 1, embeddings.shape[1])
        self.pca = PCA(n_components=n_components, whiten=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            whitened = self.pca.fit_transform(standardized)
        
        return whitened
    
    def _calculate_discrete_entropy(self, data: np.ndarray, n_bins: int) -> float:
        """
        Calculate Shannon entropy after discretization
        
        Args:
            data: Whitened embeddings
            n_bins: Number of bins per dimension
            
        Returns:
            Shannon entropy
        """
        # Get data range for binning
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        
        # Handle edge case of no variation
        ranges = data_max - data_min
        if np.all(ranges < 1e-10):
            return 0.0
        
        # Create bins for each dimension
        bins_per_dim = []
        for i in range(data.shape[1]):
            if ranges[i] > 1e-10:
                bins = np.linspace(data_min[i], data_max[i] + 1e-10, n_bins + 1)
            else:
                bins = np.array([data_min[i] - 1e-10, data_max[i] + 1e-10])
            bins_per_dim.append(bins)
        
        # Discretize data using multidimensional histogram
        try:
            # Limit dimensions to avoid memory issues
            max_dims = min(5, data.shape[1])  # Use top 5 PCA components
            hist, _ = np.histogramdd(data[:, :max_dims], bins=n_bins)
            
            # Flatten and normalize to get probabilities
            probs = hist.flatten()
            probs = probs[probs > 0]  # Remove zero bins
            probs = probs / probs.sum()
            
            # Calculate Shannon entropy
            H = -np.sum(probs * np.log(probs))
            
        except (MemoryError, ValueError):
            # Fallback: use 1D entropy on first principal component
            hist, _ = np.histogram(data[:, 0], bins=n_bins)
            probs = hist[hist > 0] / hist.sum()
            H = -np.sum(probs * np.log(probs))
        
        return H
    
    def calculate_entropy_trajectory(self, 
                                   embedding_sets: List[np.ndarray]) -> List[float]:
        """
        Calculate entropy for multiple embedding sets
        
        Useful for tracking how entropy changes with Pe_ctx
        """
        entropies = []
        
        for embeddings in embedding_sets:
            H = self.calculate_entropy(embeddings)
            entropies.append(H)
        
        return entropies
    
    def calculate_differential_entropy(self, embeddings: np.ndarray) -> float:
        """
        Calculate differential (continuous) entropy as alternative measure
        
        This assumes Gaussian distribution after whitening
        """
        if len(embeddings) < 2:
            return 0.0
        
        whitened = self._pca_whiten(embeddings)
        
        # For whitened data, differential entropy of multivariate Gaussian
        # h = (d/2) * log(2πe) where d is dimension
        d = whitened.shape[1]
        h_diff = 0.5 * d * np.log(2 * np.pi * np.e)
        
        return h_diff
    
    def calculate_renyi_entropy(self, embeddings: np.ndarray, alpha: float = 2.0) -> float:
        """
        Calculate Rényi entropy of order α
        
        α = 1 gives Shannon entropy
        α = 2 gives collision entropy
        """
        if len(embeddings) < 2:
            return 0.0
        
        whitened = self._pca_whiten(embeddings)
        n_bins = int(np.sqrt(len(embeddings)))
        
        # Discretize
        hist, _ = np.histogramdd(whitened[:, :min(3, whitened.shape[1])], bins=n_bins)
        probs = hist.flatten()
        probs = probs[probs > 0]
        probs = probs / probs.sum()
        
        if alpha == 1.0:
            # Shannon entropy
            return -np.sum(probs * np.log(probs))
        else:
            # Rényi entropy
            return np.log(np.sum(probs ** alpha)) / (1 - alpha)
    
    def estimate_entropy_scaling(self,
                               embedding_sets: List[np.ndarray],
                               pe_ctx_values: List[float]) -> Tuple[float, float]:
        """
        Estimate the entropy scaling parameter b from H = a + b*ln(Pe_ctx)
        
        Returns:
            Tuple of (b, stderr) where b should be ≈ 2/3
        """
        if len(embedding_sets) < 2:
            return 0.0, float('inf')
        
        # Calculate entropies
        H_values = self.calculate_entropy_trajectory(embedding_sets)
        
        # Fit H = a + b*ln(Pe_ctx)
        from scipy import stats
        log_pe = np.log(pe_ctx_values)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_pe, H_values)
        
        return slope, std_err