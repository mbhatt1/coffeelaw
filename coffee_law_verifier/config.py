"""
Configuration for Coffee Law Verifier
"""
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True)

@dataclass
class VerifierConfig:
    """Main configuration for the Coffee Law verifier"""
    
    # API settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    
    # Monte Carlo settings
    n_pe_ctx_variants: int = 6  # Number of Pe_ctx variants to test
    samples_per_variant: int = 100  # Samples per Pe_ctx variant
    n_embedding_samples: int = 16  # Samples for ambiguity width measurement
    
    # Statistical thresholds (from README)
    w_slope_expected: float = -1/3
    w_slope_tolerance: float = 0.07
    entropy_slope_expected: float = 2/3
    entropy_slope_tolerance: float = 0.10
    alpha_slope_expected: float = -1/3
    alpha_slope_tolerance: float = 0.10
    
    # Context engineering parameters
    min_temperature: float = 0.1
    max_temperature: float = 0.8
    default_temperature: float = 0.3
    
    # Chunk parameters
    min_chunks: int = 1
    max_chunks: int = 20
    optimal_chunks_range: tuple = (3, 7)  # From README
    
    # Numerical parameters
    n_pca_components: Optional[int] = None  # None = automatic
    entropy_bins: Optional[int] = None  # None = sqrt(n_samples)
    
    # Experiment settings
    random_seed: int = 42
    verbose: bool = True
    save_intermediate: bool = True

# Global config instance
CONFIG = VerifierConfig()