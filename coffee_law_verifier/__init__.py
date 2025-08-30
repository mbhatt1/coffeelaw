"""
Coffee Law Verifier - Production Monte Carlo simulation for Coffee Law verification

This package implements comprehensive verification of the Coffee Law claims:
1. W/√D_eff ∝ Pe_ctx^(-1/3) 
2. H = a + b*ln(Pe_ctx) with b ≈ 2/3
3. α(N) ∼ N^(-1/3)
"""

__version__ = "1.0.0"

from .run_verification import CoffeeLawVerifier

__all__ = ["CoffeeLawVerifier"]