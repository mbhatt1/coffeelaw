"""
Coffee Law Verifier - Production Monte Carlo simulation for Coffee Law verification

This package implements comprehensive verification of the Coffee Law claims:
1. Law 1 - Cube-root Sharpening: W/√D_eff = α · Pe_ctx^(-1/3)
2. Law 2 - Entropy Scaling: H = H₀ + (2/3)ln(Pe_ctx)
3. Law 3 - Logarithmic Context Scaling: Pe_ctx(N) = a + b·ln(N)
"""

__version__ = "1.0.0"

from .run_verification import CoffeeLawVerifier

__all__ = ["CoffeeLawVerifier"]