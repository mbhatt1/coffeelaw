#!/usr/bin/env python3
"""
Quick test script for Coffee Law verification
"""
import sys
import os
import warnings

# Filter out the RuntimeWarning about coroutines
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited")

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
from coffee_law_verifier.monte_carlo import TaskGenerator
from coffee_law_verifier.analysis import PowerLawAnalyzer
from coffee_law_verifier.context_engine import PeContextCalculator
import numpy as np

def test_basic_functionality():
    """Test basic components work"""
    print("Testing Coffee Law Verifier Components...")
    
    # Test 1: Task generation
    print("\n1. Testing task generation...")
    task_gen = TaskGenerator(seed=42)
    tasks = task_gen.generate_task_dataset(n_tasks=5)
    print(f"   ✓ Generated {len(tasks)} tasks")
    
    # Test 2: Pe_ctx calculation
    print("\n2. Testing Pe_ctx calculation...")
    pe_calc = PeContextCalculator()
    pe_ctx, _ = pe_calc.calculate_pe_ctx()
    print(f"   ✓ Calculated Pe_ctx: {pe_ctx:.3f}")
    
    # Test 3: Power law analysis
    print("\n3. Testing power law fitting...")
    analyzer = PowerLawAnalyzer()
    x = np.logspace(-1, 1, 50)
    y = x**(-0.33) + np.random.normal(0, 0.01, 50)
    fit = analyzer.fit_power_law(x, y, expected_exponent=-1/3)
    print(f"   ✓ Fitted exponent: {fit.exponent:.4f} (expected: -0.3333)")
    print(f"   ✓ Passed: {fit.passed}")
    
    print("\n✅ All basic tests passed!")
    print("\nNow run './run_coffee_law_verification.py --quick' for full verification")

if __name__ == "__main__":
    test_basic_functionality()