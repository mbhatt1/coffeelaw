#!/usr/bin/env python3
"""
Standalone script to run Coffee Law verification
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run
from coffee_law_verifier.run_verification import main

if __name__ == "__main__":
    main()