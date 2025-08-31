#!/usr/bin/env python3
"""
Run Coffee Law verification with 16,000 total samples
"""
import subprocess
import sys

# Calculate samples per variant for 16k total samples
# With 6 Pe_ctx variants: 16000 / 6 = 2666.67
SAMPLES_PER_VARIANT = 266788

print(f"Running Coffee Law verification with {SAMPLES_PER_VARIANT} samples per variant")
print(f"Total samples: {SAMPLES_PER_VARIANT * 6} (6 Pe_ctx variants)")
print("="*60)

# Run the verification
cmd = [
    sys.executable,
    "run_coffee_law_verification.py",
    f"--samples={SAMPLES_PER_VARIANT}"
]

# Add any command line arguments passed to this script
if len(sys.argv) > 1:
    cmd.extend(sys.argv[1:])

subprocess.run(cmd)