#!/usr/bin/env python3
"""
Run Coffee Law verification with 16,000 total samples

This script runs the Coffee Law verification with specific sample sizes
and supports model selection for both LLM and embedding providers.

Usage:
    # Default (mock models)
    python run_16k_verification.py
    
    # With OpenAI
    python run_16k_verification.py --llm-provider openai --embedding-provider openai
    
    # With Anthropic
    python run_16k_verification.py --llm-provider anthropic --embedding-provider anthropic
    
    # With Gemini
    python run_16k_verification.py --llm-provider gemini --embedding-provider gemini
    
    # Mixed providers
    python run_16k_verification.py --llm-provider anthropic --embedding-provider openai
    
    # With specific models
    python run_16k_verification.py --llm-provider openai --llm-model gpt-4 --embedding-provider openai --embedding-model text-embedding-3-large
"""
import subprocess
import sys
import argparse

# Calculate samples per variant for 16k total samples
# With 6 Pe_ctx variants: 16000 / 6 = 2666.67
SAMPLES_PER_VARIANT = 2667  # Fixed to reasonable 16k samples

def main():
    parser = argparse.ArgumentParser(
        description="Run Coffee Law verification with 16k samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection arguments
    parser.add_argument('--llm-provider', type=str,
                       choices=['mock', 'openai', 'anthropic', 'gemini'],
                       default='mock',
                       help='LLM provider to use (default: mock)')
    parser.add_argument('--llm-model', type=str,
                       help='Specific LLM model to use (provider-dependent)')
    parser.add_argument('--embedding-provider', type=str,
                       choices=['mock', 'openai', 'anthropic', 'gemini', 'vertex'],
                       default='mock',
                       help='Embedding provider to use (default: mock)')
    parser.add_argument('--embedding-model', type=str,
                       help='Specific embedding model to use (provider-dependent)')
    
    # Other options
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_VARIANT,
                       help=f'Override samples per variant (default: {SAMPLES_PER_VARIANT})')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip report generation')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick diagnostic test')
    
    # Legacy support
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI embeddings (legacy, use --embedding-provider instead)')
    
    args, unknown_args = parser.parse_known_args()
    
    # Print configuration
    print("Coffee Law Verification - 16K Sample Run")
    print("="*60)
    print(f"Samples per variant: {args.samples}")
    print(f"Total samples: {args.samples * 6} (6 Pe_ctx variants)")
    print(f"LLM Provider: {args.llm_provider}" + (f" ({args.llm_model})" if args.llm_model else ""))
    print(f"Embedding Provider: {args.embedding_provider}" + (f" ({args.embedding_model})" if args.embedding_model else ""))
    print("="*60)
    
    # Build command
    cmd = [
        sys.executable,
        "run_coffee_law_verification.py",
        f"--samples={args.samples}",
        f"--llm-provider={args.llm_provider}",
        f"--embedding-provider={args.embedding_provider}"
    ]
    
    # Add optional arguments
    if args.llm_model:
        cmd.append(f"--llm-model={args.llm_model}")
    if args.embedding_model:
        cmd.append(f"--embedding-model={args.embedding_model}")
    if args.no_report:
        cmd.append("--no-report")
    if args.quick:
        cmd.append("--quick")
    if args.use_openai:
        cmd.append("--use-openai")
    
    # Add any unknown arguments
    cmd.extend(unknown_args)
    
    # Run the verification
    subprocess.run(cmd)

if __name__ == "__main__":
    main()