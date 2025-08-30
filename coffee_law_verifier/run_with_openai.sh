#!/bin/bash

# Check if OPENAI_API_KEY is provided as argument
if [ -z "$1" ]; then
    echo "Usage: ./run_with_openai.sh YOUR_OPENAI_API_KEY [additional_args]"
    echo "Example: ./run_with_openai.sh sk-... --samples=100"
    exit 1
fi

# Set the API key
export OPENAI_API_KEY="$1"

# Shift arguments to remove API key
shift

# Run verification with OpenAI embeddings and any additional arguments
echo "Running Coffee Law verification with OpenAI embeddings..."
python3 run_verification.py --use-openai "$@"