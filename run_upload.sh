#!/bin/bash
# Wrapper script to run upload_gemma_to_s3.py with proper virtual environment

set -e

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies if needed
pip list | grep -q boto3 || pip install -q boto3 huggingface_hub

# Run the upload script with any arguments passed
python3 upload_gemma_to_s3.py "$@"
