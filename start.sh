#!/bin/bash
set -e

# Activate the virtual environment
source .venv/bin/activate

REPO_ID="Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR="/app/base-models/Qwen/Qwen3-4B-Instruct-2507"

# Check if the folder is empty or doesn't exist
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR")" ]; then
    echo "Downloading model..."

    # Using the CLI installed by huggingface-hub
    # --local-dir-use-symlinks False ensures actual files are in the volume, not symlinks to a cache
    hf download $REPO_ID \
        --local-dir $MODEL_DIR
else
    echo "Model found at $MODEL_DIR. Skipping download."
fi

if [[ "$DEV_MODE" == "true" ]]; then
    echo "Running in development mode..."
    sleep infinity
else
    uv run ./python/vaml/train_rl.py
fi
