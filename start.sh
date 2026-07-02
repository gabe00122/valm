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
    exec sleep infinity
fi

CONFIG="${VAML_CONFIG:-/app/configs/test.json}"

ARGS=(
    pipeline "$CONFIG"
    --offline-data "${VAML_OFFLINE_DATA:-./offline_data}"
    --offline-file-size "${VAML_OFFLINE_FILE_SIZE:-1000}"
    --offline-file-count "${VAML_OFFLINE_FILE_COUNT:-20}"
    --base-dir "${VAML_RESULTS_DIR:-results}"
)

if [[ -n "$VAML_OFFLINE_BATCH_SIZE" ]]; then
    ARGS+=(--offline-batch-size "$VAML_OFFLINE_BATCH_SIZE")
fi

# Remote runs keep only wandb curves plus the final checkpoint of each stage;
# set VAML_SAVE_ARTIFACTS=true for local-style periodic checkpoints and rollout logs.
if [[ "$VAML_SAVE_ARTIFACTS" != "true" ]]; then
    ARGS+=(--no-save-checkpoints --no-save-rollouts --no-track-values)
fi

if [[ -n "$VAML_WANDB_TAG" ]]; then
    ARGS+=(--wandb-tag "$VAML_WANDB_TAG")
fi

exec vaml "${ARGS[@]}"
