#!/bin/bash
set -e

# When the start command exits, runpod RESTARTS the container instead of
# stopping the pod, so on runpod this script must never exit — it has to kill
# its own pod through the API and wait to be reaped. The injected pod-scoped
# RUNPOD_API_KEY is only authorized on the GraphQL API (the REST API, and
# therefore runpodctl v2, answers 403), so call GraphQL directly: terminate
# the pod on success, stop it on failure (a stopped pod keeps its console
# logs for debugging and bills only for disk). Set VAML_KEEP_ALIVE=true to
# skip self-shutdown.
finish() {
    status=$?
    if [[ "$VAML_KEEP_ALIVE" == "true" || -z "$RUNPOD_POD_ID" ]]; then
        exit "$status"
    fi
    touch /tmp/vaml-finished || true
    if [[ "$status" -eq 0 ]]; then
        query='mutation { podTerminate(input: {podId: "'"$RUNPOD_POD_ID"'"}) }'
    else
        query='mutation { podStop(input: {podId: "'"$RUNPOD_POD_ID"'"}) { id desiredStatus } }'
    fi
    echo "requesting pod shutdown (exit status $status)"
    while true; do
        curl -sS --request POST \
            --header 'content-type: application/json' \
            --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
            --data '{"query": "'"${query//\"/\\\"}"'"}' || true
        echo
        sleep 60
    done
}
trap finish EXIT

# If a previous run in this pod already finished, the container restarted
# before the shutdown request landed; stop the pod instead of retraining.
if [[ -n "$RUNPOD_POD_ID" && -f /tmp/vaml-finished ]]; then
    echo "found /tmp/vaml-finished from an earlier run; shutting down instead of retraining"
    exit 1
fi

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
    --offline-file-count "${VAML_OFFLINE_FILE_COUNT:-40}"
    --base-dir "${VAML_RESULTS_DIR:-results}"
)

if [[ -n "$VAML_OFFLINE_BATCH_SIZE" ]]; then
    ARGS+=(--offline-batch-size "$VAML_OFFLINE_BATCH_SIZE")
fi

# PPO ablation: train the critic from scratch online instead of pretraining
# it on offline data.
if [[ "$VAML_VALUE_WARMUP" == "false" ]]; then
    ARGS+=(--no-value-warmup)
fi

# Remote runs keep only wandb curves plus the final checkpoint of each stage;
# set VAML_SAVE_ARTIFACTS=true for local-style periodic checkpoints and rollout logs.
if [[ "$VAML_SAVE_ARTIFACTS" != "true" ]]; then
    ARGS+=(--no-save-checkpoints --no-save-rollouts --no-track-values)
fi

if [[ -n "$VAML_WANDB_TAG" ]]; then
    ARGS+=(--wandb-tag "$VAML_WANDB_TAG")
fi

vaml "${ARGS[@]}"
