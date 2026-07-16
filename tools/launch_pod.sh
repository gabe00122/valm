#!/usr/bin/env bash
set -euo pipefail

# Launch a valm training run on runpod.
#
# usage: tools/launch_pod.sh <config> <wandb-tag> [extra 'pod create' flags...]
#
#   tools/launch_pod.sh configs/grpo.json grpo-baseline
#   tools/launch_pod.sh configs/test.json ppo-run --network-volume-id abc123
#
# <config> is a repo-relative path baked into the image (configs/foo.json) or
# an absolute path on an attached network volume (/workspace/configs/foo.json).
# Repo-relative configs are validated against the current schema before any
# money is spent.
#
# Any VALM_* variables set in the environment are forwarded to the pod, e.g.
#   VALM_VALUE_WARMUP=false tools/launch_pod.sh configs/mse.json ppo-cold-critic
#
# Overridable via environment:
#   IMAGE                  docker image (default: gabe00122/llm-rl:latest)
#   GPU_ID                 from 'runpodctl gpu list' (default: RTX 5090)
#   GPU_COUNT              (default: 1)
#   CLOUD_TYPE             SECURE or COMMUNITY (default: COMMUNITY)
#   DISK_GB                container disk in GB (default: 60)
#   MIN_CUDA               minimum host CUDA version (default: 13.0, required
#                          by the jax[cuda13] wheels in the image)
#   TERMINATE_AFTER_HOURS  hard-kill safety net for hung runs (default: 72)
#   NAME                   pod name (default: <config-stem>-<tag>)
#   WANDB_API_KEY          required
#   RUNPOD_API_KEY         falls back to ~/.runpod/config.toml
#   RUNPODCTL              path to runpodctl v2 (default: auto-detect)

usage() {
    sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 2 ]] || usage
CONFIG_ARG=$1
TAG=$2
shift 2

: "${WANDB_API_KEY:?set WANDB_API_KEY in your environment}"

IMAGE=${IMAGE:-docker.io/gabe0122/llmrl:latest}
GPU_ID=${GPU_ID:-"NVIDIA GeForce RTX 5090"}
GPU_COUNT=${GPU_COUNT:-1}
CLOUD_TYPE=${CLOUD_TYPE:-COMMUNITY}
DISK_GB=${DISK_GB:-60}
MIN_CUDA=${MIN_CUDA:-13.0}
TERMINATE_AFTER_HOURS=${TERMINATE_AFTER_HOURS:-72}

# The old v1 binary may still shadow v2 on PATH; prefer a known v2.
RUNPODCTL=${RUNPODCTL:-}
if [[ -z "$RUNPODCTL" ]]; then
    for candidate in "$HOME/.local/bin/runpodctl" runpodctl; do
        if command -v "$candidate" > /dev/null \
            && "$candidate" --version 2> /dev/null | grep -q "runpodctl 2"; then
            RUNPODCTL=$candidate
            break
        fi
    done
fi
[[ -n "$RUNPODCTL" ]] || {
    echo "runpodctl v2 not found (checked ~/.local/bin and PATH)" >&2
    exit 1
}

if [[ -z "${RUNPOD_API_KEY:-}" && -f "$HOME/.runpod/config.toml" ]]; then
    RUNPOD_API_KEY=$(grep -oP '(?<=api_key = ")[^"]*' "$HOME/.runpod/config.toml")
    export RUNPOD_API_KEY
fi
: "${RUNPOD_API_KEY:?no runpod api key in env or ~/.runpod/config.toml}"

if [[ "$CONFIG_ARG" == /* ]]; then
    # In-image or network-volume path; nothing to check locally.
    CONFIG_PATH=$CONFIG_ARG
else
    [[ -f "$CONFIG_ARG" ]] || {
        echo "config not found: $CONFIG_ARG" >&2
        exit 1
    }
    if [[ -f pyproject.toml ]] && command -v uv > /dev/null; then
        uv run python -c "
from pathlib import Path
from valm.config import load_config
load_config(Path('$CONFIG_ARG').read_text())
" || {
            echo "$CONFIG_ARG failed schema validation, not launching" >&2
            exit 1
        }
    fi
    CONFIG_PATH="/app/$CONFIG_ARG"
fi

NAME=${NAME:-"$(basename "$CONFIG_ARG" .json)-$TAG"}
TERMINATE_AT=$(date -u -d "+${TERMINATE_AFTER_HOURS} hours" +%Y-%m-%dT%H:%M:%SZ)
# Pod env: WANDB_API_KEY, the config/tag for this run, plus any VALM_*
# variables from the caller's environment (python handles the JSON escaping).
ENV_JSON=$(CONFIG_PATH="$CONFIG_PATH" TAG="$TAG" python3 - << 'EOF'
import json
import os

env = {key: value for key, value in os.environ.items() if key.startswith("VALM_")}
env["VALM_CONFIG"] = os.environ["CONFIG_PATH"]
env["VALM_WANDB_TAG"] = os.environ["TAG"]
env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
print(json.dumps(env))
EOF
)

echo "launching $NAME: $GPU_ID x$GPU_COUNT ($CLOUD_TYPE), $CONFIG_PATH, tag=$TAG"
echo "hard terminate at $TERMINATE_AT"

"$RUNPODCTL" pod create \
    --name "$NAME" \
    --image "$IMAGE" \
    --gpu-id "$GPU_ID" \
    --gpu-count "$GPU_COUNT" \
    --cloud-type "$CLOUD_TYPE" \
    --container-disk-in-gb "$DISK_GB" \
    --min-cuda-version "$MIN_CUDA" \
    --terminate-after "$TERMINATE_AT" \
    --env "$ENV_JSON" \
    "$@"

echo
echo "watch with: $RUNPODCTL pod list"
