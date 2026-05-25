import base64
from functools import cache

import numpy as np
from fastapi import FastAPI
from vaml.buffer import UpdateBatch
from vaml.util import load_tokenizer

app = FastAPI()

tokenizer = load_tokenizer("./base-models/Qwen/Qwen3-4B-Instruct-2507")
episodes_per_file = 100


@cache
def get_update_batch(chunk_id: int) -> UpdateBatch:
    return UpdateBatch.load_npz(
        f"./results/literate-marble-coati/rollouts/episodes_{chunk_id}.npz"
    )


def _base64_encode(array: np.ndarray) -> str:
    array = np.asarray(array, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("utf-8")


@app.get("/episode/{episode_id}")
def read_item(episode_id: int):
    chunk = episode_id // episodes_per_file
    episode_idx = episode_id % episodes_per_file
    ub = get_update_batch(chunk)

    length = int(ub.context_length[episode_idx])
    context = ub.context[episode_idx, :length].tolist()
    toks = [tokenizer.decode([token_id]) for token_id in context]

    token_metrics = {
        name: _base64_encode(np.asarray(value[episode_idx, :length], dtype=np.float32))
        for name, value in ub.update_metrics.items()
    }

    token_metrics["log_probs"] = _base64_encode(ub.log_probs[episode_idx, :length])
    token_metrics["rewards"] = _base64_encode(ub.rewards[episode_idx, :length])
    token_metrics["policy_mask"] = _base64_encode(ub.policy_mask[episode_idx, :length])

    return {
        "tokens": toks,
        "tokenMetrics": token_metrics,
    }
