import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from vaml.buffer import UpdateBatch
from vaml.util import load_tokenizer

app = FastAPI()

tokenizer = load_tokenizer("./base-models/Qwen/Qwen3-4B-Instruct-2507")
results_dir = Path("./results")
episodes_per_file = 100


def _rollouts_dir(run_name: str) -> Path:
    rollouts = results_dir / run_name / "rollouts"
    if "/" in run_name or run_name in (".", "..") or not rollouts.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_name}")
    return rollouts


@lru_cache(maxsize=8)
def get_update_batch(run_name: str, chunk_id: int) -> UpdateBatch:
    path = _rollouts_dir(run_name) / f"episodes_{chunk_id}.npz"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing chunk: {path}")
    return UpdateBatch.load_npz(path)


def _base64_encode(array: np.ndarray) -> str:
    array = np.asarray(array, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("utf-8")


@app.get("/runs")
def list_runs(limit: int = 20):
    runs = [path for path in results_dir.iterdir() if (path / "rollouts").is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return {"runs": [path.name for path in runs[:limit]]}


@app.get("/runs/{run_name}")
def read_run(run_name: str):
    rollouts = _rollouts_dir(run_name)
    chunk_ids = [
        int(path.stem.removeprefix("episodes_"))
        for path in rollouts.glob("episodes_*.npz")
    ]
    if not chunk_ids:
        return {"name": run_name, "episodeCount": 0}

    last_chunk = max(chunk_ids)
    last_chunk_size = int(get_update_batch(run_name, last_chunk).context_length.shape[0])
    return {
        "name": run_name,
        "episodeCount": last_chunk * episodes_per_file + last_chunk_size,
    }


@app.get("/runs/{run_name}/episode/{episode_id}")
def read_item(run_name: str, episode_id: int):
    chunk = episode_id // episodes_per_file
    episode_idx = episode_id % episodes_per_file
    ub = get_update_batch(run_name, chunk)

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
