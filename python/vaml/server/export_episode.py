import json
from pathlib import Path

from vaml.buffer import UpdateBatch
from vaml.server.episode_payload import (
    EPISODES_PER_FILE,
    EncodedEpisode,
    encode_episode,
)

RESULTS_DIR = Path("./results")
TOKENIZER_PATH = Path("./base-models/Qwen/Qwen3-4B-Instruct-2507")


def export_episode(
    out: Path,
    *,
    run: str,
    episode_id: int = 0,
) -> EncodedEpisode:
    """Export one episode from a run to a static JSON file."""
    if episode_id < 0:
        raise ValueError("--episode-id must be greater than or equal to 0")

    chunk = episode_id // EPISODES_PER_FILE
    episode_idx = episode_id % EPISODES_PER_FILE
    chunk_path = RESULTS_DIR / run / "rollouts" / f"episodes_{chunk}.npz"
    if not chunk_path.is_file():
        raise FileNotFoundError(f"Missing rollout chunk: {chunk_path}")

    from vaml.util import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    batch = UpdateBatch.load_npz(chunk_path)
    payload = encode_episode(batch, episode_idx, tokenizer)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return payload
