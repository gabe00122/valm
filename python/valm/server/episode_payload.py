import base64
from typing import Protocol, TypedDict

import numpy as np
from valm.buffer import UpdateBatch

# This constant is a liability
EPISODES_PER_FILE = 100


class TokenDecoder(Protocol):
    def decode(self, token_ids: list[int]) -> str: ...


class EncodedEpisode(TypedDict):
    tokens: list[str]
    tokenMetrics: dict[str, str]


def _base64_encode(array: np.ndarray) -> str:
    array = np.asarray(array, dtype=np.float32)
    return base64.b64encode(array.tobytes()).decode("utf-8")


def encode_episode(
    batch: UpdateBatch,
    episode_idx: int,
    tokenizer: TokenDecoder,
) -> EncodedEpisode:
    chunk_size = int(batch.context_length.shape[0])
    if not 0 <= episode_idx < chunk_size:
        raise IndexError(
            f"episode index {episode_idx} out of range (chunk has {chunk_size} episodes)"
        )

    length = int(batch.context_length[episode_idx])
    context = batch.context[episode_idx, :length].tolist()
    tokens = [tokenizer.decode([token_id]) for token_id in context]

    token_metrics = {
        name: _base64_encode(np.asarray(value[episode_idx, :length], dtype=np.float32))
        for name, value in batch.update_metrics.items()
    }
    token_metrics["log_probs"] = _base64_encode(batch.log_probs[episode_idx, :length])
    token_metrics["rewards"] = _base64_encode(batch.rewards[episode_idx, :length])
    token_metrics["policy_mask"] = _base64_encode(
        batch.policy_mask[episode_idx, :length]
    )

    return {"tokens": tokens, "tokenMetrics": token_metrics}
