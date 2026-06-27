import base64

import numpy as np
import pytest

from vaml.buffer import UpdateBatch
from vaml.server.episode_payload import encode_episode


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"tok-{token_ids[0]}"


def _decode_metric(payload, name: str) -> np.ndarray:
    raw = base64.b64decode(payload["tokenMetrics"][name])
    return np.frombuffer(raw, dtype=np.float32)


def test_encode_episode_matches_server_payload_shape():
    batch = UpdateBatch(
        context_length=np.array([2], dtype=np.int32),
        context=np.array([[10, 11, 12, 0]], dtype=np.int32),
        log_probs=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        rewards=np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32),
        policy_mask=np.array([[True, False, True, False]], dtype=np.bool_),
        turn_counts=np.array([1], dtype=np.int32),
        turn_start_positions=np.array([[0]], dtype=np.int32),
        update_metrics={
            "values": np.array([[4.0, 5.0, 6.0, 0.0]], dtype=np.float32)
        },
    )

    payload = encode_episode(batch, 0, FakeTokenizer())

    assert payload["tokens"] == ["tok-10", "tok-11", "tok-12"]
    np.testing.assert_array_equal(
        _decode_metric(payload, "values"), [4.0, 5.0, 6.0]
    )
    np.testing.assert_allclose(_decode_metric(payload, "log_probs"), [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(_decode_metric(payload, "rewards"), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        _decode_metric(payload, "policy_mask"), [1.0, 0.0, 1.0]
    )


def test_encode_episode_rejects_out_of_range_episode_idx():
    batch = UpdateBatch(
        context_length=np.array([1], dtype=np.int32),
        context=np.array([[10, 11]], dtype=np.int32),
        log_probs=np.array([[0.1]], dtype=np.float32),
        rewards=np.array([[1.0, 2.0]], dtype=np.float32),
        policy_mask=np.array([[True, True]], dtype=np.bool_),
        turn_counts=np.array([1], dtype=np.int32),
        turn_start_positions=np.array([[0]], dtype=np.int32),
    )

    with pytest.raises(IndexError, match="out of range"):
        encode_episode(batch, 1, FakeTokenizer())
