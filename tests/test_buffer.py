import numpy as np
import pytest
from vaml.buffer import (
    CircularBuffer,
    UpdateBatch,
    UpdateBuffer,
    array_chunks,
    bucket_chunk,
)


def test_push_and_pop():
    """Test basic push and pop operations."""
    buffer = CircularBuffer(buffer_size=4, seq_shape=(2,), dtype=np.int32)

    buffer.push(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32))
    result = buffer.pop_oldest(2)

    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_wrap_around():
    """Test circular wrap-around behavior."""
    buffer = CircularBuffer(buffer_size=3, seq_shape=(1,), dtype=np.int32)

    # Fill and overflow
    buffer.push(np.array([[1], [2], [3]], dtype=np.int32))
    buffer.push(np.array([[4]], dtype=np.int32))

    # Oldest (1) should be overwritten, so we get 2, 3, 4
    result = buffer.pop_oldest(3)
    np.testing.assert_array_equal(result, [[2], [3], [4]])


def test_push_pop_cycle():
    """Test interleaved push/pop operations."""
    buffer = CircularBuffer(buffer_size=4, seq_shape=(1,), dtype=np.int32)

    buffer.push(np.array([[1], [2]], dtype=np.int32))
    buffer.pop_oldest(1)
    buffer.push(np.array([[3], [4]], dtype=np.int32))

    result = buffer.pop_oldest(3)
    np.testing.assert_array_equal(result, [[2], [3], [4]])


def test_update_buffer_lazily_initializes_turn_metrics():
    """Test that turn metric buffers are created from stored batch data."""
    buffer = UpdateBuffer(buffer_size=3, batch_size=2, seq_length=4, max_turns=2)
    turn_metrics = {
        "score": np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
    }
    batch = UpdateBatch(
        context_length=np.array([3, 4], dtype=np.int32),
        context=np.array([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=np.int32),
        log_probs=np.zeros((2, 3), dtype=np.float32),
        rewards=np.ones((2, 4), dtype=np.float32),
        policy_mask=np.array(
            [[True, True, True, False], [True, True, True, True]],
            dtype=np.bool_,
        ),
        turn_counts=np.array([1, 2], dtype=np.int32),
        turn_start_positions=np.array([[0, 2], [0, 1]], dtype=np.int32),
        turn_metrics=turn_metrics,
    )

    buffer.store(batch)
    result = buffer.take_batch()

    assert result.turn_metrics.keys() == turn_metrics.keys()
    np.testing.assert_array_equal(result.turn_metrics["score"], turn_metrics["score"])


def _buffer_shaped_batch(
    context_lengths,
    *,
    seq_length,
    max_turns,
    with_metrics=True,
):
    """Build an UpdateBatch whose per-field shapes match what an UpdateBuffer with
    the given ``seq_length`` / ``max_turns`` produces (see ``UpdateBuffer.__init__``):

    - context / rewards / policy_mask : (n, seq_length)      -- token aligned
    - log_probs                       : (n, seq_length - 1)  -- token aligned
    - context_length / turn_counts    : (n,)
    - turn_start_positions            : (n, max_turns)       -- turn aligned
    - turn_metrics["score"]           : (n, max_turns)       -- turn aligned
    - update_metrics["value"]         : (n, seq_length + 1)  -- token aligned
      (update_step.py pads token metrics to seq_length + 1)

    Cell (r, c) == r * 1000 + c so cropped columns are easy to eyeball.
    """
    context_lengths = np.asarray(context_lengths, dtype=np.int32)
    n = context_lengths.shape[0]
    grid = np.arange(n)[:, None] * 1000 + np.arange(seq_length)[None, :]

    turn_metrics = {}
    update_metrics = {}
    if with_metrics:
        turn_metrics = {
            "score": np.arange(n * max_turns, dtype=np.float32).reshape(n, max_turns)
        }
        update_metrics = {
            "value": np.arange(n * (seq_length + 1), dtype=np.float32).reshape(
                n, seq_length + 1
            )
        }

    return UpdateBatch(
        context_length=context_lengths,
        context=grid.astype(np.int32),
        log_probs=grid[:, : seq_length - 1].astype(np.float32),
        rewards=grid.astype(np.float32),
        policy_mask=(grid % 2 == 0),
        turn_counts=np.ones(n, dtype=np.int32),
        turn_start_positions=np.zeros((n, max_turns), dtype=np.int32),
        turn_metrics=turn_metrics,
        update_metrics=update_metrics,
    )


def test_array_chunks_splits_along_batch_dim():
    """array_chunks splits axis 0 into chunk_size pieces (last one smaller)."""
    batch = _buffer_shaped_batch([4, 4, 4, 4, 4], seq_length=8, max_turns=3)

    chunks = list(array_chunks(batch, chunk_size=2))

    # ceil(5 / 2) == 3 chunks; the trailing remainder chunk is smaller.
    assert [c.context.shape[0] for c in chunks] == [2, 2, 1]

    # seq_length (8) is below the minimum bucket (128), so no sequence cropping
    # happens and concatenating the chunks must reconstruct the originals exactly.
    np.testing.assert_array_equal(
        np.concatenate([c.context for c in chunks], axis=0), batch.context
    )
    np.testing.assert_array_equal(
        np.concatenate([c.log_probs for c in chunks], axis=0), batch.log_probs
    )
    np.testing.assert_array_equal(
        np.concatenate([c.context_length for c in chunks], axis=0),
        batch.context_length,
    )
    np.testing.assert_array_equal(
        np.concatenate([c.turn_metrics["score"] for c in chunks], axis=0),
        batch.turn_metrics["score"],
    )
    np.testing.assert_array_equal(
        np.concatenate([c.update_metrics["value"] for c in chunks], axis=0),
        batch.update_metrics["value"],
    )


def test_bucket_chunk_crops_sequence_to_next_power_of_two():
    """Token-aligned fields are cropped to the next power of two >= max length."""
    # max context length 200 -> next power of two is 256.
    batch = _buffer_shaped_batch([50, 200, 100], seq_length=300, max_turns=4)

    out = bucket_chunk(batch)

    assert out.context.shape == (3, 256)
    assert out.rewards.shape == (3, 256)
    assert out.policy_mask.shape == (3, 256)
    assert out.log_probs.shape == (3, 255)  # one shorter than context
    assert out.update_metrics["value"].shape == (3, 256)

    # Cropping keeps the leading columns untouched.
    np.testing.assert_array_equal(out.context, batch.context[:, :256])
    np.testing.assert_array_equal(out.log_probs, batch.log_probs[:, :255])

    # bucket >= max context_length, so no valid (non-pad) token is dropped.
    assert 256 >= int(np.max(batch.context_length))

    # Scalar / turn-aligned fields are left untouched.
    assert out.context_length.shape == (3,)
    assert out.turn_counts.shape == (3,)
    assert out.turn_start_positions.shape == (3, 4)
    assert out.turn_metrics["score"].shape == (3, 4)
    np.testing.assert_array_equal(
        out.turn_metrics["score"], batch.turn_metrics["score"]
    )


def test_bucket_chunk_has_minimum_width_of_128():
    """The bucket is floored at 128 even for short sequences."""
    batch = _buffer_shaped_batch([10, 30, 5], seq_length=300, max_turns=2)

    out = bucket_chunk(batch)

    assert out.context.shape == (3, 128)
    assert out.log_probs.shape == (3, 127)


def test_bucket_chunk_caps_but_never_pads():
    """When the array is narrower than the bucket the slice is a no-op (no padding)."""
    # max context length 200 -> bucket 256, but the arrays are only 200 wide.
    batch = _buffer_shaped_batch([200, 150], seq_length=200, max_turns=2)

    out = bucket_chunk(batch)

    assert out.context.shape == (2, 200)
    assert out.log_probs.shape == (2, 199)


def test_array_chunks_matches_update_buffer_output_shapes():
    """End to end: store -> take_batch -> array_chunks keeps shapes consistent."""
    seq_length = 300
    buffer = UpdateBuffer(
        buffer_size=8, batch_size=4, seq_length=seq_length, max_turns=3
    )
    buffer.store(
        _buffer_shaped_batch([60, 200, 70, 130], seq_length=seq_length, max_turns=3)
    )

    batch = buffer.take_batch()
    chunks = list(array_chunks(batch, chunk_size=2))

    assert [c.context.shape[0] for c in chunks] == [2, 2]
    for chunk in chunks:
        bucket = chunk.context.shape[1]
        assert bucket == 256  # chunk maxes are 200 and 130 -> both bucket to 256
        assert chunk.log_probs.shape[1] == bucket - 1
        assert chunk.rewards.shape[1] == bucket
        assert chunk.policy_mask.shape[1] == bucket
        assert chunk.update_metrics["value"].shape[1] == bucket
        # turn-aligned data keeps its full width.
        assert chunk.turn_start_positions.shape[1] == 3
        assert chunk.turn_metrics["score"].shape[1] == 3


def test_bucket_chunk_requires_token_aligned_update_metrics():
    """Documents the current contract: bucket_chunk slices update_metrics on axis 1
    (``v[:, :bucket]``), so a scalar-per-sample (1-D) update metric raises instead of
    passing through."""
    batch = _buffer_shaped_batch([10, 20], seq_length=200, max_turns=2, with_metrics=False)
    batch = batch._replace(update_metrics={"scalar": np.zeros(2, dtype=np.float32)})

    with pytest.raises(IndexError):
        bucket_chunk(batch)
