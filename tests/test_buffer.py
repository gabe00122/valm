import numpy as np
from vaml.buffer import CircularBuffer, UpdateBatch, UpdateBuffer


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
        values=np.zeros((2, 4), dtype=np.float32),
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
