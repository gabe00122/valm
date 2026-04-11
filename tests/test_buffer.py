import numpy as np
from vaml.buffer import CircularBuffer


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
