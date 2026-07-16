import jax.numpy as jnp
import numpy as np
from valm.util import batched_put, batched_take


def test_batched_put():
    # Setup
    B, S, H, D = 2, 4, 1, 2
    target = jnp.zeros((B, S, H, D), dtype=jnp.float32)

    # Update at index 1 for batch 0, index 2 for batch 1
    indices = jnp.array([[1], [2]], dtype=jnp.int32)  # Shape (B, 1)

    # Values to insert
    # Shape (B, 1, H, D)
    values = jnp.array([[[[10.0, 11.0]]], [[[20.0, 21.0]]]], dtype=jnp.float32)

    # Execute
    result = batched_put(target, indices, values)

    # Verification
    expected = np.zeros((B, S, H, D), dtype=np.float32)
    expected[0, 1, 0, :] = [10.0, 11.0]
    expected[1, 2, 0, :] = [20.0, 21.0]

    np.testing.assert_array_equal(result, expected)

    # Test with different values to ensure no cross-batch contamination
    assert result[0, 2, 0, 0] == 0.0
    assert result[1, 1, 0, 0] == 0.0


def test_batched_put_multiple_indices():
    # Setup
    B, S, H, D = 2, 4, 1, 2
    target = jnp.zeros((B, S, H, D), dtype=jnp.float32)

    # Update at indices [1, 3] for batch 0, indices [0, 2] for batch 1
    indices = jnp.array([[1, 3], [0, 2]], dtype=jnp.int32)  # Shape (B, 2)

    # Values to insert
    # Shape (B, 2, H, D)
    values = jnp.array(
        [
            [
                [[11.0, 12.0]],  # batch 0, index 1
                [[13.0, 14.0]],  # batch 0, index 3
            ],
            [
                [[21.0, 22.0]],  # batch 1, index 0
                [[23.0, 24.0]],  # batch 1, index 2
            ],
        ],
        dtype=jnp.float32,
    )

    # Execute
    result = batched_put(target, indices, values)

    # Verification
    expected = np.zeros((B, S, H, D), dtype=np.float32)
    expected[0, 1, 0, :] = [11.0, 12.0]
    expected[0, 3, 0, :] = [13.0, 14.0]
    expected[1, 0, 0, :] = [21.0, 22.0]
    expected[1, 2, 0, :] = [23.0, 24.0]

    np.testing.assert_array_equal(result, expected)


def test_batched_take():
    # Setup
    B, S, H, D = 2, 4, 1, 2
    target = jnp.zeros((B, S, H, D), dtype=jnp.float32)

    # Fill target with identifiable values
    # Batch 0:
    #   idx 0: [0, 1]
    #   idx 1: [2, 3]
    #   idx 2: [4, 5]
    #   idx 3: [6, 7]
    target = target.at[0, 0, 0].set([0.0, 1.0])
    target = target.at[0, 1, 0].set([2.0, 3.0])
    target = target.at[0, 2, 0].set([4.0, 5.0])
    target = target.at[0, 3, 0].set([6.0, 7.0])

    # Batch 1:
    #   idx 0: [10, 11]
    #   idx 1: [12, 13]
    #   idx 2: [14, 15]
    #   idx 3: [16, 17]
    target = target.at[1, 0, 0].set([10.0, 11.0])
    target = target.at[1, 1, 0].set([12.0, 13.0])
    target = target.at[1, 2, 0].set([14.0, 15.0])
    target = target.at[1, 3, 0].set([16.0, 17.0])

    # Indices to take
    # Batch 0: indices 1, 3
    # Batch 1: indices 0, 2
    indices = jnp.array([[1, 3], [0, 2]], dtype=jnp.int32)  # Shape (B, 2)

    # Execute
    result = batched_take(target, indices)

    # Verification
    expected = np.zeros((B, 2, H, D), dtype=np.float32)

    # Batch 0
    expected[0, 0, 0] = [2.0, 3.0]  # from idx 1
    expected[0, 1, 0] = [6.0, 7.0]  # from idx 3

    # Batch 1
    expected[1, 0, 0] = [10.0, 11.0]  # from idx 0
    expected[1, 1, 0] = [14.0, 15.0]  # from idx 2

    np.testing.assert_array_equal(result, expected)
