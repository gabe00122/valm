import numpy as np
from vaml.agent.local import TurnData


def test_turn_data_lazily_initializes_metrics():
    turn_data = TurnData.create(eval_envs=3, max_turns=2)

    turn_data.update(
        np.array([0, 1, 2], dtype=np.int32),
        np.array([10, 11, 12], dtype=np.int32),
        {"score": np.array([0.0, 0.0, 0.0], dtype=np.float32)},
    )
    turn_data.update(
        np.array([0, 2], dtype=np.int32),
        np.array([20, 22], dtype=np.int32),
        {
            "score": np.array([1.5, 2.5], dtype=np.float32),
            "complete": np.array([1, 0], dtype=np.int32),
        },
    )

    turn_counts, turn_start_positions, metrics = turn_data.take(
        np.array([0, 2], dtype=np.int32)
    )

    np.testing.assert_array_equal(turn_counts, [1, 1])
    np.testing.assert_array_equal(turn_start_positions, [[20, 0], [22, 0]])
    assert metrics.keys() == {"score", "complete"}
    np.testing.assert_array_equal(metrics["score"], [[1.5, 0.0], [2.5, 0.0]])
    np.testing.assert_array_equal(metrics["complete"], [[1, 0], [0, 0]])
