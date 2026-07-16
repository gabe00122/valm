import numpy as np
import pytest
from valm.buffer import UpdateBatch
from valm.episode_listener.grouped import GroupedEpisodeListener


class _Collector:
    def __init__(self):
        self.batches: list[UpdateBatch] = []

    def on_episodes(self, batch: UpdateBatch):
        self.batches.append(batch)


def _batch(group_ids, tags, seq_length=4, max_turns=2):
    """A minimal batch where each row's `context_length` is a unique tag."""
    n = len(group_ids)
    return UpdateBatch(
        context_length=np.asarray(tags, dtype=np.int32),
        context=np.zeros((n, seq_length), dtype=np.int32),
        log_probs=np.zeros((n, seq_length - 1), dtype=np.float32),
        rewards=np.zeros((n, seq_length), dtype=np.float32),
        policy_mask=np.zeros((n, seq_length), dtype=np.bool_),
        turn_counts=np.zeros(n, dtype=np.int32),
        turn_start_positions=np.zeros((n, max_turns), dtype=np.int32),
        group_id=np.asarray(group_ids, dtype=np.uint64),
    )


def test_groups_emit_only_when_complete_and_contiguous():
    """Interleaved, out-of-order members are paired by id; emitted batches hold
    whole, contiguous groups in completion order."""
    sink = _Collector()
    listener = GroupedEpisodeListener(group_size=2, batch_size=4, listener=sink)

    # Members of groups 10/11/12 arrive interleaved across iterations.
    listener.on_episodes(_batch([10, 11], [1, 2]))
    assert sink.batches == []  # nothing complete yet

    listener.on_episodes(_batch([11, 12], [3, 4]))  # group 11 completes (tags 2,3)
    assert sink.batches == []  # only 2 ready < batch_size

    listener.on_episodes(_batch([10, 12], [5, 6]))  # groups 10 and 12 complete
    # ready = [t2,t3 (g11), t1,t5 (g10), t4,t6 (g12)] -> emit first 4
    assert len(sink.batches) == 1
    out = sink.batches[0]
    np.testing.assert_array_equal(out.context_length, [2, 3, 1, 5])
    np.testing.assert_array_equal(out.group_id, [11, 11, 10, 10])

    # consecutive group_size rows share an id (contiguous groups)
    gid = out.group_id.reshape(-1, 2)
    assert np.all(gid[:, 0] == gid[:, 1])


def test_incomplete_groups_never_emit():
    sink = _Collector()
    listener = GroupedEpisodeListener(group_size=3, batch_size=3, listener=sink)

    listener.on_episodes(_batch([5, 6, 5, 6], [1, 2, 3, 4]))  # 2 of each, none of 3
    assert sink.batches == []


def test_group_size_one_is_fifo():
    sink = _Collector()
    listener = GroupedEpisodeListener(group_size=1, batch_size=2, listener=sink)

    listener.on_episodes(_batch([1, 2, 3], [10, 20, 30]))
    # each episode is its own group; first 2 emitted, third held back
    assert len(sink.batches) == 1
    np.testing.assert_array_equal(sink.batches[0].context_length, [10, 20])


def test_batch_size_must_be_multiple_of_group_size():
    with pytest.raises(ValueError):
        GroupedEpisodeListener(group_size=3, batch_size=4, listener=_Collector())
