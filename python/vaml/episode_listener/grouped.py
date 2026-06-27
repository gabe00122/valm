from vaml.buffer import _BATCH_FIELDS, UpdateBatch, concat_batches
from vaml.episode_listener.base import EpisodeListener


def _split_rows(batch: UpdateBatch):
    """Yield each episode of a batch as its own single-row UpdateBatch."""
    n = batch.context.shape[0]
    for i in range(n):
        sl = slice(i, i + 1)
        yield batch._replace(
            **{
                f: getattr(batch, f)[sl]
                for f in _BATCH_FIELDS
                if getattr(batch, f) is not None
            },
            turn_metrics={k: v[sl] for k, v in batch.turn_metrics.items()},
            update_metrics={k: v[sl] for k, v in batch.update_metrics.items()},
        )


class GroupedEpisodeListener(EpisodeListener):
    """Reconstruct GRPO groups by id before forwarding to the next listener.

    Episodes that share a ``group_id`` are completions of the same problem.
    Group members can finish on different iterations (variable-length multi-turn
    envs), so their episodes arrive interleaved and out of order. This listener
    keys episodes by id and only forwards a group once all ``group_size`` members
    have arrived. Emitted batches always contain whole, contiguous groups, so a
    downstream GRPO loss can ``rearrange("(g k) ... -> g k ...", k=group_size)``.

    With ``group_size == 1`` each episode is its own group and is forwarded in
    arrival order, matching the behaviour of the FIFO ``UpdateBuffer``.
    """

    def __init__(self, group_size: int, batch_size: int, listener: EpisodeListener):
        if batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of "
                f"group_size ({group_size})"
            )
        self._group_size = group_size
        self._batch_size = batch_size
        self._listener = listener

        # group id -> rows seen so far for that (not-yet-complete) group
        self._pending: dict[int, list[UpdateBatch]] = {}
        # rows belonging to completed groups, waiting to fill a batch
        self._ready: list[UpdateBatch] = []

    @property
    def size(self) -> int:
        return len(self._ready)

    def on_episodes(self, batch: UpdateBatch):
        if batch.group_id is None:
            raise ValueError("GroupedEpisodeListener requires batches with group_id set")

        for row in _split_rows(batch):
            gid = int(row.group_id[0])
            members = self._pending.setdefault(gid, [])
            members.append(row)
            if len(members) >= self._group_size:
                del self._pending[gid]
                self._ready.extend(members)

        while len(self._ready) >= self._batch_size:
            out_rows = self._ready[: self._batch_size]
            del self._ready[: self._batch_size]
            self._listener.on_episodes(concat_batches(out_rows))
