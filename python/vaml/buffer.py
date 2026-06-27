from collections.abc import Mapping
from typing import NamedTuple, Iterator, cast

import numpy as np

type ArrayData = np.ndarray


# NOTE: I'm debating making this and the buffer more like dictionaries with losser typeing,
# I'm also thinking the buffer is probably better off lazy initalizing based on what fields exist on the update batch passed in.
# Passing in the env metric names is cumbersome
class UpdateBatch(NamedTuple):
    context_length: ArrayData
    context: ArrayData
    log_probs: ArrayData
    rewards: ArrayData
    policy_mask: ArrayData

    # these arrays are turn aligned not token aligned
    turn_counts: ArrayData
    turn_start_positions: ArrayData

    # per-episode GRPO group id (uint64). Optional + trailing default so old
    # rollout .npz files (saved without it) still load.
    group_id: ArrayData | None = None

    turn_metrics: Mapping[str, ArrayData] = {}

    # from the update step
    update_metrics: Mapping[str, ArrayData] = {}

    def save_npz(
        self,
        file,
        *,
        compressed: bool = True,
    ):
        """Save the batch to an .npz file."""

        payload = self._asdict()

        if self.group_id is None:
            del payload["group_id"]

        del payload["turn_metrics"]
        for name, value in self.turn_metrics.items():
            payload[f"turn_metrics_{name}"] = value

        del payload["update_metrics"]
        for name, value in self.update_metrics.items():
            payload[f"update_metrics_{name}"] = value

        if compressed:
            np.savez_compressed(file, **payload)
        else:
            np.savez(file, **payload)

    @classmethod
    def load_npz(
        cls,
        file,
    ) -> "UpdateBatch":
        """Load the batch from an .npz file."""
        with np.load(file, allow_pickle=False) as data:
            fields = {}
            turn_metrics = {}
            update_metrics = {}
            batch_fields = set(cls._fields) - {"turn_metrics", "update_metrics"}

            for key, value in data.items():
                if key.startswith("turn_metrics_"):
                    turn_metrics[key[len("turn_metrics_") :]] = value
                elif key.startswith("update_metrics_"):
                    update_metrics[key[len("update_metrics_") :]] = value
                elif key in batch_fields:
                    fields[key] = value

            return cls(
                turn_metrics=turn_metrics, update_metrics=update_metrics, **fields
            )

_BATCH_FIELDS = (
    "context_length", "context", "log_probs", "rewards",
    "policy_mask", "turn_counts", "turn_start_positions", "group_id",
)

_SEQUENCE_FIELDS = (
    "context", "rewards", "policy_mask"
)

def array_chunks(batch: UpdateBatch, chunk_size: int) -> Iterator[UpdateBatch]:
    n = batch.context.shape[0]
    for start in range(0, n, chunk_size):
        sl = slice(start, start + chunk_size)
        updates = {
            f: getattr(batch, f)[sl]
            for f in _BATCH_FIELDS
            if getattr(batch, f) is not None
        }
        updates["turn_metrics"] = {k: v[sl] for k, v in batch.turn_metrics.items()}
        updates["update_metrics"] =  {k: v[sl] for k, v in batch.update_metrics.items()}
        yield bucket_chunk(batch._replace(**updates))


def concat_batches(batches: list[UpdateBatch]) -> UpdateBatch:
    """Concatenate UpdateBatches along the batch (episode) axis."""
    first = batches[0]
    fields = {
        f: np.concatenate([getattr(b, f) for b in batches], axis=0)
        for f in _BATCH_FIELDS
        if getattr(first, f) is not None
    }
    fields["turn_metrics"] = {
        k: np.concatenate([b.turn_metrics[k] for b in batches], axis=0)
        for k in first.turn_metrics
    }
    fields["update_metrics"] = {
        k: np.concatenate([b.update_metrics[k] for b in batches], axis=0)
        for k in first.update_metrics
    }
    return UpdateBatch(**fields)


def bucket_chunk(batch: UpdateBatch) -> UpdateBatch:
    length = np.max(batch.context_length).item()
    bucket = 1 << length.bit_length()
    bucket = max(128, bucket)
    sl = slice(bucket)
    updates = {f: getattr(batch, f)[:, sl] for f in _SEQUENCE_FIELDS}
    updates["log_probs"] = batch.log_probs[:, :bucket - 1]
    updates["update_metrics"] =  {k: v[:, sl] for k, v in batch.update_metrics.items()}
    return batch._replace(**updates)


class CircularBuffer:
    def __init__(
        self,
        buffer_size: int,
        seq_shape: tuple[int, ...],
        dtype: np.typing.DTypeLike,
    ) -> None:
        self._buffer_size = buffer_size

        self._size = 0
        self._start = 0
        self._end = 0

        self._data = np.zeros((buffer_size, *seq_shape), dtype=dtype)

    def push(self, data: ArrayData):
        data = np.asarray(data)
        start = self._end
        end = start + data.shape[0]

        if end > self._buffer_size:
            overflow = end - self._buffer_size
            self._data[start:] = data[:-overflow]
            self._data[:overflow] = data[-overflow:]
        else:
            self._data[start:end] = data

        self._end = end % self._buffer_size
        self._size = self._size + data.shape[0]

        if self._size > self._buffer_size:
            self._start = self._end
            self._size = self._buffer_size

    def pop_oldest(self, num: int):
        start = self._start
        end = start + num

        if end > self._buffer_size:
            overflow = end - self._buffer_size
            part1 = self._data[start:]
            part2 = self._data[:overflow]

            out = np.concatenate((part1, part2), axis=0)
        else:
            out = self._data[start:end]

        self._start = end % self._buffer_size
        self._size -= num

        return out


class UpdateBuffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seq_length: int,
        max_turns: int,
    ) -> None:
        self._buffer_size = buffer_size
        self._batch_size = batch_size

        self._context_length = CircularBuffer(buffer_size, (), np.int32)
        self._context = CircularBuffer(buffer_size, (seq_length,), np.int32)
        self._log_probs = CircularBuffer(buffer_size, (seq_length - 1,), np.float32)
        self._rewards = CircularBuffer(buffer_size, (seq_length,), np.float32)
        self._policy_mask = CircularBuffer(buffer_size, (seq_length,), np.bool_)

        self._turn_counts = CircularBuffer(buffer_size, (), np.int32)
        self._turn_start_positions = CircularBuffer(buffer_size, (max_turns,), np.int32)
        self._group_id = CircularBuffer(buffer_size, (), np.uint64)
        self._turn_metrics = {}

        self._update_metrics = {}

    @property
    def size(self) -> int:
        return self._context._size

    @property
    def has_batch(self) -> bool:
        return self.size >= self._batch_size

    def store(self, batch: UpdateBatch):
        self._context_length.push(batch.context_length)
        self._context.push(batch.context)
        self._log_probs.push(batch.log_probs)
        self._rewards.push(batch.rewards)
        self._policy_mask.push(batch.policy_mask)

        self._turn_counts.push(batch.turn_counts)
        self._turn_start_positions.push(batch.turn_start_positions)
        # Tolerate batches without a group id (e.g. offline .npz saved before
        # group ids existed) so value-only training keeps working.
        group_id = batch.group_id
        if group_id is None:
            group_id = np.zeros(batch.context_length.shape[0], dtype=np.uint64)
        self._group_id.push(group_id)

        for name, value in batch.turn_metrics.items():
            if name not in self._turn_metrics:
                self._turn_metrics[name] = CircularBuffer(
                    self._buffer_size, value.shape[1:], value.dtype
                )

            self._turn_metrics[name].push(value)

        for name, value in batch.update_metrics.items():
            if name not in self._update_metrics:
                self._update_metrics[name] = CircularBuffer(
                    self._buffer_size, value.shape[1:], value.dtype
                )

            self._update_metrics[name].push(value)

    def take_batch(self) -> UpdateBatch:
        return UpdateBatch(
            context_length=self._context_length.pop_oldest(self._batch_size),
            context=self._context.pop_oldest(self._batch_size),
            log_probs=self._log_probs.pop_oldest(self._batch_size),
            rewards=self._rewards.pop_oldest(self._batch_size),
            policy_mask=self._policy_mask.pop_oldest(self._batch_size),
            turn_counts=self._turn_counts.pop_oldest(self._batch_size),
            turn_start_positions=self._turn_start_positions.pop_oldest(
                self._batch_size
            ),
            group_id=self._group_id.pop_oldest(self._batch_size),
            turn_metrics={
                name: buffer.pop_oldest(self._batch_size)
                for name, buffer in self._turn_metrics.items()
            },
            update_metrics={
                name: buffer.pop_oldest(self._batch_size)
                for name, buffer in self._update_metrics.items()
            },
        )
