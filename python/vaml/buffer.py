from collections.abc import Mapping
from typing import NamedTuple

import jax
import numpy as np

type ArrayData = np.ndarray | jax.Array


class UpdateBatch(NamedTuple):
    context_length: ArrayData
    context: ArrayData
    log_probs: ArrayData
    values: ArrayData
    rewards: ArrayData
    policy_mask: ArrayData

    turn_counts: ArrayData
    turn_start_positions: ArrayData
    metrics: Mapping[str, ArrayData]

    def save_npz(
        self,
        file,
        *,
        compressed: bool = True,
    ):
        """Save the batch to an .npz file."""

        payload = self._asdict()

        del payload["metrics"]
        for name, value in self.metrics.values():
            payload[f"metrics_{name}"] = value

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
            missing = [k for k in cls._fields if k not in data.files]
            if missing:
                raise KeyError(f"Missing keys in {file}: {missing}")

        data = {}
        metrics = {}

        for key in cls._fields:
            value = data[key]
            if key.startswith("metrics_"):
                metrics[key] = value
            else:
                data[key] = value

        return cls(metrics=metrics, **data)


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
        metric_names: list[str],
    ) -> None:
        self._batch_size = batch_size

        self._context_length = CircularBuffer(buffer_size, (), np.int32)
        self._context = CircularBuffer(buffer_size, (seq_length,), np.int32)
        self._log_probs = CircularBuffer(
            buffer_size, (seq_length - 1,), np.float32
        )
        self._values = CircularBuffer(buffer_size, (seq_length,), np.float32)
        self._rewards = CircularBuffer(buffer_size, (seq_length,), np.float32)
        self._policy_mask = CircularBuffer(buffer_size, (seq_length,), np.bool_)

        self._turn_counts = CircularBuffer(buffer_size, (max_turns,), np.int32)
        self._turn_start_positions = CircularBuffer(
            buffer_size, (max_turns,), np.int32
        )
        self._metrics = {
            name: CircularBuffer(buffer_size, (max_turns,), np.float32)
            for name in metric_names
        }

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
        self._values.push(batch.values)
        self._rewards.push(batch.rewards)
        self._policy_mask.push(batch.policy_mask)

        self._turn_counts.push(batch.turn_counts)
        self._turn_start_positions.push(batch.turn_start_positions)
        for name, buffer in self._metrics.items():
            buffer.push(batch.metrics[name])

    def take_batch(self) -> UpdateBatch:
        return UpdateBatch(
            context_length=self._context_length.pop_oldest(self._batch_size),
            context=self._context.pop_oldest(self._batch_size),
            log_probs=self._log_probs.pop_oldest(self._batch_size),
            values=self._values.pop_oldest(self._batch_size),
            rewards=self._rewards.pop_oldest(self._batch_size),
            policy_mask=self._policy_mask.pop_oldest(self._batch_size),
            turn_counts=self._turn_counts.pop_oldest(self._batch_size),
            turn_start_positions=self._turn_start_positions.pop_oldest(
                self._batch_size
            ),
            metrics={
                name: buffer.pop_oldest(self._batch_size)
                for name, buffer in self._metrics.items()
            },
        )
