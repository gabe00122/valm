from typing import NamedTuple

import numpy as np


class UpdateBatch(NamedTuple):
    context: np.ndarray
    kv_cache_lengths: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    policy_mask: np.ndarray

    def save_npz(
        self,
        file,
        *,
        compressed: bool = True,
    ):
        """Save the batch to an .npz file."""

        payload = self._asdict()
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

            arrays = {k: data[k] for k in cls._fields}

        return cls(**arrays)


class CircularBuffer:
    def __init__(
        self, buffer_size: int, seq_shape: tuple[int, ...], dtype: np.typing.DTypeLike
    ) -> None:
        self._buffer_size = buffer_size

        self._size = 0
        self._start = 0
        self._end = 0

        self._data = np.zeros((buffer_size, *seq_shape), dtype=dtype)

    def push(self, data: np.ndarray):
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
    def __init__(self, buffer_size: int, batch_size: int, seq_length: int) -> None:
        self._batch_size = batch_size

        self._context = CircularBuffer(buffer_size, (seq_length,), np.int32)
        self._kv_cache_lengths = CircularBuffer(buffer_size, (), np.int32)
        self._log_probs = CircularBuffer(buffer_size, (seq_length - 1,), np.float32)
        self._values = CircularBuffer(buffer_size, (seq_length,), np.float32)
        self._rewards = CircularBuffer(buffer_size, (seq_length,), np.float32)
        self._policy_mask = CircularBuffer(buffer_size, (seq_length,), np.bool_)

    @property
    def size(self) -> int:
        return self._context._size

    @property
    def has_batch(self) -> bool:
        return self.size >= self._batch_size

    def store(self, batch: UpdateBatch):
        self._context.push(batch.context)
        self._kv_cache_lengths.push(batch.kv_cache_lengths)
        self._log_probs.push(batch.log_probs)
        self._values.push(batch.values)
        self._rewards.push(batch.rewards)
        self._policy_mask.push(batch.policy_mask)

    def take_batch(self) -> UpdateBatch:
        return UpdateBatch(
            context=self._context.pop_oldest(self._batch_size),
            kv_cache_lengths=self._kv_cache_lengths.pop_oldest(self._batch_size),
            log_probs=self._log_probs.pop_oldest(self._batch_size),
            values=self._values.pop_oldest(self._batch_size),
            rewards=self._rewards.pop_oldest(self._batch_size),
            policy_mask=self._policy_mask.pop_oldest(self._batch_size),
        )
