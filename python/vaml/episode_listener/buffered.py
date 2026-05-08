from vaml.buffer import UpdateBatch, UpdateBuffer
from vaml.episode_listener.base import EpisodeListener


class BufferedEpisodeListener(EpisodeListener):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seq_length: int,
        max_turns: int,
        listener: EpisodeListener,
    ):
        self._listener = listener
        self._buffer = UpdateBuffer(buffer_size, batch_size, seq_length, max_turns)

    @property
    def size(self) -> int:
        return self._buffer.size

    def on_episodes(self, batch: UpdateBatch):
        self._buffer.store(batch)
        while self._buffer.has_batch:
            self._listener.on_episodes(self._buffer.take_batch())
