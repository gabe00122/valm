from typing import Protocol

from vaml.buffer import UpdateBatch


class EpisodeListener(Protocol):
    def on_episodes(self, batch: UpdateBatch): ...
