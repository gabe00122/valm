from typing import Protocol

from valm.buffer import UpdateBatch


class EpisodeListener(Protocol):
    def on_episodes(self, batch: UpdateBatch): ...
