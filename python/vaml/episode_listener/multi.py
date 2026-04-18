from vaml.buffer import UpdateBatch
from vaml.episode_listener.base import EpisodeListener


class MultiEpisodeListener(EpisodeListener):
    def __init__(self, listeners: list[EpisodeListener]):
        self._listeners = listeners

    def on_episodes(self, batch: UpdateBatch):
        for listener in self._listeners:
            listener.on_episodes(batch)
