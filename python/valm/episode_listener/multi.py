from valm.buffer import UpdateBatch
from valm.episode_listener.base import EpisodeListener


class MultiEpisodeListener(EpisodeListener):
    def __init__(self, *listeners: EpisodeListener):
        self._listeners = listeners

    def on_episodes(self, batch: UpdateBatch):
        for listener in self._listeners:
            listener.on_episodes(batch)
