import os
from pathlib import Path

from vaml.buffer import UpdateBatch
from vaml.episode_listener.base import EpisodeListener


class EpisodeSaver(EpisodeListener):
    def __init__(self, directory: str):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self.chunk_num = 0

    def on_episodes(self, batch: UpdateBatch):
        file_name = os.path.join(self._directory, f"episodes_{self.chunk_num}.npz")
        batch.save_npz(file_name, compressed=False)
        self.chunk_num += 1
