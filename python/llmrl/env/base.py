from typing import Protocol

import numpy as np


class Env(Protocol):
    def reset(self, batch_indices: np.ndarray) -> list[str]: ...
    def step(
        self, batch_indices: np.ndarray, actions: list[str]
    ) -> tuple[list[str], np.ndarray, np.ndarray]: ...
    def instructions(self) -> str: ...
