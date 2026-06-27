from typing import Protocol

import numpy as np


class Env(Protocol):
    max_turns: int

    def reset(
        self, batch_indices: np.ndarray
    ) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]: ...
    def step(
        self, batch_indices: np.ndarray, actions: list[str]
    ) -> tuple[
        list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]
    ]: ...
    def instructions(self) -> str: ...
