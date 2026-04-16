from typing import Protocol

import numpy as np


class Agent(Protocol):
    def reset(self) -> None: ...
    def act(
        self,
        batch_indices: np.ndarray,
        obs: list[str],
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]: ...
    def close(self) -> None:
        pass
