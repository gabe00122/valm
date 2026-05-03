from typing import Any

import numpy as np
import numpy.typing as npt

class ArithmeticEnv:
    def __init__(self, num_agents: int, seed: int, settings: Any) -> None: ...
    @property
    def max_turns(self) -> int: ...
    def reset(
        self, batch_indices: npt.NDArray[np.int32]
    ) -> tuple[list[str], dict[str, npt.NDArray[np.float32]]]: ...
    def step(
        self, batch_indices: npt.NDArray[np.int32], actions: list[str]
    ) -> tuple[
        list[str],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        dict[str, npt.NDArray[np.float32]],
    ]: ...
    def instructions(self) -> str: ...

class WordleEnv:
    def __init__(self, num_agents: int, seed: int, settings: Any) -> None: ...
    @property
    def max_turns(self) -> int: ...
    def reset(
        self, batch_indices: npt.NDArray[np.int32]
    ) -> tuple[list[str], dict[str, npt.NDArray[np.float32]]]: ...
    def step(
        self, batch_indices: npt.NDArray[np.int32], actions: list[str]
    ) -> tuple[
        list[str],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        dict[str, npt.NDArray[np.float32]],
    ]: ...
    def instructions(self) -> str: ...

def lambda_returns(
    rewards: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    discount: float,
    lambda_: float,
    targets: npt.NDArray[np.float32],
) -> None: ...
