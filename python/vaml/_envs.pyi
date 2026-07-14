from typing import Any

import numpy as np

type _Int32Array1 = np.ndarray[tuple[int], np.dtype[np.int32]]
type _UInt64Array1 = np.ndarray[tuple[int], np.dtype[np.uint64]]
type _Float32Array1 = np.ndarray[tuple[int], np.dtype[np.float32]]
type _BoolArray1 = np.ndarray[tuple[int], np.dtype[np.bool_]]
type _Float32Array2 = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type _Metrics = dict[str, _Float32Array1]

class ArithmeticEnv:
    def __init__(
        self, num_agents: int, group_size: int, seed: int, settings: Any
    ) -> None: ...
    @property
    def max_turns(self) -> int: ...
    def reset(
        self, batch_indices: _Int32Array1
    ) -> tuple[
        list[str],
        _UInt64Array1,
        _Metrics,
    ]: ...
    def step(
        self, batch_indices: _Int32Array1, actions: list[str]
    ) -> tuple[
        list[str],
        _Float32Array1,
        _BoolArray1,
        _UInt64Array1,
        _Metrics,
    ]: ...
    def instructions(self) -> str: ...

class WordleEnv:
    def __init__(
        self, num_agents: int, group_size: int, seed: int, settings: Any
    ) -> None: ...
    @property
    def max_turns(self) -> int: ...
    def reset(
        self, batch_indices: _Int32Array1
    ) -> tuple[
        list[str],
        _UInt64Array1,
        _Metrics,
    ]: ...
    def step(
        self, batch_indices: _Int32Array1, actions: list[str]
    ) -> tuple[
        list[str],
        _Float32Array1,
        _BoolArray1,
        _UInt64Array1,
        _Metrics,
    ]: ...
    def instructions(self) -> str: ...
