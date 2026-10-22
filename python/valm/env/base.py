import random
from abc import ABC, abstractmethod
from collections.abc import Callable
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


class Game(ABC):
    """One agent's text game, reset for each new episode."""

    MAX_TURNS: int
    INSTRUCTIONS: str

    @abstractmethod
    def reset(self, rng: random.Random) -> tuple[str, dict[str, float]]:
        """Start a problem sampled from `rng`, returning the first obs and metrics.

        Members of a GRPO group get the same problem by sharing the rng seed.
        """

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool, dict[str, float]]:
        """Apply the agent's response, returning (obs, reward, done, metrics)."""


class GroupSequence:
    """Hands out problem ids, the same id to `group_size` consecutive episodes.

    Each new game is seeded from its id, so the members of a GRPO group all
    get the same problem.
    """

    def __init__(self, seed: int, group_size: int):
        self._rng = random.Random(seed)
        self._group_size = max(group_size, 1)
        self._remaining = 0
        self._group_id = 0

    def take_group_id(self) -> int:
        if self._remaining == 0:
            self._group_id = self._rng.getrandbits(64)
            self._remaining = self._group_size
        self._remaining -= 1
        return self._group_id


class TextEnv:
    """A batch of independent games addressed by agent index.

    Finished games are reset automatically: the observation returned by a
    `done` step is the first observation of the next game.
    """

    def __init__(
        self,
        new_game: Callable[[], Game],
        max_turns: int,
        instructions: str,
        num_agents: int,
        group_size: int,
        seed: int,
    ):
        self.max_turns = max_turns
        self._instructions = instructions
        self._groups = GroupSequence(seed, group_size)
        self._games = [new_game() for _ in range(num_agents)]
        self._group_ids = np.zeros(num_agents, dtype=np.uint64)

    def instructions(self) -> str:
        return self._instructions

    def reset(
        self, batch_indices: np.ndarray
    ) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
        obs, metrics = [], []
        for i in batch_indices:
            o, m = self._reset_one(i)
            obs.append(o)
            metrics.append(m)

        return obs, self._group_ids[batch_indices], _stack_metrics(metrics)

    def step(
        self, batch_indices: np.ndarray, actions: list[str]
    ) -> tuple[
        list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]
    ]:
        # capture the ids before stepping: a done step resets the game, and the
        # caller needs the id of the episode that just finished
        group_ids = self._group_ids[batch_indices]

        obs, rewards, dones, metrics = [], [], [], []
        for i, action in zip(batch_indices, actions):
            o, r, d, m = self._games[i].step(action)
            if d:
                o, _ = self._reset_one(i)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            metrics.append(m)

        return (
            obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            group_ids,
            _stack_metrics(metrics),
        )

    def _reset_one(self, i: int) -> tuple[str, dict[str, float]]:
        group_id = self._groups.take_group_id()
        self._group_ids[i] = group_id
        return self._games[i].reset(random.Random(group_id))


def _stack_metrics(metrics: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Turn per-agent metric dicts into one array per key (0 where missing)."""
    keys = {key for m in metrics for key in m}
    return {
        key: np.array([m.get(key, 0.0) for m in metrics], dtype=np.float32)
        for key in keys
    }
