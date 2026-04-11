from types import SimpleNamespace

from llmrl._envs import (
    ArithmeticEnv,
    WordleEnv,
)
from llmrl.env.base import Env

ENV_DEFAULTS = {
    "arithmetic": {"max_x": 100, "max_y": 100},
    "wordle": {"max_guesses": 6},
}


def make_env(env_name: str, num_agents: int, seed: int, settings) -> Env:
    if settings is None:
        settings = SimpleNamespace(**ENV_DEFAULTS[env_name])

    if env_name == "arithmetic":
        return ArithmeticEnv(num_agents, seed, settings)
    elif env_name == "wordle":
        return WordleEnv(num_agents, seed, settings)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
