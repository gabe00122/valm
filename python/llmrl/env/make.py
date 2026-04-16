from types import SimpleNamespace
from llmrl.env.base import Env
from llmrl._envs import (
    ArithmeticEnv,
    WordleEnv,
)

ENV_DEFAULTS = {
    "arithmetic": {"max_x": 100, "max_y": 100},
    "wordle": {"max_guesses": 6},
    "countdown": {"num_numbers": 6, "num_operations": 3, "max_number": 100},
    "spatial": {"grid_size": 5, "max_steps": 10},
    "base_conversion": {"max_value": 1000},
    "date_arith": {"max_days": 365},
    "graph": {"num_nodes": 6, "num_edges": 8},
    "sudoku": {"grid_size": 4, "num_removed": 8},
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
