from valm.config import EnvConfig
from valm.env.arithmetic import ArithmeticGame
from valm.env.base import Env, TextEnv
from valm.env.wordle import WordleGame


def make_env(settings: EnvConfig, num_agents: int, group_size: int, seed: int) -> Env:
    match settings.name:
        case "arithmetic":
            game_cls = ArithmeticGame
            new_game = lambda: ArithmeticGame(settings)
        case "wordle":
            game_cls = WordleGame
            new_game = lambda: WordleGame(settings)

    return TextEnv(
        new_game, game_cls.MAX_TURNS, game_cls.INSTRUCTIONS, num_agents, group_size, seed
    )
