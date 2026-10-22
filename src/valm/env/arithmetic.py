import operator
import random
import re

from valm.config import ArithmeticEnvConfig
from valm.env.base import Game

OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d*)?")


def parse_answer(text: str) -> float | None:
    """The last number in the response, ignoring thousands separators."""
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


class ArithmeticGame(Game):
    MAX_TURNS = 1
    INSTRUCTIONS = "Solve the arithmetic expression using +, -, * or /. Show your work if needed, but end with only the numeric result on its own line. Always output with decimals such as 123.456"

    def __init__(self, settings: ArithmeticEnvConfig):
        self.settings = settings

    def reset(self, rng: random.Random) -> tuple[str, dict[str, float]]:
        x = rng.randint(0, self.settings.max_x)
        y = rng.randint(0, self.settings.max_y)
        op = rng.choice(list(OPERATORS))
        self.result = OPERATORS[op](x, y)
        return f"{x} {op} {y} = ...", {}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, float]]:
        answer = parse_answer(action)
        correct = answer is not None and abs(answer - self.result) < 0.001
        return "", 1.0 if correct else 0.0, True, {}

