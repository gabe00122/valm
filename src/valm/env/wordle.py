import random
import re
from collections import Counter
from functools import cache
from pathlib import Path

from valm.config import WordleEnvConfig
from valm.env.base import Game

WORDS_PATH = Path("env_assets/wordle_words.txt")
GUESS_RE = re.compile(r"\b[a-z]{5}\b", re.IGNORECASE)

# shaping reward for each position the first time it turns yellow-or-green,
# and again the first time it turns green
POSITION_REWARD = 0.025
SOLVED_REWARD = 0.75


@cache
def load_words() -> list[str]:
    return WORDS_PATH.read_text().split()


def score_guess(guess: str, secret: str) -> list[str]:
    """Wordle feedback per position: G (right spot), Y (wrong spot), X (absent)."""
    feedback = ["G" if g == s else "X" for g, s in zip(guess, secret)]
    unmatched = Counter(s for g, s in zip(guess, secret) if g != s)
    for i, g in enumerate(guess):
        if feedback[i] == "X" and unmatched[g] > 0:
            unmatched[g] -= 1
            feedback[i] = "Y"
    return feedback


class WordleGame(Game):
    MAX_TURNS = 6
    INSTRUCTIONS = "You are playing Wordle. Guess the secret 5-letter word. After each guess you will see a feedback row above your guess, with one symbol per letter position: G = correct letter in the correct position, Y = letter is in the word but in the wrong position, X = letter is not in the word. Think very briefly (2-3 sentences) then respond with your next guess. The last 5-letter word (a-z) that appears in your response is taken as your guess, so do not write other 5-letter words after it."

    def __init__(self, settings: WordleEnvConfig):
        self.settings = settings

    def reset(self, rng: random.Random) -> tuple[str, dict[str, float]]:
        self.secret = rng.choice(load_words())
        self.guesses = 0
        self.got_yellow = [False] * 5
        self.got_green = [False] * 5
        return "Make your first guess now", {"word_found": 0.0}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, float]]:
        guesses = GUESS_RE.findall(action)
        self.guesses += 1

        if guesses:
            guess = guesses[-1].upper()
            feedback = score_guess(guess, self.secret)
            reward = self._shaping_reward(feedback)
            word_found = guess == self.secret
            if word_found:
                reward += SOLVED_REWARD
            obs = (
                "Your feedback is:\n"
                + "".join(f"{f} " for f in feedback)
                + "\n"
                + "".join(f"{g} " for g in guess)
                + "\n"
            )
        else:
            obs, reward, word_found = "No guess was found", 0.0, False

        remaining = self.settings.max_guesses - self.guesses
        obs += f"\nYou have {remaining} {'guesses' if remaining > 1 else 'guess'} left"
        done = word_found or self.guesses >= self.settings.max_guesses
        return obs, reward, done, {"word_found": float(word_found)}

    def _shaping_reward(self, feedback: list[str]) -> float:
        reward = 0.0
        for i, f in enumerate(feedback):
            if f in "YG" and not self.got_yellow[i]:
                self.got_yellow[i] = True
                reward += POSITION_REWARD
            if f == "G" and not self.got_green[i]:
                self.got_green[i] = True
                reward += POSITION_REWARD
        return reward

