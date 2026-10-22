"""Contract tests for the text environments.

The training loop (train_rl.py) depends on this exact interface: reset/step
shapes and dtypes, GRPO group ids that pair consecutive resets onto the same
problem, and step() reporting the id of the episode that just *finished*.
"""

import random
import re

import numpy as np
import pytest
from valm.config import ArithmeticEnvConfig, WordleEnvConfig
from valm.env.arithmetic import parse_answer
from valm.env.make import make_env
from valm.env.wordle import SOLVED_REWARD, WordleGame, score_guess

_PROBLEM_RE = re.compile(r"^(-?\d+(?:\.\d+)?) ([+\-*/]) (-?\d+(?:\.\d+)?) = ")


def _solve(obs: str) -> np.float32:
    """Compute the expected answer from the prompt."""
    match = _PROBLEM_RE.match(obs)
    assert match is not None, f"unparseable arithmetic prompt: {obs!r}"
    x = np.float32(match.group(1))
    y = np.float32(match.group(3))
    op = match.group(2)
    if op == "+":
        return x + y
    if op == "-":
        return x - y
    if op == "*":
        return x * y
    return x / y


def _answer(value: np.float32) -> str:
    return np.format_float_positional(value)


def _arithmetic_env(num_agents=4, group_size=2, seed=123):
    return make_env(
        ArithmeticEnvConfig(max_x=20, max_y=20), num_agents, group_size, seed
    )


def test_reset_shapes_and_group_pairing():
    env = _arithmetic_env()
    indices = np.arange(4, dtype=np.int32)

    obs, group_ids, metrics = env.reset(indices)

    assert len(obs) == 4
    assert all(isinstance(o, str) for o in obs)
    assert group_ids.shape == (4,) and group_ids.dtype == np.uint64
    assert isinstance(metrics, dict)

    # consecutive resets pair into groups of group_size sharing one id ...
    assert group_ids[0] == group_ids[1]
    assert group_ids[2] == group_ids[3]
    assert group_ids[0] != group_ids[2]
    # ... and members of a group get the same problem
    assert obs[0] == obs[1]
    assert obs[2] == obs[3]


def test_same_seed_reproduces_the_same_problems():
    indices = np.arange(4, dtype=np.int32)

    obs_a, ids_a, _ = _arithmetic_env(seed=7).reset(indices)
    obs_b, ids_b, _ = _arithmetic_env(seed=7).reset(indices)

    assert obs_a == obs_b
    np.testing.assert_array_equal(ids_a, ids_b)


def test_correct_answer_earns_reward_and_finishes_episode():
    env = _arithmetic_env()
    indices = np.arange(4, dtype=np.int32)
    obs, reset_ids, _ = env.reset(indices)

    answers = [_answer(_solve(o)) for o in obs]
    next_obs, rewards, dones, step_ids, metrics = env.step(indices, answers)

    assert rewards.shape == (4,) and rewards.dtype == np.float32
    assert dones.shape == (4,) and dones.dtype == np.bool_
    for i, o in enumerate(obs):
        expected = 1.0 if np.isfinite(_solve(o)) else 0.0  # division by zero prompt
        assert rewards[i] == expected, f"prompt {o!r} answered {answers[i]!r}"

    # arithmetic is single-turn: every episode ends and the reported group id
    # is the id of the episode that just finished, not the freshly reset one
    assert dones.all()
    np.testing.assert_array_equal(step_ids, reset_ids)
    assert len(next_obs) == 4


def test_wrong_answer_earns_no_reward():
    env = _arithmetic_env()
    indices = np.arange(4, dtype=np.int32)
    obs, _, _ = env.reset(indices)

    answers = [_answer(_solve(o) + np.float32(1.0)) for o in obs]
    _, rewards, dones, _, _ = env.step(indices, answers)

    np.testing.assert_array_equal(rewards, 0.0)
    assert dones.all()


def test_answer_embedded_in_chatter_still_parses():
    """The env takes the *last* number in the response."""
    env = _arithmetic_env()
    indices = np.arange(4, dtype=np.int32)
    obs, _, _ = env.reset(indices)

    answers = [
        f"Let me think: 5 + 5 is not it.\nThe answer is {_answer(_solve(o))}"
        for o in obs
    ]
    _, rewards, _, _, _ = env.step(indices, answers)

    for o, r in zip(obs, rewards):
        assert r == (1.0 if np.isfinite(_solve(o)) else 0.0)


def test_wordle_smoke():
    env = make_env(WordleEnvConfig(max_guesses=6), 2, 1, 42)
    assert env.max_turns == 6
    assert "Wordle" in env.instructions()

    indices = np.arange(2, dtype=np.int32)
    obs, group_ids, _ = env.reset(indices)
    assert len(obs) == 2
    assert group_ids[0] != group_ids[1]  # group_size 1 -> every episode its own group

    obs, rewards, dones, _, _ = env.step(indices, ["my guess is crane", "slate"])
    assert len(obs) == 2
    assert rewards.shape == (2,)
    # feedback marks each position G/Y/X unless the word was solved outright
    for o, done in zip(obs, dones):
        assert done or "feedback" in o

    # after max_guesses total guesses every episode must have terminated
    finished = np.array(dones)
    for _ in range(5):
        _, _, dones, _, _ = env.step(indices, ["crane", "slate"])
        finished |= dones
    assert finished.all()


def test_parse_answer_takes_last_number():
    assert parse_answer("2 + 2 is 4, so\n1,234.5") == 1234.5
    assert parse_answer("-7.") == -7.0
    assert parse_answer("no numbers here") is None


def test_score_guess_handles_repeated_letters():
    # only one E is unmatched in the secret, so only the first spare E is yellow
    assert score_guess("EERIE", "THEME") == list("YXXXG")
    assert score_guess("CRANE", "CRANE") == list("GGGGG")
    assert score_guess("SLATE", "CRONY") == list("XXXXX")


def _wordle_game(secret: str, max_guesses: int) -> WordleGame:
    game = WordleGame(WordleEnvConfig(max_guesses=max_guesses))
    game.reset(random.Random(0))
    game.secret = secret
    return game


def test_wordle_rewards_and_feedback_text():
    game = _wordle_game("CRANE", max_guesses=6)

    obs, reward, done, metrics = game.step("I'll try trace")
    assert obs == "Your feedback is:\nX G G Y G \nT R A C E \n\nYou have 5 guesses left"
    assert reward == pytest.approx(0.025 * 4 + 0.025 * 3)  # 4 hits, 3 of them green
    assert not done and metrics == {"word_found": 0.0}

    # positions already rewarded earn nothing new
    _, reward, _, _ = game.step("trace")
    assert reward == 0.0

    _, reward, done, metrics = game.step("crane")
    assert reward == pytest.approx(0.025 * 3 + SOLVED_REWARD)  # C and N turn green
    assert done and metrics == {"word_found": 1.0}


def test_wordle_missing_guess_still_uses_a_turn():
    game = _wordle_game("CRANE", max_guesses=2)

    obs, reward, done, _ = game.step("hmm")
    assert obs == "No guess was found\nYou have 1 guess left"
    assert reward == 0.0 and not done

    _, _, done, _ = game.step("hmm")
    assert done
