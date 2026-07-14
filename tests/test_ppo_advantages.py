"""Reference tests for the PPO GAE / TD(lambda) target computation.

`calculate_advantages` implements the recursion

    target_t = r_{t+1} + d_t * ((1 - l_t) * v_{t+1} + l_t * target_{t+1})

via a reversed `jax.lax.scan` with a zero bootstrap beyond the horizon.
These tests pin that math against a naive per-episode Python loop, the
closed-form special cases (Monte Carlo, one-step TD), and the independent
Rust implementation in `vaml._envs.lambda_returns`.
"""

import numpy as np
from jax import numpy as jnp
from vaml._envs import lambda_returns
from vaml.update_step.ppo import calculate_advantages, explained_variance


def _reference_targets(rewards, values, discount, td_lambda):
    """Naive TD(lambda) recursion, one episode and one timestep at a time."""
    batch, seq = rewards.shape
    targets = np.zeros((batch, seq - 1), dtype=np.float64)
    for b in range(batch):
        acc = 0.0
        for t in reversed(range(seq - 1)):
            acc = rewards[b, t + 1] + discount[b, t] * (
                (1.0 - td_lambda[b, t]) * values[b, t + 1] + td_lambda[b, t] * acc
            )
            targets[b, t] = acc
    return targets


def _random_inputs(rng, batch=3, seq=9):
    rewards = rng.normal(size=(batch, seq)).astype(np.float32)
    values = rng.normal(size=(batch, seq)).astype(np.float32)
    discount = rng.uniform(0.0, 1.0, size=(batch, seq - 1)).astype(np.float32)
    td_lambda = rng.uniform(0.0, 1.0, size=(batch, seq - 1)).astype(np.float32)
    return rewards, values, discount, td_lambda


def test_targets_match_naive_reference():
    rewards, values, discount, td_lambda = _random_inputs(np.random.default_rng(0))

    advantages, targets = calculate_advantages(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.asarray(discount),
        jnp.asarray(td_lambda),
    )

    expected = _reference_targets(rewards, values, discount, td_lambda)
    np.testing.assert_allclose(np.asarray(targets), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(advantages), expected - values[:, :-1], rtol=1e-5, atol=1e-5
    )


def test_lambda_one_discount_one_is_monte_carlo_return():
    """With l=1, d=1 the target is the plain sum of future rewards."""
    rewards, values, _, _ = _random_inputs(np.random.default_rng(1))
    ones = np.ones((rewards.shape[0], rewards.shape[1] - 1), dtype=np.float32)

    _, targets = calculate_advantages(
        jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(ones), jnp.asarray(ones)
    )

    # target_t = r_{t+1} + r_{t+2} + ... (values never enter with lambda=1)
    expected = np.cumsum(rewards[:, 1:][:, ::-1], axis=1)[:, ::-1]
    np.testing.assert_allclose(np.asarray(targets), expected, rtol=1e-5, atol=1e-5)


def test_lambda_zero_is_one_step_td_target():
    rewards, values, discount, _ = _random_inputs(np.random.default_rng(2))
    zeros = np.zeros_like(discount)

    _, targets = calculate_advantages(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.asarray(discount),
        jnp.asarray(zeros),
    )

    expected = rewards[:, 1:] + discount * values[:, 1:]
    np.testing.assert_allclose(np.asarray(targets), expected, rtol=1e-5, atol=1e-5)


def test_zero_discount_cuts_bootstrapping_at_boundary():
    """A zero discount (as set beyond `context_length` in `update_step`) makes the
    target at that step the immediate reward, regardless of anything later."""
    rewards, values, discount, td_lambda = _random_inputs(np.random.default_rng(3))
    boundary = 4
    discount[:, boundary] = 0.0

    _, targets = calculate_advantages(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.asarray(discount),
        jnp.asarray(td_lambda),
    )

    np.testing.assert_allclose(
        np.asarray(targets)[:, boundary], rewards[:, boundary + 1], rtol=1e-5
    )

    # nothing after the boundary leaks into targets before it
    rewards2 = rewards.copy()
    values2 = values.copy()
    rewards2[:, boundary + 2 :] += 100.0
    values2[:, boundary + 2 :] += 100.0
    _, targets2 = calculate_advantages(
        jnp.asarray(rewards2),
        jnp.asarray(values2),
        jnp.asarray(discount),
        jnp.asarray(td_lambda),
    )
    np.testing.assert_allclose(
        np.asarray(targets2)[:, : boundary + 1],
        np.asarray(targets)[:, : boundary + 1],
        rtol=1e-5,
        atol=1e-5,
    )


def test_matches_rust_lambda_returns():
    """Cross-check the JAX scan against the independent Rust implementation.

    The Rust version bootstraps from the last value while the JAX version
    bootstraps from zero, so zero out the final value to align them.
    """
    rewards, values, _, _ = _random_inputs(np.random.default_rng(4))
    values[:, -1] = 0.0
    discount, lam = 0.97, 0.9
    batch, seq = rewards.shape

    _, targets = calculate_advantages(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.full((batch, seq - 1), discount, dtype=jnp.float32),
        jnp.full((batch, seq - 1), lam, dtype=jnp.float32),
    )

    rust_targets = np.zeros((batch, seq - 1), dtype=np.float32)
    lambda_returns(
        np.ascontiguousarray(rewards[:, 1:]),
        np.ascontiguousarray(values[:, 1:]),
        discount,
        lam,
        rust_targets,
    )

    np.testing.assert_allclose(np.asarray(targets), rust_targets, rtol=1e-5, atol=1e-5)


def test_explained_variance_perfect_prediction_is_one():
    rng = np.random.default_rng(5)
    targets = rng.normal(size=(3, 7)).astype(np.float32)
    values = np.concatenate([targets, rng.normal(size=(3, 1)).astype(np.float32)], 1)
    mask = np.ones((3, 8), dtype=bool)

    ev = explained_variance(jnp.asarray(values), jnp.asarray(targets), jnp.asarray(mask))

    np.testing.assert_allclose(np.asarray(ev), 1.0, atol=1e-5)


def test_explained_variance_mean_prediction_is_zero():
    rng = np.random.default_rng(6)
    targets = rng.normal(size=(3, 7)).astype(np.float32)
    values = np.full((3, 8), targets.mean(), dtype=np.float32)
    mask = np.ones((3, 8), dtype=bool)

    ev = explained_variance(jnp.asarray(values), jnp.asarray(targets), jnp.asarray(mask))

    np.testing.assert_allclose(np.asarray(ev), 0.0, atol=1e-5)


def test_explained_variance_constant_targets_returns_zero():
    """Zero target variance hits the guard branch instead of dividing by zero."""
    targets = np.full((2, 5), 3.0, dtype=np.float32)
    values = np.zeros((2, 6), dtype=np.float32)
    mask = np.ones((2, 6), dtype=bool)

    ev = explained_variance(jnp.asarray(values), jnp.asarray(targets), jnp.asarray(mask))

    assert np.asarray(ev) == 0.0


def test_explained_variance_ignores_masked_positions():
    rng = np.random.default_rng(7)
    targets = rng.normal(size=(2, 7)).astype(np.float32)
    values = rng.normal(size=(2, 8)).astype(np.float32)
    mask = np.ones((2, 8), dtype=bool)
    mask[:, 5:] = False

    clean = explained_variance(
        jnp.asarray(values), jnp.asarray(targets), jnp.asarray(mask)
    )

    junk_values = values.copy()
    junk_targets = targets.copy()
    junk_values[:, 5:] = 1e6
    junk_targets[:, 5:] = -1e6
    junk = explained_variance(
        jnp.asarray(junk_values), jnp.asarray(junk_targets), jnp.asarray(mask)
    )

    np.testing.assert_allclose(np.asarray(clean), np.asarray(junk), rtol=1e-5)
