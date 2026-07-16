"""Reference tests for the GRPO group-relative advantage.

`calculate_grpo_advantage` sums each episode's rewards over time, then
normalizes within each group of `group_size` *consecutive* episodes (the
layout GroupedEpisodeListener guarantees) to zero mean / unit std.
"""

import numpy as np
from valm.update_step.grpo import calculate_grpo_advantage


def _episode_rewards(totals, seq=5, rng=None):
    """Spread each episode's total reward over a few random time steps."""
    totals = np.asarray(totals, dtype=np.float32)
    rewards = np.zeros((totals.shape[0], seq), dtype=np.float32)
    if rng is None:
        rewards[:, -1] = totals
    else:
        split = rng.uniform(0.0, 1.0, size=totals.shape[0]).astype(np.float32)
        rewards[:, 0] = totals * split
        rewards[:, -1] = totals * (1.0 - split)
    return rewards


def test_hand_computed_example():
    # group 1: totals [1, 3] -> mean 2, std 1 -> advantages [-1, 1]
    # group 2: totals [5, 5] -> zero std, epsilon keeps it finite -> [0, 0]
    rewards = _episode_rewards([1.0, 3.0, 5.0, 5.0])

    advantages = calculate_grpo_advantage(rewards, group_size=2)

    np.testing.assert_allclose(advantages, [-1.0, 1.0, 0.0, 0.0], atol=1e-6)


def test_rewards_are_summed_over_time_before_normalizing():
    rng = np.random.default_rng(0)
    totals = [0.0, 2.0, -1.0, 3.0, 4.0, 4.5]

    sparse = calculate_grpo_advantage(_episode_rewards(totals), group_size=3)
    spread = calculate_grpo_advantage(_episode_rewards(totals, rng=rng), group_size=3)

    np.testing.assert_allclose(sparse, spread, rtol=1e-5, atol=1e-6)


def test_groups_are_consecutive_episodes_with_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    group_size = 4
    rewards = rng.normal(size=(12, 6)).astype(np.float32)

    advantages = calculate_grpo_advantage(rewards, group_size)

    grouped = advantages.reshape(-1, group_size)
    np.testing.assert_allclose(grouped.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(grouped.std(axis=1), 1.0, atol=1e-4)


def test_groups_are_normalized_independently():
    """Changing one group's rewards must not move another group's advantages."""
    rng = np.random.default_rng(2)
    rewards = rng.normal(size=(6, 4)).astype(np.float32)

    base = calculate_grpo_advantage(rewards, group_size=3)

    perturbed_rewards = rewards.copy()
    perturbed_rewards[3:] += 100.0
    perturbed = calculate_grpo_advantage(perturbed_rewards, group_size=3)

    np.testing.assert_allclose(base[:3], perturbed[:3], rtol=1e-5, atol=1e-6)


def test_constant_group_yields_zero_not_nan():
    rewards = _episode_rewards([2.0, 2.0, 2.0])

    advantages = calculate_grpo_advantage(rewards, group_size=3)

    assert np.all(np.isfinite(advantages))
    np.testing.assert_allclose(advantages, 0.0, atol=1e-6)


def test_group_size_one_degenerates_to_zero_signal():
    """With singleton groups every advantage is (r - r) / eps == 0 — the
    'grpo-cold' collapse mode; pinned so the behavior is explicit."""
    rewards = _episode_rewards([1.0, -2.0, 7.0])

    advantages = calculate_grpo_advantage(rewards, group_size=1)

    np.testing.assert_allclose(advantages, 0.0, atol=1e-6)
