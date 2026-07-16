from typing import Any, cast

import distrax
import jax
import numpy as np
from einops import rearrange
from flax import nnx
from jax import numpy as jnp
from valm.buffer import UpdateBatch, array_chunks
from valm.config import GRPOLossConfig
from valm.model.qwen3 import Qwen3
from valm.update_step.util import summery_stats


def calculate_grpo_advantage(
    rewards: np.ndarray, group_size: int, epsilon: float = 1e-8
) -> np.ndarray:
    rewards = np.sum(rewards, axis=-1)
    rewards = rearrange(rewards, "(b g) -> b g", g=group_size)

    advantages = (rewards - np.mean(rewards, -1, keepdims=True)) / (
        np.std(rewards, -1, keepdims=True) + epsilon
    )

    return rearrange(advantages, "b g -> (b g)")


def loss_fn(
    model: Qwen3,
    rollout: UpdateBatch,
    advantages: jax.Array,
    config: GRPOLossConfig,
    bounds_mask: jax.Array,
    rng_key: jax.Array,
) -> tuple[jax.Array, tuple[dict[str, Any], dict[str, Any], jax.Array]]:
    batch_len, seq_len = rollout.context.shape

    policy_mask = jnp.asarray(rollout.policy_mask)[:, :-1] & bounds_mask[:, :-1]

    positions = jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32)[None, :], batch_len, 0)

    # GRPO has no critic: ignore the value representation the model may return.
    logits, _, _, rng_key = model(
        jnp.asarray(rollout.context),
        positions,
        rewards=jnp.asarray(rollout.rewards, dtype=jnp.bfloat16),
        rng_key=rng_key,
    )

    policy = distrax.Categorical(logits=logits[:, :-1])

    log_prob = policy.log_prob(rollout.context[:, 1:])

    log_ratio = log_prob - rollout.log_probs
    pg_ratio = jnp.exp(log_ratio)
    pg_ratio = jnp.where(policy_mask, pg_ratio, 1.0)

    # One group-relative advantage per episode, shared by every token in that
    # episode's response. (batch_len,) -> (batch_len, 1) to broadcast over time.
    group_advantage = jnp.asarray(advantages)
    advantage = group_advantage[:, None]

    entropy = cast(jax.Array, policy.entropy())

    pg_loss1 = pg_ratio * advantage
    pg_loss2 = (
        jnp.clip(pg_ratio, 1.0 - config.pg_clip_low, 1.0 + config.pg_clip_high)
        * advantage
    )
    actor_loss = -jnp.minimum(pg_loss1, pg_loss2)

    clipped_tokens = (pg_ratio < 1.0 - config.pg_clip_low) | (
        pg_ratio > 1.0 + config.pg_clip_high
    )
    clip_fraction = jnp.mean(clipped_tokens, where=policy_mask)

    loss = actor_loss.mean(where=policy_mask)
    if config.entropy_coef is not None:
        loss = loss + config.entropy_coef * -entropy.mean(where=policy_mask)

    # high level metrics are all well and good but we should return token aligned values like advantage and clip for the vizualizer
    summery_metrics = {
        "actor_loss": summery_stats(actor_loss, where=policy_mask),
        "clip_fraction": clip_fraction,
        "entropy": jnp.mean(entropy, where=policy_mask),
        "approx_kl": (pg_ratio - 1 - log_ratio).mean(where=policy_mask),
        "advantage": summery_stats(group_advantage),
        "rewards": summery_stats(rollout.rewards.sum(axis=-1)),
        "episode_length": rollout.context_length.mean(),
    }

    token_metrics = {
        "advantage": jnp.broadcast_to(advantage, policy_mask.shape),
        "clipped_tokens": clipped_tokens,
        "actor_loss": actor_loss,
    }

    return loss, (summery_metrics, token_metrics, rng_key)


def multi_grpo_update_bucketed(
    policy_opt_def,
    policy_opt_state,
    model_def,
    model_state,
    rng_key: jax.Array,
    rollout: UpdateBatch,
    config: GRPOLossConfig,
    steps: int,
    group_size: int,
):
    # Advantages normalize across each whole group, so compute them on the full
    # batch before chunking for gradient accumulation.
    advantages = calculate_grpo_advantage(rollout.rewards, group_size)

    episodes = rollout.context.shape[0]
    seq_length = rollout.context.shape[1]
    chunk_size = episodes // steps
    summary_chunks = []
    token_chunks = []

    # Slice advantages on the same episode boundaries array_chunks uses.
    advantage_chunks = (
        advantages[start : start + chunk_size]
        for start in range(0, episodes, chunk_size)
    )

    for rollout_chunk, advantage_chunk in zip(
        array_chunks(rollout, chunk_size), advantage_chunks
    ):
        (
            policy_opt_state,
            model_state,
            summery_metrics,
            token_metrics,
            rng_key,
        ) = grpo_update_jit(
            policy_opt_def,
            policy_opt_state,
            model_def,
            model_state,
            rng_key,
            rollout_chunk,
            advantage_chunk,
            config,
        )

        summary_chunks.append(summery_metrics)
        token_chunks.append(token_metrics)

    summary_chunks, token_chunks = jax.device_get((summary_chunks, token_chunks))

    summery_metrics = jax.tree.map(
        lambda *xs: np.mean(np.stack(xs, axis=0), axis=0), *summary_chunks
    )
    token_metrics = jax.tree.map(
        lambda *xs: np.concatenate(
            [
                np.pad(x, pad_width=((0, 0), (0, seq_length + 1 - x.shape[1])))
                for x in xs
            ],
            axis=0,
        ),
        *token_chunks,
    )

    return (
        policy_opt_state,
        model_state,
        summery_metrics,
        token_metrics,
        rng_key,
    )


@jax.jit(
    static_argnames=(
        "policy_opt_def",
        "model_def",
        "config",
    ),
    donate_argnames=("policy_opt_state", "model_state"),
)
def grpo_update_jit(
    policy_opt_def,
    policy_opt_state,
    model_def,
    model_state,
    rng_key: jax.Array,
    rollout: UpdateBatch,
    advantages: jax.Array,
    config: GRPOLossConfig,
):
    policy_opt: nnx.Optimizer = nnx.merge(policy_opt_def, policy_opt_state)
    model: Qwen3 = nnx.merge(model_def, model_state)

    batch_len, seq_len = rollout.context.shape

    seq_range = jnp.arange(seq_len, dtype=jnp.int32)
    bounds_mask = seq_range[None, :] < rollout.context_length[:, None]

    diff = nnx.DiffState(0, policy_opt.wrt)
    grad, (summery_metrics, token_metrics, rng_key) = nnx.grad(
        loss_fn, argnums=diff, has_aux=True
    )(model, rollout, advantages, config, bounds_mask, rng_key)

    policy_opt.update(model, grad)

    return (
        nnx.state(policy_opt),
        nnx.state(model),
        summery_metrics,
        token_metrics,
        rng_key,
    )
