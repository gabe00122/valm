from typing import cast, Any
import distrax
import jax
from flax import nnx
from jax import numpy as jnp
from vaml.buffer import UpdateBatch
from vaml.config import LossConfig
from vaml.model.qwen3 import Qwen3


def summery_stats(values: jax.Array, where: jax.Array | None = None) -> dict[str, jax.Array]:
    return {
        "mean": jnp.mean(values, where=where),
        "std": jnp.std(values, where=where),
        "min": jnp.min(values, where=where),
        "max": jnp.max(values, where=where),
    }


def calculate_advantages(
    rewards: jax.Array, values: jax.Array, discount: jax.Array, td_lambda: jax.Array
) -> tuple[jax.Array, jax.Array]:
    def _body(acc, xs):
        rewards_t, value_tp1, discount_t, lambda_t = xs
        acc = rewards_t + discount_t * ((1 - lambda_t) * value_tp1 + lambda_t * acc)
        return acc, acc

    # swap to time major
    _, targets = jax.lax.scan(
        _body,
        jnp.zeros((rewards.shape[0],), dtype=jnp.float32),
        (
            jnp.swapaxes(rewards[:, 1:], 0, 1),
            jnp.swapaxes(values[:, 1:], 0, 1),
            jnp.swapaxes(discount, 0, 1),
            jnp.swapaxes(td_lambda, 0, 1),
        ),
        reverse=True,
    )
    targets = jnp.swapaxes(targets, 0, 1)
    advantages = targets - values[:, :-1]
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(targets)


def explained_variance(values: jax.Array, targets: jax.Array, bounds_mask: jax.Array) -> jax.Array:
    value_pred = values[:, :-1]
    value_mask = bounds_mask[:, :-1]

    target_var = jnp.var(targets, where=value_mask)
    return jnp.where(
        target_var > 0,
        1.0 - jnp.var(targets - value_pred, where=value_mask) / target_var,
        0.0,
    )

def loss_fn(
    model: Qwen3,
    rollout: UpdateBatch,
    td_discount: jax.Array,
    config: LossConfig,
    bounds_mask: jax.Array,
    value_only: bool,
    rng_key: jax.Array,
) -> tuple[jax.Array, tuple[dict[str, Any], dict[str, Any], jax.Array]]:
    batch_len, seq_len = rollout.context.shape

    policy_mask = jnp.asarray(rollout.policy_mask)[:, :-1]

    positions = jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32)[None, :], batch_len, 0)

    logits, value_repr, _, rng_key = model(
        jnp.asarray(rollout.context), positions, rng_key=rng_key
    )
    assert value_repr is not None

    values = value_repr.value()
    policy = distrax.Categorical(logits=logits[:, :-1])

    log_prob = policy.log_prob(rollout.context[:, 1:])

    log_ratio = log_prob - rollout.log_probs
    pg_ratio = jnp.exp(log_ratio)
    td_lambda = jnp.minimum(pg_ratio, config.gae_lambda)
    advantages, targets = calculate_advantages(
        jnp.asarray(rollout.rewards), values, td_discount, td_lambda
    )

    value_loss = value_repr[:, :-1].loss(targets)

    entropy = cast(jax.Array, policy.entropy())

    loss = 0.0
    if config.entropy_coef is not None:
        loss = loss + config.entropy_coef * -entropy.mean(where=policy_mask)

    # clip fraction
    # explained variance
    # gradient norm

    loss = loss + value_loss.mean(where=bounds_mask[:, :-1])

    # high level metrics are all well and good but we should return token aligned values like advantage and clip for the vizualizer
    summery_metrics = {
        "value_loss": summery_stats(value_loss, where=bounds_mask[:, :-1]),
        "value": summery_stats(values, where=bounds_mask),
        "entropy": jnp.mean(entropy, where=policy_mask),
        "approx_kl": (pg_ratio - 1 - log_ratio).mean(where=policy_mask),
        "td_lambda": td_lambda.mean(where=policy_mask),
        "advantage": summery_stats(advantages, where=policy_mask),
        "rewards": summery_stats(rollout.rewards.sum() / rollout.rewards.shape[0]),
        "explained_variance": explained_variance(values, targets, bounds_mask),
        "episode_length": rollout.context_length.mean(),
    }

    token_metrics = {
        "value_loss": value_loss,
        "value": values,
        "advantage": advantages
    }

    if not value_only:
        pg_loss1 = pg_ratio * advantages
        pg_loss2 = (
            jnp.clip(pg_ratio, 1.0 - config.pg_clip_low, 1.0 + config.pg_clip_high)
            * advantages
        )
        actor_loss = -jnp.minimum(pg_loss1, pg_loss2)

        clipped_tokens = (pg_ratio < 1.0 - config.pg_clip_low) | (pg_ratio > 1.0 + config.pg_clip_high)
        clip_fraction = jnp.mean(clipped_tokens, where=policy_mask)

        summery_metrics = {
            **summery_metrics,
            "actor_loss": summery_stats(actor_loss, where=policy_mask),
            "clip_fraction": clip_fraction
        }
        token_metrics = {
            **token_metrics,
            "clipped_tokens": clipped_tokens,
            "actor_loss": actor_loss
        }
        loss = loss + actor_loss.mean(where=policy_mask)

    return loss, (summery_metrics, token_metrics, rng_key)


@jax.jit(
    static_argnames=(
        "policy_opt_def",
        "value_opt_def",
        "model_def",
        "config",
        "value_only",
    ),
    donate_argnames=("policy_opt_state", "value_opt_state", "model_state"),
)
def update_step(
    policy_opt_def,
    policy_opt_state,
    value_opt_def,
    value_opt_state,
    model_def,
    model_state,
    rng_key: jax.Array,
    rollout: UpdateBatch,
    config: LossConfig,
    value_only: bool,
):
    if not value_only:
        policy_opt: nnx.Optimizer = nnx.merge(policy_opt_def, policy_opt_state)

    value_opt: nnx.Optimizer = nnx.merge(value_opt_def, value_opt_state)
    model: Qwen3 = nnx.merge(model_def, model_state)

    batch_len, seq_len = rollout.context.shape

    seq_range = jnp.arange(seq_len, dtype=jnp.int32)
    bounds_mask = seq_range[None, :] < rollout.context_length[:, None]

    # we have turn indecies now so this could be a scatter
    turn_boundries = ~rollout.policy_mask[:, :-1] & rollout.policy_mask[:, 1:]
    td_discount = jnp.where(turn_boundries, config.turn_discount, config.gae_discount)

    wrt = value_opt.wrt
    if not value_only:
        wrt = nnx.Any(policy_opt.wrt, wrt)

    diff = nnx.DiffState(0, wrt)
    grad, (summery_metrics, token_metrics, rng_key) = nnx.grad(loss_fn, argnums=diff, has_aux=True)(
        model, rollout, td_discount, config, bounds_mask, value_only, rng_key
    )

    if not value_only:
        policy_opt.update(model, grad)
    value_opt.update(model, grad)

    policy_opt_state = None if value_only else nnx.state(policy_opt)
    return policy_opt_state, nnx.state(value_opt), nnx.state(model), summery_metrics, token_metrics, rng_key
