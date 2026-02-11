import distrax
import jax
from flax import nnx
from jax import numpy as jnp
from llmrl.buffer import UpdateBatch
from llmrl.config import LossConfig
from llmrl.model.qwen3 import Qwen3


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


def loss_fn(
    model: Qwen3,
    rollout: UpdateBatch,
    td_discount: jax.Array,
    config: LossConfig,
    bounds_mask: jax.Array,
    value_only: bool,
    rng_key: jax.Array,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], jax.Array]]:
    batch_len, seq_len = rollout.context.shape

    policy_mask = rollout.policy_mask

    positions = jnp.repeat(jnp.arange(seq_len, dtype=jnp.int32)[None, :], batch_len, 0)

    logits, values_logits, _, rng_key = model(jnp.asarray(rollout.context), positions, rng_key=rng_key)
    values = model.get_value(values_logits)
    policy = distrax.Categorical(logits=logits[:, :-1])

    log_prob = policy.log_prob(rollout.context[:, 1:])

    log_ratio = log_prob - rollout.log_probs
    pg_ratio = jnp.exp(log_ratio)
    td_lambda = config.gae_lambda * jnp.minimum(pg_ratio, 1.0)
    advantages, targets = calculate_advantages(jnp.asarray(rollout.rewards), values, td_discount, td_lambda)

    value_loss = model.get_value_loss(values_logits[:, :-1], targets).mean(
        where=bounds_mask[:, :-1]
    )
    entropy = policy.entropy().mean(where=policy_mask[:, :-1])
    # entropy_loss = 0.0001 * -entropy

    loss = value_loss
    metrics = {
        "value_loss": value_loss,
        "value": values.mean(where=bounds_mask),
        "entropy": policy.entropy().mean(where=policy_mask[:, :-1]),
        "approx_kl": (pg_ratio - 1 - log_ratio).mean(where=policy_mask[:, :-1]),
        "td_lambda": td_lambda.mean(where=policy_mask[:, :-1]),
    }

    if not value_only:
        # log_prob = policy.log_prob(rollout.context[:, 1:])
        actor_loss: jax.Array = -(log_prob * advantages).mean(where=policy_mask[:, :-1])
        # pg_ratio = jnp.exp(log_prob - rollout.log_probs)
        # pg_loss1 = pg_ratio * advantages
        # pg_loss2 = (
        #     jnp.clip(pg_ratio, 1.0 - config.pg_clip_low, 1.0 + config.pg_clip_high)
        #     * advantages
        # )
        # actor_loss = -jnp.minimum(pg_loss1, pg_loss2).mean(where=policy_mask[:, :-1])

        metrics = {**metrics, "actor_loss": actor_loss}
        loss = loss + actor_loss # + entropy_loss
    else:
        _, true_targets = calculate_advantages(
            jnp.asarray(rollout.rewards), values, td_discount, jnp.ones_like(td_lambda)
        )
        value_error = jnp.mean(jnp.abs(values[:, :-1] - true_targets), where=bounds_mask[:, :-1])
        metrics = {**metrics, "value_error": value_error}

    return loss, (metrics, rng_key)


@jax.jit(
    static_argnames=("policy_opt_def", "value_opt_def", "model_def", "config", "value_only"),
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
    bounds_mask = seq_range[None, :] < rollout.kv_cache_lengths[:, None]
    policy_mask = jnp.logical_and(rollout.policy_mask, bounds_mask)

    values = jnp.where(bounds_mask, rollout.values, 0.0)

    rollout = rollout._replace(policy_mask=policy_mask, values=values)

    turn_boundries = ~rollout.policy_mask[:, :-1] & rollout.policy_mask[:, 1:]
    td_discount = jnp.where(turn_boundries, config.turn_discount, config.gae_discount)
    # gae_lambda = jnp.where(turn_boundries, config.turn_lambda, config.gae_lambda)

    # advantages, targets = calculate_advantages(
    #     jnp.asarray(rollout.rewards), values, td_discount, gae_lambda
    # )

    wrt =  value_opt.wrt
    if not value_only:
        wrt = nnx.Any(policy_opt.wrt, wrt)

    diff = nnx.DiffState(0, wrt)
    grad, (metrics, rng_key) = nnx.grad(loss_fn, argnums=diff, has_aux=True)(
        model, rollout, td_discount, config, bounds_mask, value_only, rng_key
    )

    if not value_only:
        policy_opt.update(model, grad)
    value_opt.update(model, grad)

    # metrics["value"] = values.mean(where=bounds_mask)
    metrics["episode_length"] = rollout.kv_cache_lengths.mean()

    policy_opt_state = None if value_only else nnx.state(policy_opt)
    return policy_opt_state, nnx.state(value_opt), nnx.state(model), metrics, rng_key
