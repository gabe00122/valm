"""Integration tests for the training data flow on a tiny model.

These drive the real update path — nnx.split/merge, jit'd update steps,
gradient-accumulation chunking, and the Trainer/GRPOTrainer listeners — with a
few-thousand-parameter Qwen3 so the whole flow runs in seconds. They check the
system-level invariants:

- only the intended parameters move (LoRA for the policy, ValueParam for the
  critic, base weights frozen),
- the first update on freshly generated (on-policy) log probs has ratio ~1,
- metrics come back finite and shaped for the visualizer,
- the trainers wire batches through to downstream listeners, the logger, and
  the checkpointer.
"""

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from valm.buffer import UpdateBatch
from valm.config import (
    AdamWConfig,
    ArithmeticEnvConfig,
    Config,
    GRPOLossConfig,
    LLMConfig,
    LoggerConfig,
    LoraConfig,
    MseCriticConfig,
    OptimizerConfig,
    PPOLossConfig,
    ValueConfig,
)
from valm.episode_listener.grpo_trainer import GRPOTrainer
from valm.episode_listener.trainer import Trainer
from valm.model.qwen3 import Qwen3
from valm.model.value_network import ValueParam
from valm.update_step.grpo import calculate_grpo_advantage, multi_grpo_update_bucketed
from valm.update_step.ppo import multi_update_step_bucket
from valm.utils.optimizer import make_optimizer

VOCAB = 32
BATCH = 4
SEQ = 32

BASE_PARAMS = nnx.All(nnx.Param, nnx.Not(nnx.Any(nnx.LoRAParam, ValueParam)))


def _llm_config() -> LLMConfig:
    return LLMConfig(
        embed=8,
        q_heads=2,
        kv_heads=1,
        num_layers=2,
        head_dim=8,
        vocab_size=VOCAB,
        mlp_ffw_size=16,
    )


def _value_config() -> ValueConfig:
    return ValueConfig(
        latent_encoder_rank=4,
        backbone=LLMConfig(
            embed=8, q_heads=2, kv_heads=1, num_layers=1, head_dim=8, mlp_ffw_size=16
        ),
        head=MseCriticConfig(),
    )


def _make_model(with_value_net: bool) -> Qwen3:
    model = Qwen3(_llm_config(), rngs=nnx.Rngs(0))
    if with_value_net:
        model.initialize_value_net(_value_config(), rngs=nnx.Rngs(1))
    model.initialize_lora(LoraConfig(attn=True, mlp=True, rank=2), rngs=nnx.Rngs(2))
    return model


def _ppo_loss_config() -> PPOLossConfig:
    return PPOLossConfig(
        gae_lambda=0.95,
        gae_discount=0.99,
        turn_lambda=1.0,
        turn_discount=1.0,
        pg_clip_high=0.2,
        pg_clip_low=0.2,
        entropy_coef=0.01,
    )


def _grpo_loss_config() -> GRPOLossConfig:
    return GRPOLossConfig(pg_clip_high=0.2, pg_clip_low=0.2, entropy_coef=0.01)


def _rollout_batch(model: Qwen3, rewards_by_group: bool = False) -> UpdateBatch:
    """Build an on-policy batch: log_probs come from the model itself, so the
    first update sees importance ratios of ~1."""
    rng = np.random.default_rng(0)
    context = rng.integers(0, VOCAB, size=(BATCH, SEQ)).astype(np.int32)
    context_length = np.array([24, 32, 20, 28], dtype=np.int32)

    positions = np.arange(SEQ, dtype=np.int32)
    in_bounds = positions[None, :] < context_length[:, None]
    policy_mask = (positions[None, :] >= 4) & in_bounds

    rewards = np.zeros((BATCH, SEQ), dtype=np.float32)
    if rewards_by_group:
        # distinct totals inside each group so GRPO advantages are non-trivial
        rewards[np.arange(BATCH), context_length - 1] = [0.0, 1.0, 1.0, 0.0]
    else:
        rewards[np.arange(BATCH), context_length - 1] = 1.0

    logits, _, _, _ = model(
        jnp.asarray(context),
        jnp.repeat(jnp.arange(SEQ, dtype=jnp.int32)[None, :], BATCH, 0),
        rewards=jnp.asarray(rewards, dtype=jnp.bfloat16),
        rng_key=jax.random.key(0),
    )
    log_probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    log_probs = jnp.take_along_axis(
        log_probs, jnp.asarray(context[:, 1:, None]), axis=-1
    )[..., 0]

    return UpdateBatch(
        context_length=context_length,
        context=context,
        log_probs=np.asarray(log_probs, dtype=np.float32),
        rewards=rewards,
        policy_mask=policy_mask,
        turn_counts=np.ones(BATCH, dtype=np.int32),
        turn_start_positions=np.zeros((BATCH, 2), dtype=np.int32),
        group_id=np.array([0, 0, 1, 1], dtype=np.uint64),
        turn_metrics={"score": rng.uniform(size=(BATCH, 2)).astype(np.float32)},
    )


def _snapshot(model: Qwen3, filter_) -> list[np.ndarray]:
    return [np.array(leaf) for leaf in jax.tree.leaves(jax.device_get(nnx.state(model, filter_)))]


def _any_changed(before: list[np.ndarray], after: list[np.ndarray]) -> bool:
    return any(not np.array_equal(b, a) for b, a in zip(before, after, strict=True))


def _assert_finite_tree(tree, name: str):
    for leaf in jax.tree.leaves(tree):
        assert np.all(np.isfinite(leaf)), f"non-finite values in {name}"


def _optimizer(model, wrt, lr=1e-3):
    return make_optimizer(model, OptimizerConfig(opt=AdamWConfig(lr=lr)), 10, None, wrt)


def test_ppo_update_trains_lora_and_value_but_not_base():
    model = _make_model(with_value_net=True)
    batch = _rollout_batch(model)

    base_before = _snapshot(model, BASE_PARAMS)
    lora_before = _snapshot(model, nnx.LoRAParam)
    value_before = _snapshot(model, ValueParam)

    policy_opt = _optimizer(model, nnx.LoRAParam)
    value_opt = _optimizer(model, ValueParam)
    policy_opt_def, policy_opt_state = nnx.split(policy_opt)
    value_opt_def, value_opt_state = nnx.split(value_opt)
    model_def, model_state = nnx.split(model)

    (
        policy_opt_state,
        value_opt_state,
        model_state,
        summary,
        token_metrics,
        _,
    ) = multi_update_step_bucket(
        policy_opt_def,
        policy_opt_state,
        value_opt_def,
        value_opt_state,
        model_def,
        model_state,
        jax.random.key(1),
        batch,
        _ppo_loss_config(),
        steps=2,
        value_only=False,
    )

    updated = nnx.merge(model_def, model_state)
    assert _any_changed(lora_before, _snapshot(updated, nnx.LoRAParam))
    assert _any_changed(value_before, _snapshot(updated, ValueParam))
    assert not _any_changed(base_before, _snapshot(updated, BASE_PARAMS))

    _assert_finite_tree(summary, "summary metrics")
    # on-policy log probs -> importance ratio ~1 on the first update
    assert abs(float(summary["approx_kl"])) < 1e-2
    assert float(summary["clip_fraction"]) <= 0.05

    # token metrics are padded to seq_length + 1 for the visualizer
    for name in ("value_loss", "value", "advantage", "actor_loss", "clipped_tokens"):
        assert token_metrics[name].shape == (BATCH, SEQ + 1), name


def test_ppo_value_only_update_leaves_policy_untouched():
    model = _make_model(with_value_net=True)
    batch = _rollout_batch(model)

    lora_before = _snapshot(model, nnx.LoRAParam)
    value_before = _snapshot(model, ValueParam)

    value_opt = _optimizer(model, ValueParam)
    value_opt_def, value_opt_state = nnx.split(value_opt)
    model_def, model_state = nnx.split(model)

    (
        policy_opt_state,
        value_opt_state,
        model_state,
        summary,
        token_metrics,
        _,
    ) = multi_update_step_bucket(
        None,
        None,
        value_opt_def,
        value_opt_state,
        model_def,
        model_state,
        jax.random.key(1),
        batch,
        _ppo_loss_config(),
        steps=1,
        value_only=True,
    )

    assert policy_opt_state is None
    updated = nnx.merge(model_def, model_state)
    assert not _any_changed(lora_before, _snapshot(updated, nnx.LoRAParam))
    assert _any_changed(value_before, _snapshot(updated, ValueParam))
    _assert_finite_tree(summary, "summary metrics")
    assert "actor_loss" not in summary


def test_grpo_update_trains_lora_only_with_group_advantages():
    model = _make_model(with_value_net=False)
    batch = _rollout_batch(model, rewards_by_group=True)

    base_before = _snapshot(model, BASE_PARAMS)
    lora_before = _snapshot(model, nnx.LoRAParam)

    policy_opt = _optimizer(model, nnx.LoRAParam)
    policy_opt_def, policy_opt_state = nnx.split(policy_opt)
    model_def, model_state = nnx.split(model)

    (
        policy_opt_state,
        model_state,
        summary,
        token_metrics,
        _,
    ) = multi_grpo_update_bucketed(
        policy_opt_def,
        policy_opt_state,
        model_def,
        model_state,
        jax.random.key(1),
        batch,
        _grpo_loss_config(),
        steps=2,
        group_size=2,
    )

    updated = nnx.merge(model_def, model_state)
    assert _any_changed(lora_before, _snapshot(updated, nnx.LoRAParam))
    assert not _any_changed(base_before, _snapshot(updated, BASE_PARAMS))

    _assert_finite_tree(summary, "summary metrics")
    assert abs(float(summary["approx_kl"])) < 1e-2

    # every token in an episode shares that episode's group advantage, and the
    # advantages match the reference computation on the full (pre-chunk) batch
    expected = calculate_grpo_advantage(batch.rewards, group_size=2)
    token_adv = token_metrics["advantage"]
    assert token_adv.shape == (BATCH, SEQ + 1)
    for row, adv in zip(token_adv, expected, strict=True):
        np.testing.assert_allclose(row[: SEQ - 1], adv, rtol=1e-5, atol=1e-6)


class _ModelProvider:
    def __init__(self, model: Qwen3):
        self.model_def, self.model_state = nnx.split(model)


class _FakeCheckpointer:
    def __init__(self):
        self.saved_steps: list[int] = []

    def save(self, targets, step, filter_):
        self.saved_steps.append(step)


class _FakeLogger:
    def __init__(self):
        self.logged: list[tuple[dict, int]] = []

    def log_dict(self, metrics, step):
        self.logged.append((metrics, step))


class _Collector:
    def __init__(self):
        self.batches: list[UpdateBatch] = []

    def on_episodes(self, batch: UpdateBatch):
        self.batches.append(batch)


def _base_config(loss, group_size=1) -> Config:
    return Config(
        seed=0,
        base_model="unused",
        lora=LoraConfig(attn=True, mlp=True, rank=2),
        value_net=_value_config(),
        logger=LoggerConfig(),
        policy_optimizer=OptimizerConfig(opt=AdamWConfig(lr=1e-3)),
        value_optimizer=OptimizerConfig(opt=AdamWConfig(lr=1e-3)),
        loss=loss,
        env=ArithmeticEnvConfig(max_x=10, max_y=10),
        eval_envs=BATCH,
        update_envs=BATCH,
        max_seq_length=SEQ,
        total_update_episodes=4,
        checkpoint_every=1,
        group_size=group_size,
    )


def test_trainer_updates_model_and_notifies_listener_logger_checkpointer():
    model = _make_model(with_value_net=True)
    batch = _rollout_batch(model)
    provider = _ModelProvider(model)
    state_before = provider.model_state
    checkpointer = _FakeCheckpointer()
    logger = _FakeLogger()
    collector = _Collector()

    trainer = Trainer(
        provider,
        _optimizer(model, nnx.LoRAParam),
        _optimizer(model, ValueParam),
        jax.random.key(0),
        checkpointer,
        logger,
        _base_config(_ppo_loss_config()),
        episode_listener=collector,
    )

    trainer.on_episodes(batch)

    assert provider.model_state is not state_before
    assert trainer.progress == 1 / 4

    # downstream listener gets the batch enriched with token-aligned metrics
    assert len(collector.batches) == 1
    assert "value" in collector.batches[0].update_metrics
    np.testing.assert_array_equal(collector.batches[0].context, batch.context)

    # logger receives the summary, including summarized env metrics
    (metrics, step), = logger.logged
    assert step == 0
    assert "value_loss" in metrics and "approx_kl" in metrics
    np.testing.assert_allclose(
        metrics["env"]["score"],
        np.mean(batch.turn_metrics["score"].sum(axis=1)),
        rtol=1e-6,
    )

    # checkpoint_every=1 -> checkpoint after the first update
    assert checkpointer.saved_steps == [1]


def test_grpo_trainer_updates_model_and_notifies_listener_logger_checkpointer():
    model = _make_model(with_value_net=False)
    batch = _rollout_batch(model, rewards_by_group=True)
    provider = _ModelProvider(model)
    checkpointer = _FakeCheckpointer()
    logger = _FakeLogger()
    collector = _Collector()

    trainer = GRPOTrainer(
        provider,
        _optimizer(model, nnx.LoRAParam),
        jax.random.key(0),
        checkpointer,
        logger,
        _base_config(_grpo_loss_config(), group_size=2),
        episode_listener=collector,
    )

    trainer.on_episodes(batch)

    assert trainer.progress == 1 / 4
    assert len(collector.batches) == 1
    assert "advantage" in collector.batches[0].update_metrics
    (metrics, step), = logger.logged
    assert step == 0
    assert "actor_loss" in metrics
    assert checkpointer.saved_steps == [1]
