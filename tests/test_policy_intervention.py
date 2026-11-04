"""Behavioral checks for the ladder policy, independent of model downloads.

The attention backend follows the production code (cuDNN, so a GPU is needed).
Failures are deliberately left visible as regression cases for this review.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from valm.config import (
    LLMConfig,
    MseCriticConfig,
    PolicyNetworkConfig,
    ValueConfig,
    load_config,
)
from valm.model.params import PolicyParam
from valm.model.qwen3 import Qwen3
from valm.model.value_network import ValueParam


def _config(num_layers=2):
    return LLMConfig(
        embed=16, q_heads=2, kv_heads=1, num_layers=num_layers,
        head_dim=8, vocab_size=32, mlp_ffw_size=32,
    )


def _model(intervention_layer=1, policy_depth=1):
    model = Qwen3(_config(), rngs=nnx.Rngs(0))
    model.initialize_policy(
        policy=PolicyNetworkConfig(
            backbone=_config(policy_depth), latent_encoder_rank=4,
        ),
        intervention_layer=intervention_layer,
        rngs=nnx.Rngs(1),
    )
    return model


def _inputs():
    tokens = jnp.asarray([[1, 7, 2, 9, 4, 3, 6, 5]], dtype=jnp.int32)
    return tokens, jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]


def _forward(model, tokens=None):
    default_tokens, positions = _inputs()
    return model(
        default_tokens if tokens is None else tokens,
        positions, rng_key=jax.random.key(2),
    )[0]


def _activate_policy(model):
    # A zero head hides errors in the backbone and cache. Use a real residual.
    kernel = model.policy_net.out_proj.kernel
    kernel[...] = 0.1 * jax.random.normal(jax.random.key(3), kernel.shape)


@pytest.mark.parametrize("intervention_layer", [0, 1])
def test_zero_initialized_policy_preserves_base_logits(intervention_layer):
    base = Qwen3(_config(), rngs=nnx.Rngs(0))
    model = _model(intervention_layer)
    np.testing.assert_array_equal(_forward(model), _forward(base))
    state = nnx.state(model.policy_net, PolicyParam)
    assert jax.tree.leaves(state)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(state))
    assert not jax.tree.leaves(nnx.state(model.policy_net, nnx.Not(PolicyParam)))


@pytest.mark.parametrize("intervention_layer", [0, 1])
def test_active_policy_cached_logits_match_full_sequence(intervention_layer):
    model = _model(intervention_layer)
    _activate_policy(model)
    tokens, positions = _inputs()
    expected = _forward(model)
    base = Qwen3(_config(), rngs=nnx.Rngs(0))
    assert not np.allclose(expected, _forward(base), atol=1e-3)
    carry = model.initialize_carry(1, tokens.shape[1])
    outputs = []
    for i in range(tokens.shape[1]):
        logits, value, carry, _ = model(
            tokens[:, i:i + 1], positions[:, i:i + 1], carry,
            rng_key=jax.random.key(2),
        )
        assert value is None
        assert len(carry["policy"]) == 1
        outputs.append(logits)
    np.testing.assert_allclose(
        jnp.concatenate(outputs, axis=1), expected, rtol=2e-2, atol=2e-2,
    )


def test_active_policy_is_causal():
    model = _model(0)
    _activate_policy(model)
    tokens, _ = _inputs()
    changed = tokens.at[:, 4:].set(jnp.asarray([[12, 13, 14, 15]]))
    np.testing.assert_array_equal(_forward(model)[:, :4], _forward(model, changed)[:, :4])


@pytest.mark.parametrize("intervention_layer", [0, 1])
def test_policy_gradient_trains_head_then_backbone_and_freezes_base(intervention_layer):
    model = _model(intervention_layer)
    frozen_filter = nnx.All(nnx.Param, nnx.Not(PolicyParam))
    before = [np.array(x) for x in jax.tree.leaves(nnx.state(model, frozen_filter))]
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=PolicyParam)

    def loss(model):
        logits = _forward(model)
        return -jax.nn.log_softmax(logits, axis=-1)[..., 11].mean()

    grad_fn = nnx.grad(loss, argnums=nnx.DiffState(0, PolicyParam))
    for step in range(2):
        grad = grad_fn(model)
        head = jax.tree.leaves(grad["policy_net"]["out_proj"])
        backbone = jax.tree.leaves(grad["policy_net"]["backbone"])
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
        assert any(np.any(np.asarray(x) != 0) for x in head)
        if step == 0:
            assert all(np.all(np.asarray(x) == 0) for x in backbone)
        else:
            assert any(np.any(np.asarray(x) != 0) for x in backbone)
        optimizer.update(model, grad)

    after = jax.tree.leaves(nnx.state(model, frozen_filter))
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)
    assert not np.allclose(_forward(model), _forward(Qwen3(_config(), rngs=nnx.Rngs(0))))


def test_critic_loss_does_not_backpropagate_into_policy():
    model = _model(0)
    _activate_policy(model)
    model.initialize_value_net(
        ValueConfig(backbone=_config(1), latent_encoder_rank=4, head=MseCriticConfig()),
        rngs=nnx.Rngs(4),
    )
    tokens, positions = _inputs()

    def loss(model):
        _, value, _, _ = model(
            tokens, positions, rewards=jnp.zeros_like(tokens, dtype=jnp.bfloat16),
            rng_key=jax.random.key(2),
        )
        return value.loss(jnp.ones(tokens.shape)).mean()

    grad = nnx.grad(loss, argnums=nnx.DiffState(0, nnx.Any(PolicyParam, ValueParam)))(model)
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(grad["policy_net"]))
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grad["value_net"]))


@pytest.mark.parametrize("intervention_layer", [-1, 2])
def test_invalid_intervention_layer_is_rejected(intervention_layer):
    with pytest.raises(ValueError, match="intervention"):
        model = _model(intervention_layer)
        _forward(model)


def test_policy_deeper_than_available_latents_has_clear_validation():
    # Either support reuse of latents or reject this configuration explicitly;
    # a slice-step-zero exception is not an actionable configuration error.
    with pytest.raises(ValueError, match="latent|layer|depth"):
        _forward(_model(intervention_layer=0, policy_depth=2))


@pytest.mark.parametrize("path", sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.json")), ids=lambda p: p.name)
def test_shipped_config_still_loads(path):
    load_config(path.read_text())


def test_checkpoint_evaluation_uses_training_intervention_layer(monkeypatch):
    import valm.eval as evaluation

    config = SimpleNamespace(
        env=SimpleNamespace(name="arithmetic"), base_model="unused",
        lora=None, policy_net=object(), intervention_layer=0,
    )
    experiment = SimpleNamespace(config=config, params_seed=0)
    monkeypatch.setattr(evaluation.Experiment, "load", lambda *args: experiment)

    class InitializationChecked(Exception):
        pass

    class Model:
        def initialize_policy(self, **kwargs):
            assert kwargs.get("intervention_layer") == config.intervention_layer
            raise InitializationChecked

    monkeypatch.setattr(evaluation, "load_base_model", lambda *args: (Model(), None, None))
    with pytest.raises(InitializationChecked):
        evaluation.eval_checkpoint("unused")
