import typing as tp
from typing import Any, Protocol

import jax
import optax
from einops import rearrange
from flax import nnx
from flax.nnx import variablelib
from jax import numpy as jnp
from jax.scipy.stats import norm

from valm.model.ladder import LadderNet
from valm.config import HlGaussConfig, LLMConfig, MseCriticConfig, ValueConfig
from valm.model.layer import Qwen3Layer

A = tp.TypeVar("A")


class ValueRepresentation(Protocol):
    def __getitem__(self, idx) -> ValueRepresentation: ...

    def value(self) -> jax.Array: ...

    def loss(self, target: jax.Array) -> jax.Array: ...


def calculate_supports(config: HlGaussConfig):
    support = jnp.linspace(
        config.min, config.max, config.n_logits + 1, dtype=jnp.float32
    )
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :]

    return support, centers


class HlGaussValueRepresentation:
    def __init__(self, config: HlGaussConfig, logits: jax.Array):
        self.config = config
        self.logits = logits

    def __getitem__(self, idx):
        return HlGaussValueRepresentation(self.config, self.logits[idx])

    def value(self) -> jax.Array:
        _, centers = calculate_supports(self.config)
        probs = nnx.softmax(self.logits, axis=-1)
        return (probs * centers).sum(-1)

    def loss(self, target: jax.Array) -> jax.Array:
        b, t = target.shape
        supports, _ = calculate_supports(self.config)

        logits = rearrange(self.logits, "b t l -> (b t) l")
        target = rearrange(target, "b t -> (b t)")

        targets = jnp.clip(target, self.config.min, self.config.max)

        cdf_evals = norm.cdf(supports, loc=targets[:, None], scale=self.config.sigma)

        z = cdf_evals[:, -1] - cdf_evals[:, 0]

        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]

        target_probs = bin_probs / z[:, None]

        loss = optax.softmax_cross_entropy(logits, target_probs, axis=-1)
        return loss.reshape(b, t)


class HlGaussHead(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hl_gauss_config: HlGaussConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hl_gauss_config = hl_gauss_config
        self.dense = nnx.Linear(
            in_features,
            hl_gauss_config.n_logits,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> ValueRepresentation:
        x = self.dense(x).astype(jnp.float32)
        return HlGaussValueRepresentation(self.hl_gauss_config, x)


class MseValueRepresentation:
    def __init__(self, values: jax.Array):
        self.values = values

    def __getitem__(self, idx):
        return MseValueRepresentation(self.values[idx])

    def value(self) -> jax.Array:
        return self.values

    def loss(self, target: jax.Array) -> jax.Array:
        return 0.5 * jnp.square(self.values - target)


class MseHead(nnx.Module):
    def __init__(self, in_features: int, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(
            in_features,
            1,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, x) -> ValueRepresentation:
        x = self.dense(x).squeeze(axis=-1)
        return MseValueRepresentation(x)


class ValueParam(variablelib.Param[A]):
    pass


class ValueBackbone(nnx.Module):
    def __init__(self, config: ValueConfig, latent_size: int, *, rngs: nnx.Rngs):
        self._last_latent_only = config.last_latent_only
        self.backbone = LadderNet(
            config.backbone,
            latent_size,
            config.latent_encoder_rank,
            config.last_latent_only,
            True,
            rngs=rngs
        )

        if isinstance(config.head, HlGaussConfig):
            self._head = HlGaussHead(config.backbone.embed, config.head, rngs=rngs)
        elif isinstance(config.head, MseCriticConfig):
            self._head = MseHead(config.backbone.embed, rngs=rngs)


    def __call__(
        self,
        latents: list[jax.Array],
        rewards: jax.Array,
        positions: jax.Array,
        carry: tuple[Any, ...] | None = None,
        *,
        rng_key: jax.Array,
    ) -> tuple[ValueRepresentation, tuple[Any, ...] | None, jax.Array]:
        x, carry, rng_key = self.backbone(latents, rewards, positions, carry, rng_key=rng_key)

        value_repr = self._head(x)
        return value_repr, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int):
        return self.backbone.initialize_carry(batch_size, seq_length)
