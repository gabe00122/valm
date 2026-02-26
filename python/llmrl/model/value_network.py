from llmrl.config import HlGaussConfig, LLMConfig, ValueConfig, MseCriticConfig
from llmrl.model.layer import Qwen3Layer

import typing as tp

import jax
from flax import nnx
from flax.nnx import variablelib
from jax import numpy as jnp

A = tp.TypeVar("A")

from typing import Any, Protocol
from einops import rearrange
from flax import nnx
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax


class ValueRepresentation(Protocol):
    def __getitem__(self, idx) -> "ValueRepresentation":
        ...

    def value(self) -> jax.Array:
        ...

    def loss(self, target: jax.Array) -> jax.Array:
        ...


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
        self, in_features: int, hl_gauss_config: HlGaussConfig, *, rngs: nnx.Rngs
    ) -> None:
        self.hl_gauss_config = hl_gauss_config
        self.dense = nnx.Linear(in_features, hl_gauss_config.n_logits, param_dtype=jnp.bfloat16, rngs=rngs)

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
        self.dense = nnx.Linear(in_features, 1, rngs=rngs)

    def __call__(self, x) -> ValueRepresentation:
        x = self.dense(x).squeeze(axis=-1)
        return MseValueRepresentation(x)


class ValueParam(variablelib.Param[A]):
    pass


class ValueNetEncode(nnx.Module):
    def __init__(self, latent_size: int, latent_encode_rank: int, out_size: int, *, rngs: nnx.Rngs):
        self._dropout = nnx.Dropout(0.1)
        self._normalize = nnx.RMSNorm(latent_size, rngs=rngs)
        self._encode_up = nnx.Linear(
            latent_size,
            latent_encode_rank,
            param_dtype=jnp.bfloat16,
            rngs=rngs
        )
        self._up_gate = nnx.Linear(
            latent_size,
            latent_encode_rank,
            param_dtype=jnp.bfloat16,
            rngs=rngs
        )
        self._encode_down = nnx.Linear(
            latent_encode_rank,
            out_size,
            param_dtype=jnp.bfloat16,
            rngs=rngs
        )

    def __call__(self, x: jax.Array, *, rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        rng_key, dropout_rng = jax.random.split(rng_key)

        x = jax.lax.stop_gradient(x)
        x = self._dropout(x, rngs=dropout_rng)
        x = self._normalize(x)
        gate = self._up_gate(x)
        x = self._encode_up(x)
        x = jax.nn.silu(x) * gate
        x = self._encode_down(x)
        return x, rng_key


class ValueNetLayer(nnx.Module):
    def __init__(self, config: LLMConfig, latent_size: int, latent_encode_rank: int, *, rngs: nnx.Rngs):
        self._latent_encode = ValueNetEncode(latent_size, latent_encode_rank, config.embed, rngs=rngs)
        self._layer = Qwen3Layer(config, rngs=rngs)

    def __call__(self, x: jax.Array, latent: jax.Array, positions: jax.Array, carry: Any = None, *, rng_key: jax.Array) -> tuple[jax.Array, Any, jax.Array]:
        latent, rng_key = self._latent_encode(latent, rng_key=rng_key)
        x, carry = self._layer(x + latent, positions, carry)
        return x, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int) -> Any:
        return self._layer.initialize_carry(batch_size, seq_length)


class ValueBackbone(nnx.Module):
    def __init__(self, config: ValueConfig, latent_size: int, *, rngs: nnx.Rngs):
        self._embeding_encode = ValueNetEncode(latent_size, config.laten_encoder_rank, config.backbone.embed, rngs=rngs)

        self.layers = nnx.List([
            ValueNetLayer(
                config=config.backbone,
                latent_size=latent_size,
                latent_encode_rank=config.laten_encoder_rank,
                rngs=rngs,
            ) for _ in range(config.backbone.num_layers)
        ])

        self.final_norm = nnx.RMSNorm(
            config.backbone.embed,
            epsilon=config.backbone.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        if isinstance(config.head, HlGaussConfig):
            self._head = HlGaussHead(config.backbone.embed, config.head, rngs=rngs)
        elif isinstance(config.head, MseCriticConfig):
            self._head = MseHead(config.backbone.embed, rngs=rngs)

    def __call__(self, latents: list[jax.Array], positions: jax.Array, carry: tuple[Any, ...] | None = None, *, rng_key: jax.Array) -> tuple[ValueRepresentation, tuple[Any, ...] | None, jax.Array]:
        x, *layer_latents = latents

        take_every = len(layer_latents) // len(self.layers)
        layer_latents = layer_latents[::take_every][:len(self.layers)]

        x, rng_key = self._embeding_encode(x, rng_key=rng_key)

        if carry is not None:
            out_carry = []
            for layer, latent, carry_in in zip(self.layers, layer_latents, carry):
                x, carry_out, rng_key = layer(x, latent, positions, carry_in, rng_key=rng_key)

                out_carry.append(carry_out)

            carry = tuple(out_carry)
        else:
            for layer, latent in zip(self.layers, layer_latents):
                x, _, rng_key = layer(x, latent, positions, rng_key=rng_key)

        x = self.final_norm(x)
        value_repr = self._head(x)
        return value_repr, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int):
        return tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )

