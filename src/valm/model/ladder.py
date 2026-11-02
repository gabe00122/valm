from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp

from valm.config import LLMConfig
from valm.model.layer import Qwen3Layer


class LadderNetEncode(nnx.Module):
    def __init__(
        self,
        latent_size: int,
        latent_encode_rank: int,
        out_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self._normalize = nnx.RMSNorm(
            latent_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self._encode_up = nnx.Linear(
            latent_size,
            latent_encode_rank,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self._up_gate = nnx.Linear(
            latent_size,
            latent_encode_rank,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self._encode_down = nnx.Linear(
            latent_encode_rank,
            out_size,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, *, rng_key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:

        x = jax.lax.stop_gradient(x)
        x = self._normalize(x)
        gate = self._up_gate(x)
        x = self._encode_up(x)
        x = jax.nn.silu(x) * gate
        x = self._encode_down(x)
        return x, rng_key


class LadderNetLayer(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        latent_size: int,
        latent_encode_rank: int,
        *,
        rngs: nnx.Rngs,
    ):
        self._latent_encode = LadderNetEncode(
            latent_size, latent_encode_rank, config.embed, rngs=rngs
        )
        self._layer = Qwen3Layer(config, param_dtype=jnp.float32, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        latent: jax.Array,
        positions: jax.Array,
        carry: Any = None,
        *,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, Any, jax.Array]:
        latent, rng_key = self._latent_encode(latent, rng_key=rng_key)
        x, carry = self._layer(x + latent, positions, carry)
        return x, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int) -> Any:
        return self._layer.initialize_carry(batch_size, seq_length)


class LadderNet(nnx.Module):
    def __init__(self, config: LLMConfig, latent_size: int, latent_encoder_rank: int, last_latent_only: bool, reward_encode: bool, *, rngs: nnx.Rngs):
        self._last_latent_only = last_latent_only

        if reward_encode:
            self._reward_encode = nnx.Linear(
                1,
                config.embed,
                dtype=jnp.bfloat16,
                param_dtype=jnp.float32,
                rngs=rngs,
            )

        self._embedding_encode = LadderNetEncode(
            latent_size,
            latent_encoder_rank,
            config.embed,
            rngs=rngs,
        )

        self.layers = nnx.List(
            [
                LadderNetLayer(
                    config=config,
                    latent_size=latent_size,
                    latent_encode_rank=latent_encoder_rank,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nnx.RMSNorm(
            config.embed,
            epsilon=config.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self,
        latents: list[jax.Array],
        rewards: jax.Array | None,
        positions: jax.Array,
        carry: tuple[Any, ...] | None = None,
        *,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, tuple[Any, ...] | None, jax.Array]:
        x, *layer_latents = latents

        # This assumes no layers are configured
        if self._last_latent_only:
            x = latents[-1]

        take_every = len(layer_latents) // (len(self.layers) or 1)
        layer_latents = layer_latents[::take_every][: len(self.layers)]

        x, rng_key = self._embedding_encode(x, rng_key=rng_key)
        if hasattr(self, '_reward_encode') and rewards is not None:
            x = x + self._reward_encode(rewards[..., None])

        layer_carries = carry if carry is not None else (None,) * len(self.layers)

        next_layer_carry = []
        for layer, latent, layer_carry in zip(self.layers, layer_latents, layer_carries):
            x, carry_out, rng_key = layer(
                x, latent, positions, layer_carry, rng_key=rng_key
            )

            next_layer_carry.append(carry_out)

        carry = tuple(next_layer_carry) if carry is not None else None

        x = self.final_norm(x)
        return x, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int):
        return tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )
