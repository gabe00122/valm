from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp

from valm.config import PolicyNetworkConfig
from valm.model.ladder import LadderNet


class PolicyNet(nnx.Module):
    def __init__(
        self, config: PolicyNetworkConfig, latent_size: int, *, rngs: nnx.Rngs
    ):
        self._config = config

        # this probably should be reward conditioned otherwise it's a information asymetry
        self.backbone = LadderNet(config.backbone, latent_size, config.latent_encoder_rank, False, False, rngs=rngs)
        self.out_proj = nnx.Linear(
            config.backbone.embed,
            latent_size,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(
        self,
        latents: list[jax.Array],
        positions: jax.Array,
        carry: tuple[Any, ...] | None = None,
        *,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, tuple[Any, ...] | None, jax.Array]:
        x, carry, rng_key = self.backbone(latents, None, positions, carry, rng_key=rng_key)
        x = self.out_proj(x)
        return x, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int):
        return self.backbone.initialize_carry(batch_size, seq_length)
