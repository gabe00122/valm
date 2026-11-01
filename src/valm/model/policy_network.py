from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp

from valm.config import PolicyNetworkConfig
from valm.model.params import PolicyParam  # noqa: F401
from valm.model.qwen3 import Qwen3Layer


class PolicyNet(nnx.Module):
    def __init__(
        self, config: PolicyNetworkConfig, latent_size: int, *, rngs: nnx.Rngs
    ):
        self._config = config
        self.base_layer_index = config.base_layer

        self.in_proj = nnx.Linear(
            latent_size,
            config.backbone.embed,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

        self.out_proj = nnx.Linear(
            config.backbone.embed,
            latent_size,
            use_bias=False,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        self.layers = nnx.List(
            [
                Qwen3Layer(
                    config=config.backbone,
                    rngs=rngs,
                )
                for _ in range(config.backbone.num_layers)
            ]
        )

    def __call__(
        self, x: jax.Array, positions: jax.Array, carry: tuple[Any, ...] | None = None
    ) -> tuple[jax.Array, tuple[Any, ...] | None]:
        x = self.in_proj(x)

        layers_carry = carry if carry is not None else (None,) * len(self.layers)
        out_carry = []
        for layer, carry_in in zip(self.layers, layers_carry):
            x, carry_out = layer(x, positions, carry_in)
            out_carry.append(carry_out)

        carry = tuple(out_carry) if carry is not None else None

        x = self.out_proj(x)
        return x, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        return tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )
