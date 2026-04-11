import math

import jax
from flax import nnx
from jax import numpy as jnp


class LoRAGeneral(nnx.Module):
    def __init__(
        self,
        in_features: int | tuple[int, ...],
        rank: int,
        out_features: int | tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        def prod_features(feat):
            return feat if isinstance(feat, int) else math.prod(feat)

        prod_in = prod_features(in_features)
        prod_out = prod_features(out_features)

        self._out_shape = (
            (out_features,) if isinstance(out_features, int) else out_features
        )

        self.lora = nnx.LoRA(
            prod_in,
            rank,
            prod_out,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch = x.shape[0]
        seq_length = x.shape[1]

        x = x.reshape(batch, seq_length, -1)
        x = self.lora(x)
        x = x.reshape(batch, seq_length, *self._out_shape)

        return x
