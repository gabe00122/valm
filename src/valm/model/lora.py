import math

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp

kernel_init = nnx.initializers.lecun_normal()
default_a_initializer = nnx.initializers.he_uniform()
default_b_initializer = nnx.initializers.zeros


def prod_features(feat):
    return feat if isinstance(feat, int) else math.prod(feat)


class LoRALinear(nnx.Module):
    def __init__(
        self,
        in_features: int | tuple[int, ...],
        out_features: int | tuple[int, ...],
        *,
        param_dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self._prod_in = prod_features(in_features)
        self._prod_out = prod_features(out_features)
        self._out_shape = (
            (out_features,) if isinstance(out_features, int) else out_features
        )

        self.use_lora = False

        linear_key = rngs.params()
        self.linear = nnx.Param(
            kernel_init(linear_key, (self._prod_in, self._prod_out), param_dtype)
        )

    def initialize_lora(self, rank: int, *, rngs: nnx.Rngs):
        lora_a_key = rngs.params()
        lora_b_key = rngs.params()

        self.lora_a = nnx.LoRAParam(
            default_a_initializer(lora_a_key, (self._prod_in, rank), jnp.float32)
        )
        self.lora_b = nnx.LoRAParam(
            default_b_initializer(lora_b_key, (rank, self._prod_out), jnp.float32)
        )
        self.use_lora = True

    def load_params(self, param: np.ndarray):
        assert self.linear.shape == param.shape
        assert self.linear.dtype == param.dtype
        self.linear[...] = jnp.asarray(param, device=self.linear[...].device)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch = x.shape[0]
        seq_length = x.shape[1]

        linear_in = x.reshape(batch, seq_length, -1).astype(jnp.bfloat16)

        x = linear_in @ self.linear[...].astype(jnp.bfloat16)
        if self.use_lora:
            lora = linear_in @ self.lora_a[...] @ self.lora_b[...]
            x = x + lora.astype(jnp.bfloat16)
        x = x.reshape(batch, seq_length, *self._out_shape)

        return x


class LoRAGeneral(nnx.Module):
    def __init__(
        self,
        in_features: int | tuple[int, ...],
        rank: int,
        out_features: int | tuple[int, ...],
        *,
        param_dtype=jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
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
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch = x.shape[0]
        seq_length = x.shape[1]

        x = x.reshape(batch, seq_length, -1)
        x = self.lora(x)
        x = x.reshape(batch, seq_length, *self._out_shape)

        return x
