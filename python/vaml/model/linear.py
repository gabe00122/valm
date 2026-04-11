import jax
from flax import nnx
from jax import numpy as jnp


class CustomLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nnx.Param(jnp.zeros((in_features, out_features), dtype=jnp.bfloat16))
    
    def init_lora(self, rank: int, rngs: nnx.Rngs):
        if rank is not None:
            lora_init = nnx.initializers.he_uniform()
            self.lora_up = nnx.LoRAParam(lora_init(rngs.param(), (self.in_features, rank), dtype=jnp.bfloat16))
            self.lora_down = nnx.LoRAParam(jnp.zeros((rank, self.out_features), dtype=jnp.bfloat16))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        ...
