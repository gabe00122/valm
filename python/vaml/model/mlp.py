import jax
from flax import nnx
import numpy as np
from jax import numpy as jnp
from vaml.config import LLMConfig, LoraConfig
from vaml.model.util import load_param


class MlpLayer(nnx.Module):
    def __init__(self, config: LLMConfig, *, rngs: nnx.Rngs):
        super().__init__()
        self._embed_dim = config.embed
        self._ffw_dim = config.mlp_ffw_size

        self.up_proj = nnx.Linear(
            config.embed,
            config.mlp_ffw_size * 2,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.mlp_ffw_size,
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self._use_lora = False

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        if not lora_config.mlp:
            self._use_lora = False
            return

        self._use_lora = True
        self.up_proj_lora = nnx.LoRA(
            self._embed_dim,
            lora_config.rank,
            self._ffw_dim * 2,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )
        self.down_proj_lora = nnx.LoRA(
            self._ffw_dim,
            lora_config.rank,
            self._embed_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def load_params(self, params):
        # pass in the mlp dict
        up_proj = np.concatenate([
            params["gate_proj"]["weight"].T,
            params["up_proj"]["weight"].T
        ], axis=-1)
        load_param(self.up_proj.kernel, up_proj)
        load_param(self.down_proj.kernel, params["down_proj"]["weight"].T)

    def __call__(self, inputs):
        up_combined = self.up_proj(inputs)
        if self._use_lora:
            up_combined = up_combined + self.up_proj_lora(inputs)

        gate_in, up_in = jnp.split(up_combined, 2, axis=-1)
        down_in = up_in * jax.nn.silu(gate_in)
        out = self.down_proj(down_in)

        if self._use_lora:
            out = out + self.down_proj_lora(down_in)

        return out
