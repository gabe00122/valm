import jax
from flax import nnx
import numpy as np
from jax import numpy as jnp
from vaml.config import LLMConfig, LoraConfig
from vaml.model.util import load_param
from vaml.model.lora import LoRALinear


class MlpLayer(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        param_dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self._embed_dim = config.embed
        self._ffw_dim = config.mlp_ffw_size

        self.up_proj = LoRALinear(
            config.embed,
            config.mlp_ffw_size * 2,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            config.mlp_ffw_size,
            config.embed,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        if not lora_config.mlp:
            self._use_lora = False
            return

        self.up_proj.initialize_lora(lora_config.rank, rngs=rngs)
        self.down_proj.initialize_lora(lora_config.rank, rngs=rngs)

    def merge_lora(self):
        self.up_proj.merge_lora()
        self.down_proj.merge_lora()

    def unmerge_lora(self):
        self.up_proj.unmerge_lora()
        self.down_proj.unmerge_lora()

    def load_params(self, params):
        # pass in the mlp dict
        up_proj = np.concatenate(
            [params["gate_proj"]["weight"].T, params["up_proj"]["weight"].T], axis=-1
        )
        self.up_proj.load_params(up_proj)
        self.down_proj.load_params(params["down_proj"]["weight"].T)

    def __call__(self, inputs):
        up_combined = self.up_proj(inputs)

        gate_in, up_in = jnp.split(up_combined, 2, axis=-1)
        down_in = up_in * jax.nn.silu(gate_in)
        out = self.down_proj(down_in)

        return out
