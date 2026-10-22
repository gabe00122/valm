import jax
from flax import nnx
from jax import numpy as jnp
from valm.config import LLMConfig, LoraConfig
from valm.model.attention import AttentionLayer, KVCache
from valm.model.mlp import MlpLayer
from valm.model.util import load_param


class Qwen3Layer(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        param_dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attn = AttentionLayer(config, param_dtype=param_dtype, rngs=rngs)
        self.mlp = MlpLayer(config, param_dtype=param_dtype, rngs=rngs)

        self.attn_pre_norm = nnx.RMSNorm(
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            epsilon=config.norm_eps,
            rngs=rngs,
        )
        self.attn_post_norm = nnx.RMSNorm(
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            epsilon=config.norm_eps,
            rngs=rngs,
        )

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        self.attn.initialize_lora(lora_config, rngs=rngs)
        self.mlp.initialize_lora(lora_config, rngs=rngs)

    def __call__(
        self, inputs: jax.Array, positions: jax.Array, carry: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        attn_in = self.attn_pre_norm(inputs)
        attn_out, carry = self.attn(attn_in, positions, carry)
        x = inputs + attn_out

        ff_in = self.attn_post_norm(x)
        ff_out = self.mlp(ff_in)
        x = x + ff_out

        return x, carry

    def initialize_carry(self, batch_size: int, seq_length: int):
        return self.attn.initialize_carry(batch_size, seq_length)

    def load_params(self, params):
        load_param(self.attn_pre_norm.scale, params["input_layernorm"]["weight"])
        load_param(
            self.attn_post_norm.scale, params["post_attention_layernorm"]["weight"]
        )
        self.attn.load_params(params["self_attn"])
        self.mlp.load_params(params["mlp"])
