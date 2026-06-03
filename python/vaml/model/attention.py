from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp
from vaml.config import LLMConfig, LoraConfig
from vaml.model.lora import LoRAGeneral, LoRALinear
from vaml.model.rope import apply_rope
from vaml.model.util import load_param
from vaml.util import batched_put


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


class AttentionLayer(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        param_dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        self._num_kv_heads = config.kv_heads
        self._q_heads = config.q_heads
        self._embed_dim = config.embed
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        self.key_proj = LoRALinear(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.value_proj = LoRALinear(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.query_proj = LoRALinear(
            in_features=config.embed,
            out_features=(config.q_heads, config.head_dim),
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.out = LoRALinear(
            in_features=(config.q_heads, config.head_dim),
            out_features=config.embed,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.query_norm = nnx.RMSNorm(
            config.head_dim,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            epsilon=config.norm_eps,
            rngs=rngs,
        )
        self.key_norm = nnx.RMSNorm(
            config.head_dim,
            dtype=jnp.bfloat16,
            param_dtype=param_dtype,
            epsilon=config.norm_eps,
            rngs=rngs,
        )

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        if not lora_config.attn:
            self._use_lora = False
            return

        self.key_proj.initialize_lora(
            lora_config.rank,
            rngs=rngs,
        )
        self.value_proj.initialize_lora(
            lora_config.rank,
            rngs=rngs,
        )
        self.query_proj.initialize_lora(
            lora_config.rank,
            rngs=rngs,
        )
        self.out.initialize_lora(
            lora_config.rank,
            rngs=rngs,
        )

    def merge_lora(self):
        self.key_proj.merge_lora()
        self.value_proj.merge_lora()
        self.query_proj.merge_lora()
        self.out.merge_lora()

    def unmerge_lora(self):
        self.key_proj.unmerge_lora()
        self.value_proj.unmerge_lora()
        self.query_proj.unmerge_lora()
        self.out.unmerge_lora()

    def initialize_carry(self, batch_size: int, seq_length: int) -> KVCache:
        shape = (batch_size, seq_length, self._num_kv_heads, self._head_dim)
        key = jnp.zeros(shape, dtype=jnp.bfloat16)
        value = jnp.zeros(shape, dtype=jnp.bfloat16)

        return KVCache(key, value)

    def _update_carry(
        self,
        carry: KVCache,
        positions: jax.Array,
        key_update: jax.Array,
        value_update: jax.Array,
    ) -> KVCache:
        new_key = batched_put(carry.key, positions, key_update)
        new_value = batched_put(carry.value, positions, value_update)

        return KVCache(new_key, new_value)

    def __call__(
        self, inputs: jax.Array, positions: jax.Array, carry: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        query = self.query_proj(inputs)

        key = self.key_norm(key)
        query = self.query_norm(query)

        key = apply_rope(key, positions, self._head_dim, self._rope_theta)
        query = apply_rope(query, positions, self._head_dim, self._rope_theta)

        if carry is not None:
            carry = self._update_carry(carry, positions, key, value)

            x = jax.nn.dot_product_attention(
                query,
                carry.key,
                carry.value,
                key_value_seq_lengths=positions[:, -1] + 1,
                implementation="cudnn",
            )
        else:
            x = jax.nn.dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                implementation="cudnn",
            )

        out = self.out(x)

        return out, carry

    def load_params(self, params):
        k_proj = params["k_proj"]["weight"].T.reshape(self.key_proj.linear.shape)
        q_proj = params["q_proj"]["weight"].T.reshape(self.query_proj.linear.shape)
        v_proj = params["v_proj"]["weight"].T.reshape(self.value_proj.linear.shape)
        o_proj = params["o_proj"]["weight"].T.reshape(self.out.linear.shape)

        self.key_proj.load_params(k_proj)
        self.query_proj.load_params(q_proj)
        self.value_proj.load_params(v_proj)
        self.out.load_params(o_proj)

        load_param(self.query_norm.scale, params["q_norm"]["weight"])
        load_param(self.key_norm.scale, params["k_norm"]["weight"])
