from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp
from llmrl.config import LLMConfig, LoraConfig
from llmrl.model.lora import LoRAGeneral
from llmrl.model.rope import apply_rope
from llmrl.model.util import load_param
from llmrl.util import batched_put


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


class AttentionLayer(nnx.Module):
    def __init__(self, config: LLMConfig, *, rngs: nnx.Rngs) -> None:
        super().__init__()

        self._num_kv_heads = config.kv_heads
        self._q_heads = config.q_heads
        self._embed_dim = config.embed
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        self.key_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.value_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.kv_heads, config.head_dim),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.query_proj = nnx.LinearGeneral(
            in_features=config.embed,
            out_features=(config.q_heads, config.head_dim),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(config.q_heads, config.head_dim),
            out_features=config.embed,
            axis=(-2, -1),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            use_bias=False,
            rngs=rngs,
        )

        self.query_norm = nnx.RMSNorm(
            config.head_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            epsilon=config.norm_eps,
            rngs=rngs,
        )
        self.key_norm = nnx.RMSNorm(
            config.head_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            epsilon=config.norm_eps,
            rngs=rngs,
        )

        self._use_lora = False

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        if not lora_config.attn:
            self._use_lora = False
            return

        self._use_lora = True
        self.key_proj_lora = LoRAGeneral(
            self._embed_dim,
            lora_config.rank,
            (self._num_kv_heads, self._head_dim),
            rngs=rngs,
        )
        self.value_proj_lora = LoRAGeneral(
            self._embed_dim,
            lora_config.rank,
            (self._num_kv_heads, self._head_dim),
            rngs=rngs,
        )
        self.query_proj_lora = LoRAGeneral(
            self._embed_dim,
            lora_config.rank,
            (self._q_heads, self._head_dim),
            rngs=rngs,
        )
        self.out_lora = LoRAGeneral(
            (self._q_heads, self._head_dim),
            lora_config.rank,
            self._embed_dim,
            rngs=rngs,
        )

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

        if self._use_lora:
            key = key + self.key_proj_lora(inputs)
            value = value + self.value_proj_lora(inputs)
            query = query + self.query_proj_lora(inputs)

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
        if self._use_lora:
            out = out + self.out_lora(x)

        return out, carry

    def load_params(self, params):
        k_proj = params["k_proj"]["weight"].T.reshape(self.key_proj.kernel.shape)
        q_proj = params["q_proj"]["weight"].T.reshape(self.query_proj.kernel.shape)
        v_proj = params["v_proj"]["weight"].T.reshape(self.value_proj.kernel.shape)
        o_proj = params["o_proj"]["weight"].T.reshape(self.out.kernel.shape)

        load_param(self.key_proj.kernel, k_proj)
        load_param(self.query_proj.kernel, q_proj)
        load_param(self.value_proj.kernel, v_proj)
        load_param(self.out.kernel, o_proj)

        load_param(self.query_norm.scale, params["q_norm"]["weight"])
        load_param(self.key_norm.scale, params["k_norm"]["weight"])
