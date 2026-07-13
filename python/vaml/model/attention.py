from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp
from vaml.config import LLMConfig, LoraConfig
from vaml.model.lora import LoRALinear
from vaml.model.rope import apply_rope
from vaml.model.util import load_param
from vaml.util import batched_put


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


def _chunk_attention(
    query: jax.Array,      # [B, T, Hq, D]
    k_cache: jax.Array,    # [B, S, Hkv, D]
    v_cache: jax.Array,
    k_chunk: jax.Array,    # [B, T, Hkv, D], this chunk's k/v (also in cache)
    v_chunk: jax.Array,
    positions: jax.Array,  # [B, T], non-decreasing per row (repeat-last pad)
) -> jax.Array:
    """Causal attention for a multi-token chunk against the KV cache: query
    token t of row b attends cache positions <= positions[b, t].

    cuDNN's causal mask is top-left aligned (query i vs key i), but chunk
    queries sit at cache offset prefix_len — so a single flash call cannot
    express this. Instead: two flash calls merged by logsumexp, the standard
    split-KV trick. (1) chunk vs the pre-existing prefix — every prefix key
    precedes every query, padding mask only; (2) chunk vs itself, where
    top-left causal IS correct. ~7x faster than a masked einsum at T=512."""
    prefix_lens = positions[:, 0]                        # cache rows before chunk
    chunk_lens = positions[:, -1] - positions[:, 0] + 1  # real (unpadded) tokens

    o1, lse1 = jax.nn.dot_product_attention(
        query, k_cache, v_cache,
        key_value_seq_lengths=prefix_lens,
        implementation="cudnn", return_residual=True,
    )
    # chunk_lens (not T): padded repeat-last queries then attend exactly the
    # keys the real last token attends, keeping their outputs — and therefore
    # the duplicate cache writes in later layers — bit-identical.
    o2, lse2 = jax.nn.dot_product_attention(
        query, k_chunk, v_chunk,
        is_causal=True, key_value_seq_lengths=chunk_lens,
        implementation="cudnn", return_residual=True,
    )
    lse1, lse2 = lse1.astype(jnp.float32), lse2.astype(jnp.float32)
    m = jnp.maximum(lse1, lse2)
    # a row's prefix can be empty (lse = -inf); its exp must be exactly 0
    w1 = jnp.where(jnp.isneginf(lse1), 0.0, jnp.exp(lse1 - m))[..., None]
    w2 = jnp.where(jnp.isneginf(lse2), 0.0, jnp.exp(lse2 - m))[..., None]
    out = (o1.astype(jnp.float32) * w1 + o2.astype(jnp.float32) * w2) / (w1 + w2)
    return out.astype(query.dtype)


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

        # q, k, v fused into one GEMM; split along the heads axis after.
        self.qkv_proj = LoRALinear(
            in_features=config.embed,
            out_features=(config.q_heads + 2 * config.kv_heads, config.head_dim),
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

        self.qkv_proj.initialize_lora(
            lora_config.rank,
            rngs=rngs,
        )
        self.out.initialize_lora(
            lora_config.rank,
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
        qkv = self.qkv_proj(inputs)
        query = qkv[:, :, : self._q_heads]
        key = qkv[:, :, self._q_heads : self._q_heads + self._num_kv_heads]
        value = qkv[:, :, self._q_heads + self._num_kv_heads :]

        key = self.key_norm(key)
        query = self.query_norm(query)

        key = apply_rope(key, positions, self._head_dim, self._rope_theta)
        query = apply_rope(query, positions, self._head_dim, self._rope_theta)

        if carry is not None:
            carry = self._update_carry(carry, positions, key, value)

            if inputs.shape[1] == 1:
                x = jax.nn.dot_product_attention(
                    query,
                    carry.key,
                    carry.value,
                    key_value_seq_lengths=positions[:, -1] + 1,
                    implementation="cudnn",
                )
            else:
                # multi-token prefill chunk: per-token causality against the
                # cache, which one cuDNN call cannot express
                x = _chunk_attention(
                    query, carry.key, carry.value, key, value, positions
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
        import jax.numpy as jnp

        qkv_proj = jnp.concatenate(
            [
                params["q_proj"]["weight"].T,
                params["k_proj"]["weight"].T,
                params["v_proj"]["weight"].T,
            ],
            axis=-1,
        ).reshape(self.qkv_proj.linear.shape)
        o_proj = params["o_proj"]["weight"].T.reshape(self.out.linear.shape)

        self.qkv_proj.load_params(qkv_proj)
        self.out.load_params(o_proj)

        load_param(self.query_norm.scale, params["q_norm"]["weight"])
        load_param(self.key_norm.scale, params["k_norm"]["weight"])
