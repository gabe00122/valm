import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from valm.config import LLMConfig
from valm.model.attention import AttentionLayer


def test_attention_kv_cache_matches_no_cache_causal_path(monkeypatch):
    batch_size = 2
    seq_length = 6
    config = LLMConfig(
        embed=16,
        q_heads=4,
        kv_heads=2,
        num_layers=1,
        head_dim=8,
    )
    layer = AttentionLayer(config, rngs=nnx.Rngs(0))

    inputs = jax.random.normal(
        jax.random.key(1),
        (batch_size, seq_length, config.embed),
        dtype=jnp.float32,
    ).astype(jnp.bfloat16)
    positions = jnp.broadcast_to(
        jnp.arange(seq_length, dtype=jnp.int32),
        (batch_size, seq_length),
    )

    no_cache_output, no_cache_carry = layer(inputs, positions)

    carry = layer.initialize_carry(batch_size, seq_length)
    cached_outputs = []
    for index in range(seq_length):
        output, carry = layer(
            inputs[:, index : index + 1],
            positions[:, index : index + 1],
            carry,
        )
        cached_outputs.append(output)

    cached_output = jnp.concatenate(cached_outputs, axis=1)

    assert no_cache_carry is None
    np.testing.assert_allclose(
        np.asarray(cached_output, dtype=np.float32),
        np.asarray(no_cache_output, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
