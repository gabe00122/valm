import jax
import jax.numpy as jnp
from flax import nnx

from valm.config import (
    HlGaussConfig,
    LLMConfig,
    LoraConfig,
    MseCriticConfig,
    ValueConfig,
)
from valm.model.lora import LoRALinear
from valm.model.qwen3 import Qwen3
from valm.model.value_network import ValueParam


def _tiny_llm_config() -> LLMConfig:
    return LLMConfig(
        embed=8,
        q_heads=2,
        kv_heads=1,
        num_layers=1,
        head_dim=4,
        vocab_size=16,
        mlp_ffw_size=16,
    )


def _state_dtypes(state) -> set[jnp.dtype]:
    return {getattr(leaf, "value", leaf).dtype for leaf in jax.tree.leaves(state)}


def _layer_dtypes(model) -> dict[str, jnp.dtype]:
    return {
        ".".join(str(part) for part in path): jnp.dtype(node.dtype)
        for path, node in nnx.iter_graph(model)
        if hasattr(node, "dtype") and hasattr(node, "param_dtype")
    }


def test_lora_params_are_float32():
    model = Qwen3(_tiny_llm_config(), rngs=nnx.Rngs(0))
    model.initialize_lora(
        LoraConfig(attn=True, mlp=True, rank=2),
        rngs=nnx.Rngs(1),
    )

    dtypes = _state_dtypes(nnx.state(model, nnx.LoRAParam))

    assert dtypes == {jnp.dtype(jnp.float32)}


def test_custom_lora_linear_params_are_float32():
    layer = LoRALinear(4, 4, rngs=nnx.Rngs(0))
    layer.initialize_lora(2, rngs=nnx.Rngs(1))

    dtypes = _state_dtypes(nnx.state(layer, nnx.LoRAParam))

    assert dtypes == {jnp.dtype(jnp.float32)}


def test_value_net_params_are_float32():
    llm_config = _tiny_llm_config()
    model = Qwen3(llm_config, rngs=nnx.Rngs(0))
    model.initialize_value_net(
        ValueConfig(
            latent_encoder_rank=4,
            backbone=llm_config,
            head=MseCriticConfig(),
        ),
        rngs=nnx.Rngs(1),
    )

    dtypes = _state_dtypes(nnx.state(model, ValueParam))

    assert dtypes == {jnp.dtype(jnp.float32)}


def test_layer_compute_dtypes_remain_bfloat16():
    for head in (
        MseCriticConfig(),
        HlGaussConfig(min=-1.0, max=1.0, n_logits=3, sigma=1.0),
    ):
        llm_config = _tiny_llm_config()
        model = Qwen3(llm_config, rngs=nnx.Rngs(0))
        model.initialize_lora(
            LoraConfig(attn=True, mlp=True, rank=2),
            rngs=nnx.Rngs(1),
        )
        model.initialize_value_net(
            ValueConfig(
                latent_encoder_rank=4,
                backbone=llm_config,
                head=head,
            ),
            rngs=nnx.Rngs(2),
        )

        bad_dtypes = {
            path: dtype
            for path, dtype in _layer_dtypes(model).items()
            if dtype != jnp.dtype(jnp.bfloat16)
        }

        assert bad_dtypes == {}
