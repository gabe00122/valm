from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from vaml.config import LLMConfig, LoraConfig, ValueConfig
from vaml.model.layer import Qwen3Layer
from vaml.model.util import load_param, wrap_param
from vaml.model.value_network import ValueBackbone, ValueParam, ValueRepresentation


class Qwen3(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self._embed = config.embed
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta

        self.embeddings = nnx.Embed(
            config.vocab_size,
            config.embed,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

        self.layers = nnx.List(
            [
                Qwen3Layer(
                    config=config,
                    rngs=rngs,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nnx.RMSNorm(
            config.embed,
            epsilon=config.norm_eps,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=rngs,
        )

    def initialize_value_net(self, value_config: ValueConfig, *, rngs: nnx.Rngs):
        self.value_net = ValueBackbone(value_config, self._embed, rngs=rngs)
        wrap_param(self.value_net, ValueParam)

    def initialize_lora(self, lora_config: LoraConfig, *, rngs: nnx.Rngs):
        for layer in self.layers:
            layer.initialize_lora(lora_config, rngs=rngs)

    def load_params(self, params: dict[str, Any]):
        embed_params = jnp.asarray(
            params["model"]["embed_tokens"]["weight"],
            device=self.embeddings.embedding.device,
        )
        assert embed_params.shape == self.embeddings.embedding.shape

        self.embeddings.embedding[...] = embed_params

        for i, layer in enumerate(self.layers):
            layer_params = params["model"]["layers"][f"{i}"]
            layer.load_params(layer_params)

        load_param(self.final_norm.scale, params["model"]["norm"]["weight"])

    def __call__(
        self,
        tokens: jax.Array,
        positions: jax.Array,
        carry: Any = None,
        *,
        rewards: jax.Array | None = None,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, ValueRepresentation | None, Any, jax.Array]:
        value_repr = None
        checkpoint = True

        with jax.named_scope("qwen3_embeddings"):
            x = self.embeddings(tokens)

        if carry is not None:
            out_carry = []
            for i, (layer, carry_in) in enumerate(zip(self.layers, carry)):
                with jax.named_scope(f"qwen3_layer_{i:02d}"):
                    x, layer_carry_out = layer(x, positions, carry_in)
                out_carry.append(layer_carry_out)

            carry = tuple(out_carry)
        else:
            latents = [x]
            for i, layer in enumerate(self.layers):
                with jax.named_scope(f"qwen3_layer_{i:02d}"):
                    if checkpoint:
                        layer = jax.checkpoint(layer)
                    x, _ = layer(x, positions)
                latents.append(x)

            if hasattr(self, "value_net"):
                with jax.named_scope("qwen3_value_net"):
                    value_repr, _, rng_key = self.value_net(
                        latents, rewards, positions, rng_key=rng_key
                    )

        with jax.named_scope("qwen3_final_norm"):
            x = self.final_norm(x)
        with jax.named_scope("qwen3_lm_head"):
            logits = x @ self.embeddings.embedding.T

        with jax.named_scope("qwen3_logits_to_float32"):
            logits = logits.astype(jnp.float32)

        return logits, value_repr, carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int):
        base_carry = tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )
        value_carry = (
            self.value_net.initialize_carry(batch_size, seq_length)
            if hasattr(self, "value_net")
            else None
        )
        return base_carry, value_carry
