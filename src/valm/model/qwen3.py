from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp

from valm.config import LLMConfig, LoraConfig, PolicyNetworkConfig, ValueConfig
from valm.model.layer import Qwen3Layer
from valm.model.lora import kernel_init
from valm.model.policy_network import PolicyNet
from valm.model.util import load_param, wrap_param
from valm.model.value_network import ValueBackbone, ValueParam, ValueRepresentation
from valm.model.params import PolicyParam

class Qwen3(nnx.Module):
    def __init__(
        self,
        config: LLMConfig,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self._config = config

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

        self._tie_word_embeddings = config.tie_word_embeddings
        if not self._tie_word_embeddings:
            self.lm_head = nnx.Param(
                kernel_init(
                    rngs.params(), (config.embed, config.vocab_size), jnp.bfloat16
                )
            )

    def initialize_value_net(self, value_config: ValueConfig, *, rngs: nnx.Rngs):
        self.value_net = ValueBackbone(value_config, self._config.embed, rngs=rngs)
        wrap_param(self.value_net, ValueParam)

    def initialize_policy(
        self,
        *,
        rngs: nnx.Rngs,
        lora: LoraConfig | None = None,
        policy: PolicyNetworkConfig | None = None,
    ):
        if lora is not None:
            for layer in self.layers:
                layer.initialize_lora(lora, rngs=rngs)

        if policy is not None:
            self.policy_net = PolicyNet(policy, latent_size=self._config.embed, rngs=rngs)
            wrap_param(self.policy_net, PolicyParam)

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

        if not self._tie_word_embeddings:
            load_param(self.lm_head, params["lm_head"]["weight"].T)

    def __call__(
        self,
        tokens: jax.Array,
        positions: jax.Array,
        carry: Any = None,
        *,
        rewards: jax.Array | None = None,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, ValueRepresentation | None, Any, jax.Array]:
        has_carry = carry is not None
        carry = carry if has_carry else {}
        base_carry = carry["base"] if has_carry else (None,) * len(self.layers)
        policy_carry = carry.get("policy")

        value_repr = None

        with jax.named_scope("qwen3_embeddings"):
            x = self.embeddings(tokens)

        latents = [x]
        out_carries = []
        for i, (layer, in_carry) in enumerate(zip(self.layers, base_carry)):
            with jax.named_scope(f"qwen3_layer_{i:02d}"):
                if self._config.remat:
                    layer = jax.remat(layer)
                x, out_carry = layer(x, positions, in_carry)

            if hasattr(self, "policy_net") and self.policy_net.base_layer_index == i:
                policy_vec, policy_carry = self.policy_net(x, positions, policy_carry)
                x = x + policy_vec

            latents.append(x)
            out_carries.append(out_carry)

        base_carry = tuple(out_carries) if carry is not None else None

        # the value net is training only so it's not calculated in inference mode
        if not has_carry and hasattr(self, "value_net"):
            with jax.named_scope("qwen3_value_net"):
                value_repr, _, rng_key = self.value_net(
                    latents, rewards, positions, rng_key=rng_key
                )

        with jax.named_scope("qwen3_final_norm"):
            x = self.final_norm(x)

        with jax.named_scope("qwen3_lm_head"):
            if self._tie_word_embeddings:
                weight = self.embeddings.embedding[...].T
            else:
                weight = self.lm_head[...]

            logits = jnp.matmul(x, weight, preferred_element_type=jnp.float32)

        combined_out_carry = { "base": base_carry}
        if policy_carry is not None:
            combined_out_carry["policy"] = policy_carry

        if not has_carry:
            combined_out_carry = None

        return logits, value_repr, combined_out_carry, rng_key

    def initialize_carry(self, batch_size: int, seq_length: int):
        out = {}
        out["base"] = tuple(
            layer.initialize_carry(batch_size, seq_length) for layer in self.layers
        )

        if hasattr(self, "policy_net"):
            out["policy"] = self.policy_net.initialize_carry(batch_size, seq_length)

        return out
