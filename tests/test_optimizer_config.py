import json

from flax import nnx
from vaml.config import load_config
from vaml.utils.optimizer import make_optimizer


class DummyModel(nnx.Module):
    def __init__(self, rngs):
        self.param = nnx.Param(nnx.Rngs(0).params())


def test_config_variants():
    # 1. Test AdamW + WarmupCosine + MultiStep
    config_json = {
        "seed": 42,
        "base_model": "test",
        "lora": {"mlp": True, "attn": True, "rank": 8},
        "logger": {},
        "policy_optimizer": {
            "opt": {
                "type": "adamw",
                "lr": 1e-4,
                "beta1": 0.9,
                "beta2": 0.95,
                "weight_decay": 0.01,
            },
            "schedule": {"type": "warmup_cosine", "warmup_ratio": 0.1},
            "multi_step": 2,
        },
        "value_optimizer": {"opt": {"type": "adamw", "lr": 1e-4}},
        "value_net": {
            "latent_encoder_rank": 64,
            "backbone": {
                "embed": 64,
                "q_heads": 4,
                "kv_heads": 4,
                "num_layers": 2,
                "head_dim": 16,
            },
            "head": {"type": "mse"},
        },
        "loss": {
            "gae_lambda": 0.95,
            "gae_discount": 0.99,
            "turn_lambda": 0.8,
            "turn_discount": 0.9,
            "pg_clip_high": 0.2,
            "pg_clip_low": 0.2,
        },
        "env": {"name": "arithmetic", "max_x": 10, "max_y": 10},
        "eval_envs": 1,
        "update_envs": 1,
        "max_seq_length": 128,
        "total_update_episodes": 1000,
        "checkpoint_every": 100,
    }

    config = load_config(json.dumps(config_json))
    model = DummyModel(nnx.Rngs(0))
    opt = make_optimizer(
        model,
        config.policy_optimizer,
        config.total_update_episodes,
        nnx.Param,
    )

    print("Test 1 (Full): Success")

    # 2. Test No Schedule, No Multi-step
    config_json["policy_optimizer"]["schedule"] = None
    config_json["policy_optimizer"]["multi_step"] = None
    config = load_config(json.dumps(config_json))
    opt = make_optimizer(
        model,
        config.policy_optimizer,
        config.total_update_episodes,
        nnx.Param,
    )

    print("Test 2 (Minimal): Success")


if __name__ == "__main__":
    test_config_variants()
