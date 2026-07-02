import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Environment Config
class ArithmeticEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["arithmetic"] = "arithmetic"
    max_x: int
    max_y: int


class WordleEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Literal["wordle"] = "wordle"
    max_guesses: int


# Base Model Config
class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    embed: int
    q_heads: int
    kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int = -1
    mlp_ffw_size: int = -1
    norm_eps: float = 1e-6
    rope_theta: float = 500000.0


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: float
    top_k: int
    top_p: float


# experiment config
class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mlp: bool = False
    attn: bool = False
    rank: int = 0


class MseCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["mse"] = "mse"


class HlGaussConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["hl_gauss"] = "hl_gauss"

    min: float
    max: float
    n_logits: int
    sigma: float


class ValueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    latent_encoder_rank: int
    backbone: LLMConfig
    last_latent_only: bool = False
    head: HlGaussConfig | MseCriticConfig = Field(discriminator="type")


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_name: str = "vaml"
    use_tb: bool = False
    use_console: bool = True
    use_wandb: bool = False
    use_jsonl: bool = True


class SGDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["sgd"] = "sgd"
    lr: float
    momentum: float | None = None
    nesterov: bool = False


class AdamWConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["adamw"] = "adamw"
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    eps: float = 1e-8


class WarmupCosineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["warmup_cosine"] = "warmup_cosine"
    warmup_ratio: float = 0.1


ScheduleConfig = WarmupCosineConfig | None


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    opt: AdamWConfig | SGDConfig = Field(discriminator="type")
    max_grad_norm: float | None = None
    schedule: ScheduleConfig = None


class GRPOLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["grpo"] = "grpo"
    pg_clip_high: float
    pg_clip_low: float
    entropy_coef: float | None = None


class PPOLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["ppo"] = "ppo"
    gae_lambda: float
    gae_discount: float
    turn_lambda: float
    turn_discount: float
    pg_clip_high: float
    pg_clip_low: float
    entropy_coef: float | None = None
    is_correction: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int | Literal["random"] = "random"
    base_model: str
    lora: LoraConfig
    value_net: ValueConfig
    logger: LoggerConfig
    policy_optimizer: OptimizerConfig | None = None
    value_optimizer: OptimizerConfig
    loss: PPOLossConfig | GRPOLossConfig = Field(discriminator="type")
    env: ArithmeticEnvConfig | WordleEnvConfig = Field(discriminator="name")

    gradient_accumulations: int | None = None
    eval_envs: int
    update_envs: int
    max_seq_length: int
    total_update_episodes: int
    checkpoint_every: int

    # GRPO group size. 1 means no grouping (each episode is its own group).
    group_size: int = 1

    @model_validator(mode="after")
    def _check_group_size(self):
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.eval_envs % self.group_size != 0:
            raise ValueError("eval_envs must be a multiple of group_size")
        if self.update_envs % self.group_size != 0:
            raise ValueError("update_envs must be a multiple of group_size")
        return self


def load_config(json_config: str) -> Config:
    config = Config.model_validate_json(json_config, strict=True)
    if config.seed == "random":
        config = config.model_copy(update={"seed": random.getrandbits(31)})

    return config
