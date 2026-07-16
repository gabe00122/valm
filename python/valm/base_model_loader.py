import json
import os
from pathlib import Path
from typing import Any

from flax import nnx
from rich.progress import track
from safetensors import safe_open
from transformers import TokenizersBackend
from valm.config import LLMConfig, SamplingConfig
from valm.model.qwen3 import Qwen3
from valm.util import load_tokenizer


def parse_hf_llm_config(hf_config: Any | dict[str, Any]) -> LLMConfig:
    def _get(x, k, default=None):
        return (
            getattr(x, k, default)
            if not isinstance(hf_config, dict)
            else hf_config.get(k, default)
        )

    return LLMConfig(
        embed=_get(hf_config, "hidden_size"),
        mlp_ffw_size=_get(hf_config, "intermediate_size", -1),
        q_heads=_get(hf_config, "num_attention_heads"),
        kv_heads=_get(hf_config, "num_key_value_heads"),
        num_layers=_get(hf_config, "num_hidden_layers"),
        head_dim=_get(hf_config, "head_dim"),
        vocab_size=_get(hf_config, "vocab_size"),
        norm_eps=_get(hf_config, "rms_norm_eps"),
        rope_theta=_get(hf_config, "rope_theta"),
    )


def load_hf_llm_config(config_path: str | os.PathLike[str] | Path) -> "LLMConfig":
    return parse_hf_llm_config(json.loads(Path(config_path).read_text()))


def load_sampling_config(config_path: str | os.PathLike[str] | Path) -> SamplingConfig:
    with open(config_path, "r") as f:
        data: dict[str, Any] = json.load(f)

    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 20)
    top_p = data.get("top_p", 1.0)

    return SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p)


def _put_path(data: dict, path: list[str], value) -> None:
    """Insert `value` into a nested dict following `path` segments."""
    head, *tail = path
    if not tail:
        data[head] = value
        return
    child = data.setdefault(head, {})
    _put_path(child, tail, value)


def load_param_dict(params: dict[str, object], file_path: Path):
    """Load a safetensors checkpoint into a nested python dict."""
    with safe_open(file_path, framework="np") as f:
        for key in track(f.keys(), description="Loading weights"):
            key_path = key.split(".")
            value = f.get_tensor(key)
            _put_path(params, key_path, value)


def load_safetensors(file_path: str):
    params: dict[str, object] = {}

    files = list(Path(file_path).glob("**/model*safetensors"))
    for file in files:
        load_param_dict(params, file)

    return params


def load_base_model(
    model_name: str, rngs: nnx.Rngs
) -> tuple[Qwen3, TokenizersBackend, SamplingConfig]:
    model_path = f"base-models/{model_name}"
    config = load_hf_llm_config(f"{model_path}/config.json")
    params = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)
    sampling = load_sampling_config(f"{model_path}/generation_config.json")

    model = Qwen3(config, rngs=rngs)
    model.load_params(params)

    return model, tokenizer, sampling
