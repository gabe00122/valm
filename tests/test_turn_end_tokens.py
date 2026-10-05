from pathlib import Path

import pytest
from valm.engine import TurnEndTokens, get_turn_end_tokens
from valm.util import load_tokenizer

BASE_MODELS = Path(__file__).resolve().parent.parent / "base-models"


@pytest.mark.parametrize(
    "model_name, expected",
    [
        # <|im_end|>, "\n"
        ("Qwen/Qwen3-4B-Instruct-2507", TurnEndTokens(stop=151645, separator=198)),
        ("openbmb/MiniCPM5-1B", TurnEndTokens(stop=130073, separator=220)),
    ],
)
def test_turn_end_tokens_from_chat_template(model_name, expected):
    model_path = BASE_MODELS / model_name
    if not (model_path / "tokenizer_config.json").exists():
        pytest.skip(f"{model_name} not downloaded to {model_path}")

    assert get_turn_end_tokens(load_tokenizer(model_path)) == expected
