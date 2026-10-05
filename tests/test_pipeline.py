import json
from pathlib import Path

import pytest
import valm.pipeline as pipeline
from valm.logger import JsonLogger

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _write_config(tmp_path: Path, loss: dict) -> str:
    config = json.loads((CONFIGS / "grpo.json").read_text())
    config["loss"] = loss
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(path)


PPO_LOSS = {
    "type": "ppo",
    "gae_lambda": 0.95,
    "gae_discount": 1.0,
    "turn_lambda": 0.97,
    "turn_discount": 0.97,
    "pg_clip_high": 0.28,
    "pg_clip_low": 0.2,
}
GRPO_LOSS = {"type": "grpo", "pg_clip_high": 0.28, "pg_clip_low": 0.2}


@pytest.fixture
def stages(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline, "_run_stage", lambda _console, args: calls.append(args)
    )
    return calls


def _flag(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


def test_run_id_names_ppo_stages(tmp_path, stages):
    config = _write_config(tmp_path, PPO_LOSS)
    pipeline.run_pipeline(
        config, offline_data_dir=str(tmp_path / "offline"), run_id="my-run"
    )

    assert [args[0] for args in stages] == ["build-offline", "train-value", "train"]
    _, value, train = stages
    assert _flag(value, "--run-id") == "my-run-value"
    assert _flag(train, "--run-id") == "my-run"
    assert _flag(train, "--value-net-id") == "my-run-value"


def test_run_id_names_grpo_train_stage(tmp_path, stages):
    config = _write_config(tmp_path, GRPO_LOSS)
    pipeline.run_pipeline(config, run_id="my-run")

    assert [args[0] for args in stages] == ["train"]
    assert _flag(stages[0], "--run-id") == "my-run"


def test_without_run_id_names_are_generated(tmp_path, stages):
    config = _write_config(tmp_path, PPO_LOSS)
    pipeline.run_pipeline(config, offline_data_dir=str(tmp_path / "offline"))

    _, value, train = stages
    assert _flag(value, "--run-id") is not None
    assert _flag(train, "--run-id") is None
    assert _flag(train, "--value-net-id") == _flag(value, "--run-id")


def test_json_logger_writes_step(tmp_path):
    logger = JsonLogger(str(tmp_path))
    logger.start()
    logger.log_dict({"rewards": {"mean": 0.5}, "entropy": 1.0}, step=3)
    logger.close()

    row = json.loads((tmp_path / "logs.jsonl").read_text())
    assert row == {"step": 3, "rewards.mean": 0.5, "entropy": 1.0}
