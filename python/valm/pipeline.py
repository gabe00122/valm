import subprocess
import sys
from pathlib import Path

import fsspec
from rich.console import Console
from valm.config import load_config
from valm.experiment import generate_unique_token


def _run_stage(console: Console, args: list[str]) -> None:
    console.print(f"[bold cyan]pipeline ▸[/bold cyan] {' '.join(args)}")
    subprocess.run([sys.executable, "-m", "valm.cli", *args], check=True)


def run_pipeline(
    config_url: str,
    offline_data_dir: str = "./offline_data",
    offline_file_size: int = 1000,
    offline_file_count: int = 20,
    offline_batch_size: int | None = None,
    base_dir: str = "results",
    value_warmup: bool = True,
    save_checkpoints: bool = True,
    save_rollouts: bool = True,
    track_values: bool = True,
    wandb_tags: list[str] | None = None,
) -> None:
    """Run the full training pipeline for one config.

    PPO: build offline data -> pretrain the value net -> train online.
    GRPO has no critic, so the offline and value stages are skipped, as they
    are for PPO with value_warmup=False (the critic then learns from scratch
    online).

    Stages run as subprocesses so each gets a fresh JAX process and device
    memory from one stage is fully released before the next starts.
    """
    with fsspec.open(config_url, "r") as f:
        config = load_config(f.read())

    console = Console()
    tag_args = [arg for tag in wandb_tags or [] for arg in ("--wandb-tag", tag)]
    rollout_args = [] if save_rollouts else ["--no-save-rollouts"]

    value_net_id = None
    if config.loss.type == "ppo" and value_warmup:
        existing_files = len(list(Path(offline_data_dir).glob("*.npz")))
        if existing_files >= offline_file_count:
            console.print(
                f"Found {existing_files} offline files in {offline_data_dir}; "
                "skipping build-offline."
            )
        else:
            batch_args = (
                ["--batch-size", str(offline_batch_size)]
                if offline_batch_size is not None
                else []
            )
            _run_stage(
                console,
                [
                    "build-offline",
                    config_url,
                    offline_data_dir,
                    str(offline_file_size),
                    str(offline_file_count),
                    *batch_args,
                ],
            )

        value_net_id = generate_unique_token()
        _run_stage(
            console,
            [
                "train-value",
                config_url,
                offline_data_dir,
                "--run-id",
                value_net_id,
                "--base-dir",
                base_dir,
                "--track-values" if track_values else "--no-track-values",
                *rollout_args,
                *tag_args,
            ],
        )

    value_args = ["--value-net-id", value_net_id] if value_net_id is not None else []
    checkpoint_args = [] if save_checkpoints else ["--no-save-checkpoints"]
    _run_stage(
        console,
        [
            "train",
            config_url,
            *value_args,
            "--base-dir",
            base_dir,
            *checkpoint_args,
            *rollout_args,
            *tag_args,
        ],
    )
