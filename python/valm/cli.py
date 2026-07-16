from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from valm.build_offline import build_offline as build_offline_fn
from valm.pipeline import run_pipeline
from valm.server.export_episode import export_episode as export_episode_fn
from valm.train_rl import train_cli
from valm.train_value import train_value_cli

app = typer.Typer(pretty_exceptions_show_locals=False)
eval_app = typer.Typer(help="Evaluate agents against environments")
app.add_typer(eval_app, name="eval")


@app.command()
def play(
    env: Annotated[
        str,
        typer.Argument(
            help="Environment name (arithmetic, wordle, countdown, spatial)"
        ),
    ],
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
):
    """Play an environment interactively from the terminal."""
    from valm.play import play as play_fn

    play_fn(env, seed)


RunIdOption = Annotated[
    Optional[str],
    typer.Option("--run-id", help="Experiment token for this run (default: generated)"),
]
BaseDirOption = Annotated[
    str, typer.Option("--base-dir", help="Base directory for experiment output")
]
SaveRolloutsOption = Annotated[
    bool,
    typer.Option(
        "--save-rollouts/--no-save-rollouts",
        help="Save every episode to the run's rollouts directory",
    ),
]
SaveCheckpointsOption = Annotated[
    bool,
    typer.Option(
        "--save-checkpoints/--no-save-checkpoints",
        help="Save periodic checkpoints (a final checkpoint is always written)",
    ),
]
WandbTagOption = Annotated[
    Optional[list[str]],
    typer.Option("--wandb-tag", help="Tag added to the wandb run (repeatable)"),
]
TrackValuesOption = Annotated[
    bool,
    typer.Option(
        "--track-values/--no-track-values",
        help="Track the first episode's value function each update and render it "
        "as an animation (extra forward pass per update; disable to save compute)",
    ),
]


@app.command()
def train(
    config_url: str,
    value_net_id: Annotated[
        Optional[str],
        typer.Option("--value-net-id", help="Experiment token to start training from"),
    ] = None,
    lora_init_id: Annotated[
        Optional[str],
        typer.Option(
            "--lora-init-id",
            help="Experiment token to warm-start LoRA parameters from (LoRA only)",
        ),
    ] = None,
    lora_init_step: Annotated[
        Optional[int],
        typer.Option(
            "--lora-init-step",
            help="Checkpoint step to load LoRA parameters from (default: latest)",
        ),
    ] = None,
    run_id: RunIdOption = None,
    base_dir: BaseDirOption = "results",
    save_checkpoints: SaveCheckpointsOption = True,
    save_rollouts: SaveRolloutsOption = True,
    wandb_tag: WandbTagOption = None,
):
    train_cli(
        config_url,
        value_net_id,
        lora_init_id=lora_init_id,
        lora_init_step=lora_init_step,
        run_id=run_id,
        base_dir=base_dir,
        save_checkpoints=save_checkpoints,
        save_rollouts=save_rollouts,
        wandb_tags=wandb_tag,
    )


@app.command()
def train_value(
    config_url: str,
    offline_data_url: str,
    track_values: TrackValuesOption = True,
    run_id: RunIdOption = None,
    base_dir: BaseDirOption = "results",
    save_rollouts: SaveRolloutsOption = True,
    wandb_tag: WandbTagOption = None,
):
    train_value_cli(
        config_url,
        offline_data_url,
        track_values=track_values,
        run_id=run_id,
        base_dir=base_dir,
        save_rollouts=save_rollouts,
        wandb_tags=wandb_tag,
    )


@app.command()
def build_offline(
    config_url: str,
    output_path: str,
    file_size: int,
    file_count: int,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--batch-size",
            help="Env batch size for data generation (default: config eval_envs)",
        ),
    ] = None,
):
    build_offline_fn(config_url, output_path, file_size, file_count, batch_size)


@app.command()
def pipeline(
    config_url: str,
    offline_data: Annotated[
        str,
        typer.Option(
            "--offline-data",
            help="Directory for offline episode data (reused if already populated)",
        ),
    ] = "./offline_data",
    offline_file_size: Annotated[
        int, typer.Option("--offline-file-size", help="Episodes per offline data file")
    ] = 1000,
    offline_file_count: Annotated[
        int, typer.Option("--offline-file-count", help="Number of offline data files")
    ] = 20,
    offline_batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--offline-batch-size",
            help="Env batch size for data generation (default: config eval_envs)",
        ),
    ] = None,
    base_dir: BaseDirOption = "results",
    value_warmup: Annotated[
        bool,
        typer.Option(
            "--value-warmup/--no-value-warmup",
            help="Pretrain the value net on offline data before RL (PPO only; "
            "--no-value-warmup trains the critic from scratch online)",
        ),
    ] = True,
    save_checkpoints: SaveCheckpointsOption = True,
    save_rollouts: SaveRolloutsOption = True,
    track_values: TrackValuesOption = True,
    wandb_tag: WandbTagOption = None,
):
    """Run the full pipeline for one config: offline data -> value net -> RL.

    GRPO configs skip straight to RL (no critic to pretrain). Each stage runs
    as a subprocess so JAX device memory is released between stages.
    """
    run_pipeline(
        config_url,
        offline_data_dir=offline_data,
        offline_file_size=offline_file_size,
        offline_file_count=offline_file_count,
        offline_batch_size=offline_batch_size,
        base_dir=base_dir,
        value_warmup=value_warmup,
        save_checkpoints=save_checkpoints,
        save_rollouts=save_rollouts,
        track_values=track_values,
        wandb_tags=wandb_tag,
    )


@app.command()
def export_episode(
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Destination JSON file")
    ],
    run: Annotated[
        str,
        typer.Option("--run", help="Run name under results/<run>/rollouts"),
    ],
    episode_id: Annotated[
        int,
        typer.Option(
            "--episode-id",
            "-e",
            help="Global episode id",
        ),
    ] = 0,
) -> None:
    """Export one episode from a run to a static JSON file."""
    try:
        payload = export_episode_fn(
            out,
            run=run,
            episode_id=episode_id,
        )
    except (ValueError, FileNotFoundError, IndexError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Wrote {len(payload['tokens'])} tokens to {out}")


@eval_app.command("api")
def eval_api_cmd(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Model identifier (e.g., 'openrouter/meta-llama/llama-3.3-8b-instruct:free')",
        ),
    ] = "blank",
    env: Annotated[
        Literal["arithmetic", "wordle"],
        typer.Option("--env", "-e", help="Environment name"),
    ] = "arithmetic",
    num_envs: Annotated[
        int,
        typer.Option("--num-envs", "-n", help="Number of parallel environments"),
    ] = 4,
    num_episodes: Annotated[
        int, typer.Option("--episodes", "-ep", help="Number of episodes to run")
    ] = 100,
    base_url: Annotated[
        str, typer.Option("--base-url", help="API base URL")
    ] = "http://localhost:8080",
    api_key: Annotated[str, typer.Option("--api-key", help="API key")] = "no-key",
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Environment random seed")
    ] = 42,
    reasoning: Annotated[
        bool,
        typer.Option(
            "--reasoning/--no-reasoning", help="Enable reasoning in API requests"
        ),
    ] = True,
):
    """
    Evaluate an api based model against an environment.

    Examples:
        uv run valm eval api --model openrouter/meta-llama/llama-3.3-8b-instruct:free --env arithmetic
        uv run valm eval api --model openrouter/google/gemma-3-4b-it:free --env wordle --episodes 50
    """
    from valm.eval import eval_api

    eval_api(
        model=model,
        env_name=env,
        num_envs=num_envs,
        num_episodes=num_episodes,
        base_url=base_url,
        api_key=api_key,
        env_seed=seed,
        reasoning_enabled=reasoning,
    )


@eval_app.command("checkpoint")
def eval_checkpoint_cmd(
    experiment: Annotated[
        str,
        typer.Argument(help="Experiment name (e.g., 'winged-tortoise-of-glory')"),
    ],
    num_episodes: Annotated[
        int, typer.Option("--episodes", "-ep", help="Number of episodes to run")
    ] = 100,
    step: Annotated[
        Optional[int],
        typer.Option("--step", "-s", help="Checkpoint step (default: latest)"),
    ] = None,
    base_dir: Annotated[
        str,
        typer.Option("--base-dir", "-d", help="Base directory for experiments"),
    ] = "results",
):
    """
    Evaluate a trained model checkpoint against its configured environment.

    Examples:
        uv run valm eval checkpoint winged-tortoise-of-glory
        uv run valm eval checkpoint my-experiment --episodes 200 --step 1000
    """
    from valm.eval import eval_checkpoint

    eval_checkpoint(
        experiment_name=experiment,
        num_episodes=num_episodes,
        checkpoint_step=step,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    app()
