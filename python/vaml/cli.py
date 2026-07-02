from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from vaml.build_offline import build_offline as build_offline_fn
from vaml.server.export_episode import export_episode as export_episode_fn
from vaml.train_rl import train_cli
from vaml.train_value import train_value_cli

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
    from vaml.play import play as play_fn

    play_fn(env, seed)


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
):
    train_cli(config_url, value_net_id, lora_init_id, lora_init_step)


@app.command()
def train_value(
    config_url: str,
    offline_data_url: str,
    track_values: Annotated[
        bool,
        typer.Option(
            "--track-values/--no-track-values",
            help="Track the first episode's value function each update and render it "
            "as an animation (extra forward pass per update; disable to save compute)",
        ),
    ] = True,
):
    train_value_cli(config_url, offline_data_url, track_values=track_values)


@app.command()
def build_offline(config_url: str, output_path: str, file_size: int, file_count: int):
    build_offline_fn(config_url, output_path, file_size, file_count)


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
        uv run vaml eval api --model openrouter/meta-llama/llama-3.3-8b-instruct:free --env arithmetic
        uv run vaml eval api --model openrouter/google/gemma-3-4b-it:free --env wordle --episodes 50
    """
    from vaml.eval import eval_api

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
        uv run vaml eval checkpoint winged-tortoise-of-glory
        uv run vaml eval checkpoint my-experiment --episodes 200 --step 1000
    """
    from vaml.eval import eval_checkpoint

    eval_checkpoint(
        experiment_name=experiment,
        num_episodes=num_episodes,
        checkpoint_step=step,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    app()
