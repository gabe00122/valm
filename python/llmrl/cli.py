from typing import Annotated, Optional

import typer
from llmrl.buffer import UpdateBatch
from llmrl.build_offline import build_offline as build_offline_fn
from llmrl.train_rl import train_cli
from llmrl.train_value import train_value_cli
from llmrl.util import load_tokenizer
from llmrl.utils.episode_to_jsonl import episode_to_jsonl as episode_to_jsonl_fn

app = typer.Typer(pretty_exceptions_show_locals=False)
eval_app = typer.Typer(help="Evaluate agents against environments")
app.add_typer(eval_app, name="eval")


@app.command()
def play(
    env: Annotated[str, typer.Argument(help="Environment name (arithmetic, wordle, countdown, spatial)")],
    seed: Annotated[int, typer.Option("--seed", "-s", help="Random seed")] = 42,
):
    """Play an environment interactively from the terminal."""
    from llmrl.play import play as play_fn
    play_fn(env, seed)


@app.command()
def train(
    config_url: str,
    value_net_id: Annotated[Optional[str], typer.Option("--value-net-id", help="Experiment token to start training from")] = None,
):
    train_cli(config_url, value_net_id)


@app.command()
def train_value(
    config_url: str,
    offline_data_url: str,
):
    train_value_cli(config_url, offline_data_url)

@app.command()
def build_offline(config_url: str, output_path: str, file_size: int, file_count: int):
    build_offline_fn(config_url, output_path, file_size, file_count)

@app.command()
def episode_to_jsonl():
    tokenizer = load_tokenizer("./base-models/Qwen/Qwen3-4B-Instruct-2507")
    episode_to_jsonl_fn(
        "./episode_viewer/episodes.jsonl",
        UpdateBatch.load_npz("./episode_viewer/episodes.npz"),
        tokenizer,
    )


@eval_app.command("api")
def eval_api_cmd(
    model: Annotated[
        str,
        typer.Argument(
            help="Model identifier (e.g., 'openrouter/meta-llama/llama-3.3-8b-instruct:free')"
        ),
    ],
    env: Annotated[
        str, typer.Option("--env", "-e", help="Environment name")
    ] = "arithmetic",
    num_envs: Annotated[
        int, typer.Option("--num-envs", "-n", help="Number of parallel environments")
    ] = 4,
    num_episodes: Annotated[
        int, typer.Option("--episodes", "-ep", help="Number of episodes to run")
    ] = 100,
    base_url: Annotated[
        Optional[str], typer.Option("--base-url", help="Custom API base URL")
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="Environment random seed")
    ] = 42,
    # Arithmetic env options
    max_x: Annotated[
        int, typer.Option("--max-x", help="Max X value for arithmetic env")
    ] = 100,
    max_y: Annotated[
        int, typer.Option("--max-y", help="Max Y value for arithmetic env")
    ] = 100,
    # Wordle env options
    max_guesses: Annotated[
        int, typer.Option("--max-guesses", help="Max guesses for wordle env")
    ] = 6,
):
    """
    Evaluate an OpenRouter/LiteLLM model against an environment.

    Examples:
        uv run llmrl eval openrouter openrouter/meta-llama/llama-3.3-8b-instruct:free --env arithmetic
        uv run llmrl eval openrouter openrouter/google/gemma-3-4b-it:free --env wordle --episodes 50
    """
    from llmrl.eval import eval_api

    if env not in ("arithmetic", "wordle"):
        raise typer.BadParameter(
            f"Unknown environment: {env}. Must be 'arithmetic' or 'wordle'."
        )

    eval_api(
        model=model,
        env_name=env,  # type: ignore[arg-type]
        num_envs=num_envs,
        num_episodes=num_episodes,
        base_url=base_url,
        env_seed=seed,
        arithmetic_max_x=max_x,
        arithmetic_max_y=max_y,
        wordle_max_guesses=max_guesses,
    )


@eval_app.command("checkpoint")
def eval_checkpoint_cmd(
    experiment: Annotated[
        str, typer.Argument(help="Experiment name (e.g., 'winged-tortoise-of-glory')")
    ],
    num_episodes: Annotated[
        int, typer.Option("--episodes", "-ep", help="Number of episodes to run")
    ] = 100,
    step: Annotated[
        Optional[int],
        typer.Option("--step", "-s", help="Checkpoint step (default: latest)"),
    ] = None,
    base_dir: Annotated[
        str, typer.Option("--base-dir", "-d", help="Base directory for experiments")
    ] = "results",
):
    """
    Evaluate a trained model checkpoint against its configured environment.

    Examples:
        uv run llmrl eval checkpoint winged-tortoise-of-glory
        uv run llmrl eval checkpoint my-experiment --episodes 200 --step 1000
    """
    from llmrl.eval import eval_checkpoint

    eval_checkpoint(
        experiment_name=experiment,
        num_episodes=num_episodes,
        checkpoint_step=step,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    app()
