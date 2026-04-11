from dataclasses import dataclass
from typing import Literal

import numpy as np
from flax import nnx
from vaml.agent.base import Agent
from vaml.agent.lite import LiteAgent
from vaml.agent.local import LocalAgent
from vaml.base_model_loader import load_base_model
from vaml.checkpointer import Checkpointer
from vaml.env.base import Env
from vaml.env.make import make_env
from vaml.experiment import Experiment
from vaml.logger import ConsoleLogger
from vaml.model.value_network import ValueParam
from vaml.utils.performance import PerformanceTracker
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)


@dataclass
class EvalResult:
    """Results from an evaluation run."""

    total_episodes: int
    total_reward: float
    mean_reward: float
    min_reward: float
    max_reward: float
    std_reward: float


def _run_eval_loop(
    agent: Agent,
    env: Env,
    num_envs: int,
    num_episodes: int,
    console: Console,
) -> EvalResult:
    """
    Run the evaluation loop for a given agent and environment.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate on
        num_envs: Number of parallel environments
        num_episodes: Total number of episodes to run
        console: Rich console for output

    Returns:
        EvalResult with statistics from the evaluation
    """
    env_indices = np.arange(num_envs, dtype=np.int32)
    rewards = np.zeros((num_envs,), dtype=np.float32)
    dones = np.zeros((num_envs,), dtype=np.bool_)

    obs = env.reset(env_indices)

    episode_rewards: list[float] = []
    current_episode_rewards = np.zeros((num_envs,), dtype=np.float32)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=num_episodes)

        while len(episode_rewards) < num_episodes:
            env_indices, actions = agent.act(env_indices, obs, rewards, dones)
            obs, rewards, dones = env.step(env_indices, actions)

            # Accumulate rewards for current episodes
            current_episode_rewards[env_indices] += rewards

            # Record completed episodes
            done_indices = env_indices[dones]
            for idx in done_indices:
                if len(episode_rewards) < num_episodes:
                    episode_rewards.append(current_episode_rewards[idx])
                    current_episode_rewards[idx] = 0.0

            if dones.any():
                console.print(f"Episode rewards: {np.array(episode_rewards).mean()}")
                progress.update(task, completed=len(episode_rewards))

    agent.close()

    episode_rewards_arr = np.array(episode_rewards)
    return EvalResult(
        total_episodes=len(episode_rewards),
        total_reward=float(episode_rewards_arr.sum()),
        mean_reward=float(episode_rewards_arr.mean()),
        min_reward=float(episode_rewards_arr.min()),
        max_reward=float(episode_rewards_arr.max()),
        std_reward=float(episode_rewards_arr.std()),
    )


def eval_api(
    model: str,
    env_name: Literal["arithmetic", "wordle"],
    num_envs: int = 4,
    num_episodes: int = 100,
    base_url: str | None = None,
    env_seed: int = 42,
    # Environment-specific settings
    arithmetic_max_x: int = 100,
    arithmetic_max_y: int = 100,
    wordle_max_guesses: int = 6,
    wordle_words: list[str] | None = None,
) -> EvalResult:
    """
    Evaluate an OpenRouter/LiteLLM model against an environment.

    Args:
        model: Model identifier (e.g., "openrouter/meta-llama/llama-3.3-8b-instruct:free")
        env_name: Environment to evaluate on ("arithmetic" or "wordle")
        num_envs: Number of parallel environments
        num_episodes: Total number of episodes to run
        base_url: Optional custom base URL for the API
        env_seed: Random seed for the environment
        arithmetic_max_x: Max X value for arithmetic env
        arithmetic_max_y: Max Y value for arithmetic env
        wordle_max_guesses: Max guesses for wordle env
        wordle_words: Word list for wordle env

    Returns:
        EvalResult with evaluation statistics
    """
    console = Console()
    env = make_env(env_name, num_envs, env_seed, None)

    agent = LiteAgent(
        model=model,
        agent_count=num_envs,
        base_url=base_url,
    )
    agent.set_episode_instructions(env.instructions())

    console.print(f"[bold]Evaluating model:[/bold] {model}")
    console.print(f"[bold]Environment:[/bold] {env_name}")
    console.print(f"[bold]Episodes:[/bold] {num_episodes}")
    console.print()

    result = _run_eval_loop(agent, env, num_envs, num_episodes, console)

    console.print()
    console.print("[bold green]Evaluation Complete[/bold green]")
    console.print(f"  Total Episodes: {result.total_episodes}")
    console.print(f"  Mean Reward: {result.mean_reward:.4f}")
    console.print(f"  Std Reward: {result.std_reward:.4f}")
    console.print(f"  Min Reward: {result.min_reward:.4f}")
    console.print(f"  Max Reward: {result.max_reward:.4f}")

    return result


def eval_checkpoint(
    experiment_name: str,
    num_episodes: int = 100,
    checkpoint_step: int | None = None,
    base_dir: str = "results",
) -> EvalResult:
    """
    Evaluate a trained model checkpoint against its configured environment.

    Args:
        experiment_name: Name of the experiment (e.g., "winged-tortoise-of-glory")
        num_episodes: Total number of episodes to run
        checkpoint_step: Specific checkpoint step to load (None for latest)
        base_dir: Base directory for experiments

    Returns:
        EvalResult with evaluation statistics
    """
    console = Console()

    # Load experiment
    experiment = Experiment.load(experiment_name, base_dir)
    config = experiment.config

    console.print(f"[bold]Evaluating experiment:[/bold] {experiment_name}")
    console.print(f"[bold]Environment:[/bold] {config.env.name}")
    console.print(f"[bold]Base Model:[/bold] {config.base_model}")
    console.print(f"[bold]Episodes:[/bold] {num_episodes}")
    console.print()

    # Load model
    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, _ = load_base_model(config.base_model, rngs)
    model.initialize_lora(config.lora, rngs=rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)

    # Load checkpoint
    checkpointer = Checkpointer(experiment.checkpoints_url)
    if checkpoint_step is not None:
        checkpointer.restore(
            {"model": model},
            checkpoint_step,
            nnx.Any(nnx.LoRAParam, ValueParam),
        )
        console.print(f"[bold]Checkpoint step:[/bold] {checkpoint_step}")
    else:
        checkpointer.restore_latest(
            {"model": model},
            nnx.Any(nnx.LoRAParam, ValueParam),
        )
        console.print(
            f"[bold]Checkpoint step:[/bold] latest ({checkpointer.mngr.latest_step()})"
        )
    console.print()

    # Create environment
    num_envs = config.eval_envs
    env = make_env(
        config.env.name,
        num_envs,
        experiment.environments_seed,
        config.env,
    )

    # Create agent
    performance_tracker = PerformanceTracker()
    logger = ConsoleLogger(experiment_name, console)

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        logger,
        performance_tracker,
        rngs.agent(),
    )
    agent.set_episode_instructions(env.instructions())

    result = _run_eval_loop(agent, env, num_envs, num_episodes, console)

    checkpointer.close()

    console.print()
    console.print("[bold green]Evaluation Complete[/bold green]")
    console.print(f"  Total Episodes: {result.total_episodes}")
    console.print(f"  Mean Reward: {result.mean_reward:.4f}")
    console.print(f"  Std Reward: {result.std_reward:.4f}")
    console.print(f"  Min Reward: {result.min_reward:.4f}")
    console.print(f"  Max Reward: {result.max_reward:.4f}")

    return result
