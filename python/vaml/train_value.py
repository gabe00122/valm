from pathlib import Path

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from rich.console import Console
from vaml.base_model_loader import load_base_model
from vaml.buffer import UpdateBatch, UpdateBuffer
from vaml.checkpointer import Checkpointer
from vaml.episode_listener.buffered import BufferedEpisodeListener
from vaml.episode_listener.saver import EpisodeSaver
from vaml.experiment import Experiment
from vaml.logger import create_logger
from vaml.model.value_network import ValueParam
from vaml.update_step import multi_update_step_v2
from vaml.utils.optimizer import make_optimizer


@jax.jit(static_argnames=("model_def"))
def calculate_values(
    model_def, model_state, rng_key, context: jax.Array, rewards: jax.Array
):
    model = nnx.merge(model_def, model_state)
    positions = jnp.arange(context.shape[0])
    _, values_repr, _, rng_key = model(
        context[None, :],
        positions[None, :],
        rewards=rewards[None, :],
        rng_key=rng_key,
    )
    values = values_repr.value()
    return jnp.squeeze(values, 0), rng_key


def print_value_param_count(model):
    value_params = nnx.state(model, ValueParam)
    param_count = sum(x.size for x in jax.tree.leaves(value_params))
    print(f"Value Parameters: {param_count}")


def train_value_cli(config_url: str, offline_data_url: str, track_values: bool = False):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()
    logger = create_logger(experiment, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, _, _ = load_base_model(config.base_model, rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)
    print_value_param_count(model)

    data_dir = Path(offline_data_url)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(
            f"offline_data_url {offline_data_url} does not exist or is not a directory."
        )

    data_files = sorted(list(data_dir.glob("*.npz")))
    if not data_files:
        raise ValueError(f"No .npz files found in {offline_data_url}")

    first_batch = UpdateBatch.load_npz(data_files[0])
    num_episodes_per_file = first_batch.context.shape[0]

    first_episode_context = jnp.asarray(first_batch.context[0])
    first_episode_rewards = jnp.asarray(first_batch.rewards[0], dtype=jnp.bfloat16)
    update_envs = config.update_envs
    grad_accum = config.gradient_accumulations or 1
    micro_batch_size = update_envs // grad_accum

    buffer_size = update_envs + num_episodes_per_file

    max_turns = first_batch.turn_start_positions.shape[1]
    input_buffer = UpdateBuffer(
        buffer_size,
        update_envs,
        config.max_seq_length,
        max_turns,
    )
    output_buffer = BufferedEpisodeListener(
        buffer_size,
        num_episodes_per_file,
        config.max_seq_length,
        max_turns,
        EpisodeSaver(experiment.rollout_dir),
    )
    input_buffer.store(first_batch)

    total_updates = (len(data_files) * num_episodes_per_file) // config.update_envs

    value_opt = make_optimizer(
        model,
        config.value_optimizer,
        total_updates,
        config.gradient_accumulations,
        ValueParam,
    )
    value_opt_def, value_opt_state = nnx.split(value_opt)
    model_def, model_state = nnx.split(model)

    rng_key = rngs()
    # Separate rng stream so tracking values doesn't perturb the training rngs.
    value_rng_key = rngs()
    value_history: list[np.ndarray] = []

    step = 0
    input_file_idx = 1
    logger.start()
    while input_file_idx < len(data_files) or input_buffer.has_batch:
        # Load more data if buffer doesn't have a batch and there are more files
        if not input_buffer.has_batch and input_file_idx < len(data_files):
            batch_data = UpdateBatch.load_npz(data_files[input_file_idx])
            input_buffer.store(batch_data)
            input_file_idx += 1
            # Check again if we have enough after loading
            if not input_buffer.has_batch:
                continue

        if not input_buffer.has_batch:
            break

        batch = input_buffer.take_batch()
        _, value_opt_state, model_state, summery_metrics, token_metrics, rng_key = (
            multi_update_step_v2(
                None,
                None,
                value_opt_def,
                value_opt_state,
                model_def,
                model_state,
                rng_key,
                batch,
                config.loss,
                micro_batch_size,
                True,
            )
        )

        logger.log_dict(summery_metrics, step)
        step += 1

        if track_values:
            # Recalculate the first episode's value function with the updated model.
            # multi_update_step_v2 applies exactly one optimizer update per call, so
            # every iteration corresponds to a distinct set of model params.
            first_episode_values, value_rng_key = calculate_values(
                model_def,
                model_state,
                value_rng_key,
                first_episode_context,
                first_episode_rewards,
            )
            value_history.append(np.asarray(first_episode_values))

        output_batch = batch._replace(update_metrics=token_metrics)
        output_buffer.on_episodes(output_batch)

    logger.close()
    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        opt = nnx.merge(value_opt_def, value_opt_state)
        model = nnx.merge(model_def, model_state)
        checkpointer.save(
            {"value_opt": opt, "model": model},
            step,
            nnx.filterlib.Any(nnx.OptState, ValueParam),
        )

    if track_values:
        save_value_history(experiment, console, value_history)


def save_value_history(
    experiment: Experiment, console: Console, value_history: list[np.ndarray]
):
    """Save the per-update value history for the first episode and render it."""
    if not value_history:
        console.print("[yellow]No updates ran; skipping value history.[/yellow]")
        return

    values = np.stack(value_history, axis=0)
    values_path = f"{experiment.root}/first_episode_values.npy"
    with experiment.fs.open(values_path, "wb") as f:
        np.save(f, values)
    console.print(f"Saved first-episode value history {values.shape} to {values_path}")

    # Render the evolution of the value function as an animation.
    from vaml.visualization.render_values import render_values_video

    video_path = f"{experiment.root}/first_episode_values.mp4"
    try:
        render_values_video(values_path, video_path)
    except Exception as e:  # rendering needs ffmpeg/matplotlib; don't fail training
        console.print(f"[yellow]Could not render value animation: {e}[/yellow]")
        console.print(
            "Render manually with: "
            f"python -m vaml.visualization.render_values {values_path} -o {video_path}"
        )
