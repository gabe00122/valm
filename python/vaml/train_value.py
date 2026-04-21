from pathlib import Path

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from rich.console import Console
from vaml.base_model_loader import load_base_model
from vaml.buffer import UpdateBatch, UpdateBuffer
from vaml.checkpointer import Checkpointer
from vaml.experiment import Experiment
from vaml.logger import create_logger
from vaml.model.value_network import ValueParam
from vaml.update_step import update_step
from vaml.utils.optimizer import make_optimizer


@jax.jit(static_argnames=('model_def'))
def calculate_values(model_def, model_state, rng_key, context: jax.Array):
    model = nnx.merge(model_def, model_state)
    positions = jnp.arange(context.shape[0])
    _, values_repr, _, rng_key = model(context[None, :], positions[None, :], None, rng_key=rng_key)
    values = values_repr.value()
    return jnp.squeeze(values, 0), rng_key


def print_value_param_count(model):
    value_params = nnx.state(model, ValueParam)
    param_count = sum(x.size for x in jax.tree.leaves(value_params))
    print(f"Value Parameters: {param_count}")


def train_value_cli(config_url: str, offline_data_url: str):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()
    logger = create_logger(experiment, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)
    print_value_param_count(model)

    data_dir = Path(offline_data_url)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"offline_data_url {offline_data_url} does not exist or is not a directory.")

    data_files = sorted(list(data_dir.glob("*.npz")))
    if not data_files:
        raise ValueError(f"No .npz files found in {offline_data_url}")

    first_batch = UpdateBatch.load_npz(data_files[0])
    num_episodes_per_file = first_batch.context.shape[0]
    buffer_size = config.update_envs + num_episodes_per_file

    buffer = UpdateBuffer(buffer_size, config.update_envs, config.max_seq_length)
    buffer.store(first_batch)

    total_updates = (len(data_files) * num_episodes_per_file) // config.update_envs
    value_opt = make_optimizer(model, config.value_optimizer, total_updates, ValueParam)
    value_opt_def, value_opt_state = nnx.split(value_opt)
    model_def, model_state = nnx.split(model)

    ref_context = first_batch.context[0]
    output_values = np.zeros((total_updates, config.max_seq_length))

    rng_key = rngs()

    step = 0
    file_idx = 1
    while file_idx < len(data_files) or buffer.has_batch:
        # Load more data if buffer doesn't have a batch and there are more files
        if not buffer.has_batch and file_idx < len(data_files):
            batch_data = UpdateBatch.load_npz(data_files[file_idx])
            buffer.store(batch_data)
            file_idx += 1
            # Check again if we have enough after loading
            if not buffer.has_batch:
                continue

        if not buffer.has_batch:
            break

        batch = buffer.take_batch()
        _, value_opt_state, model_state, metrics, rng_key = update_step(
            None,
            None,
            value_opt_def,
            value_opt_state,
            model_def,
            model_state,
            rng_key,
            batch,
            config.loss,
            True,
        )
        values, rng_key = calculate_values(model_def, model_state, rng_key, ref_context)
        output_values[step] = np.array(values)

        metrics["reward"] = batch.rewards.sum(axis=1).mean()
        logger.log_dict(metrics, step)
        step += 1

    np.save("./values", output_values)

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        opt = nnx.merge(value_opt_def, value_opt_state)
        model = nnx.merge(model_def, model_state)
        checkpointer.save({"value_opt": opt, "model": model}, step, nnx.filterlib.Any(nnx.OptState, ValueParam))

    logger.close()
