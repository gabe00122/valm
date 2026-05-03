from pathlib import Path

import numpy as np
from flax import nnx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from vaml.agent.local import LocalAgent
from vaml.base_model_loader import load_base_model
from vaml.env.make import make_env
from vaml.episode_listener import BufferedEpisodeListener, EpisodeSaver
from vaml.experiment import Experiment


def _get_start(p: str):
    path = Path(p)
    files = path.glob("*.npz")

    max_num = 0
    for f in files:
        num = int(f.name[9:-4])
        max_num = max(max_num, num)

    return max_num


def build_offline(
    config_url: str, output_path: str, file_size: int, file_count: int
):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)

    eval_batch_size = config.eval_envs
    env = make_env(
        config.env.name,
        eval_batch_size,
        experiment.environments_seed,
        config.env,
    )

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    obs, metrics = env.reset(env_indices)
    metric_names = list(metrics.keys())

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        env.max_turns,
        metric_names,
        rngs.agent(),
    )

    agent.set_episode_instructions(env.instructions())

    saver = EpisodeSaver(output_path)
    saver.chunk_num = _get_start(output_path)
    buffered_listener = BufferedEpisodeListener(
        file_size + config.eval_envs,
        file_size,
        config.max_seq_length,
        env.max_turns,
        metric_names,
        saver,
    )
    agent.episode_listener = buffered_listener

    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        chunks_task = progress.add_task(
            "Chunks     ", total=file_count, completed=saver.chunk_num
        )
        chunk_task = progress.add_task("Current    ", total=file_size)

        while saver.chunk_num < file_count:
            env_indices, actions = agent.act(
                env_indices, obs, rewards, dones, metrics
            )
            obs, rewards, dones, metrics = env.step(env_indices, actions)

            progress.update(chunks_task, completed=saver.chunk_num)
            progress.update(chunk_task, completed=buffered_listener.size)
