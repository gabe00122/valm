from llmrl.agent.local import EpisodeSaver
import numpy as np
from flax import nnx
from pathlib import Path

from llmrl.agent.local import (
    BufferedEpisodeListener,
    LocalAgent,
)
from llmrl.base_model_loader import load_base_model
from llmrl.env.make import make_env
from llmrl.experiement import Experiment
from llmrl.logger import create_logger
from llmrl.utils.performance import PerformanceTracker

from rich.console import Console

def _get_start(p: str):
    path = Path(p)
    files = path.glob("*.npz")

    max_num = 0
    for f in files:
        num = int(f.name[9:-4])
        max_num = max(max_num, num)

    return max_num


def build_offline(config_url: str, output_path: str, file_size: int, file_count: int):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()
    performance_tracker = PerformanceTracker()
    logger = create_logger(config, experiment.unique_token, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    model.initalize_value_net(config.value_net, rngs=rngs)

    eval_batch_size = config.eval_envs
    env = make_env(
        config.env.name, eval_batch_size, experiment.environments_seed, config.env
    )

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        logger,
        performance_tracker,
        rngs.agent(),
    )

    agent.set_episode_instructions(env.instructions())

    saver = EpisodeSaver(output_path)
    saver.chunk_num = _get_start(output_path)
    agent.episode_listener = BufferedEpisodeListener(file_size + config.eval_envs, file_size, config.max_seq_length, saver)

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    obs = env.reset(env_indices)

    while saver.chunk_num < file_count:
        console.print(f"Chunk {saver.chunk_num}")
        env_indices, actions = agent.act(env_indices, obs, rewards, dones)
        with performance_tracker.time("env_step"):
            obs, rewards, dones = env.step(env_indices, actions)
