import jax
import numpy as np
from flax import nnx

from llmrl.model.value_network import ValueParam
from llmrl.agent.local import (
    BufferedEpisodeListener,
    LocalAgent,
    Trainer,
)
from llmrl.base_model_loader import load_base_model
from llmrl.checkpointer import Checkpointer
from llmrl.env.make import make_env
from llmrl.experiement import Experiment
from llmrl.logger import create_logger, MetricsAccumulator
from llmrl.utils.performance import PerformanceTracker
from llmrl.utils.optimizer import make_optimizer

from rich.console import Console

def train_cli(
    config_url: str,
    value_net_id: str | None = None,
):
    experiment = Experiment.from_config_file(config_url)
    # jax.config.update("jax_log_compiles", True)
    # jax.config.update("jax_explain_cache_misses", True)

    config = experiment.config
    console = Console()
    performance_tracker = PerformanceTracker()
    logger = MetricsAccumulator(create_logger(config, experiment.unique_token, console))

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    model.initalize_value_net(config.value_net, rngs=rngs)
    model.initialize_lora(config.lora, rngs=rngs)

    checkpointer = Checkpointer(experiment.checkpoints_url)

    eval_batch_size = config.eval_envs
    env = make_env(
        config.env.name, eval_batch_size, experiment.environments_seed, config.env
    )

    policy_opt = make_optimizer(model, config.policy_optimizer, config.total_update_episodes, nnx.LoRAParam)
    value_opt = make_optimizer(model, config.value_optimizer, config.total_update_episodes, ValueParam)

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        logger,
        performance_tracker,
        rngs.agent(),
    )

    agent.set_episode_instructions(env.instructions())

    trainer = Trainer(
        agent,
        policy_opt,
        value_opt,
        rngs.trainer(),
        checkpointer,
        performance_tracker,
        logger,
        config,
    )

    if value_net_id is not None:
        other_exp = Experiment.load(value_net_id)
        with Checkpointer(other_exp.checkpoints_url) as other_checkpointer:
            trainer.restore_checkpoint(checkpointer=other_checkpointer, wrt=ValueParam)

    agent.episode_listener = BufferedEpisodeListener(
        config.update_envs + config.eval_envs,
        config.update_envs,
        config.max_seq_length,
        trainer,
    )

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    obs = env.reset(env_indices)
    last_progress = 0.0

    logger.start()

    while trainer.progress < 1.0:
        env_indices, actions = agent.act(env_indices, obs, rewards, dones)
        with performance_tracker.time("env_step"):
            obs, rewards, dones = env.step(env_indices, actions)

        if trainer.progress > last_progress:
            last_progress = trainer.progress
            logger.flush(trainer._update_step)

    logger.close()
    checkpointer.close()
