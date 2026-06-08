import numpy as np
from flax import nnx
from rich.console import Console
from vaml.agent.local import LocalAgent
from vaml.base_model_loader import load_base_model
from vaml.checkpointer import Checkpointer
from vaml.env.make import make_env
from vaml.episode_listener import (
    BufferedEpisodeListener,
    EpisodeSaver,
    Trainer,
)
from vaml.experiment import Experiment
from vaml.logger import create_logger
from vaml.model.value_network import ValueParam
from vaml.utils.optimizer import make_optimizer


def train_cli(
    config_url: str,
    value_net_id: str | None = None,
):
    experiment = Experiment.from_config_file(config_url)

    config = experiment.config
    console = Console()
    logger = create_logger(experiment, console)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)
    model.initialize_lora(config.lora, rngs=rngs)

    checkpointer = Checkpointer(experiment.checkpoints_url)

    eval_batch_size = config.eval_envs
    env = make_env(
        config.env.name,
        eval_batch_size,
        experiment.environments_seed,
        config.env,
    )

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    obs, metrics = env.reset(env_indices)

    assert config.policy_optimizer is not None
    policy_opt = make_optimizer(
        model,
        config.policy_optimizer,
        config.total_update_episodes,
        nnx.LoRAParam,
    )
    value_opt = make_optimizer(
        model, config.value_optimizer, config.total_update_episodes, ValueParam
    )

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        env.max_turns,
        rngs.agent(),
    )
    # agent.post_update() # just to merge the lora

    agent.set_episode_instructions(env.instructions())

    rollout_log_size = 100
    rollout_logger = BufferedEpisodeListener(
        rollout_log_size + config.update_envs,
        rollout_log_size,
        config.max_seq_length,
        env.max_turns,
        EpisodeSaver(experiment.rollout_dir),
    )

    trainer = Trainer(
        agent,
        policy_opt,
        value_opt,
        rngs.trainer(),
        checkpointer,
        logger,
        config,
        episode_listener=rollout_logger,
    )

    if value_net_id is not None:
        other_exp = Experiment.load(value_net_id)
        with Checkpointer(other_exp.checkpoints_url) as other_checkpointer:
            trainer.restore_checkpoint(checkpointer=other_checkpointer, wrt=ValueParam)

    trainer_listener = BufferedEpisodeListener(
        config.update_envs + config.eval_envs,
        config.update_envs,
        config.max_seq_length,
        env.max_turns,
        trainer,
    )

    agent.episode_listener = trainer_listener

    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    logger.start()

    while trainer.progress < 1.0:
        env_indices, actions = agent.act(env_indices, obs, rewards, dones, metrics)
        obs, rewards, dones, metrics = env.step(env_indices, actions)

    logger.close()
    checkpointer.close()
