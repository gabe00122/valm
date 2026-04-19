import numpy as np
from flax import nnx
from vaml.agent.local import LocalAgent
from vaml.base_model_loader import load_base_model
from vaml.checkpointer import Checkpointer
from vaml.episode_listener import BufferedEpisodeListener, Trainer, EpisodeSaver, MultiEpisodeListener
from vaml.env.make import make_env
from vaml.experiment import Experiment
from vaml.logger import create_logger
from vaml.model.value_network import ValueParam
from vaml.utils.optimizer import make_optimizer
from rich.console import Console


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
        config.env.name, eval_batch_size, experiment.environments_seed, config.env
    )

    policy_opt = make_optimizer(model, config.policy_optimizer, config.total_update_episodes, nnx.LoRAParam)
    value_opt = make_optimizer(model, config.value_optimizer, config.total_update_episodes, ValueParam)

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        rngs.agent(),
    )

    agent.set_episode_instructions(env.instructions())

    trainer = Trainer(
        agent,
        policy_opt,
        value_opt,
        rngs.trainer(),
        checkpointer,
        logger,
        config,
    )

    if value_net_id is not None:
        other_exp = Experiment.load(value_net_id)
        with Checkpointer(other_exp.checkpoints_url) as other_checkpointer:
            trainer.restore_checkpoint(checkpointer=other_checkpointer, wrt=ValueParam)

    rollout_log_size = 100
    rollout_logger = BufferedEpisodeListener(
        rollout_log_size + config.eval_envs,
        rollout_log_size,
        config.max_seq_length,
        EpisodeSaver(experiment.rollout_dir),
    )

    trainer_listener = BufferedEpisodeListener(
        config.update_envs + config.eval_envs,
        config.update_envs,
        config.max_seq_length,
        trainer,
    )

    agent.episode_listener = MultiEpisodeListener(rollout_logger, trainer_listener)

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    obs = env.reset(env_indices)

    logger.start()

    while trainer.progress < 1.0:
        env_indices, actions = agent.act(env_indices, obs, rewards, dones)
        obs, rewards, dones, _ = env.step(env_indices, actions)

    logger.close()
    checkpointer.close()
