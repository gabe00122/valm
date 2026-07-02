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
    GroupedEpisodeListener,
    GRPOTrainer,
    Trainer,
)
from vaml.experiment import Experiment
from vaml.logger import create_logger
from vaml.model.value_network import ValueParam
from vaml.utils.optimizer import make_optimizer


def train_cli(
    config_url: str,
    value_net_id: str | None = None,
    lora_init_id: str | None = None,
    lora_init_step: int | None = None,
    run_id: str | None = None,
    base_dir: str = "results",
    save_checkpoints: bool = True,
    save_rollouts: bool = True,
    wandb_tags: list[str] | None = None,
):
    if lora_init_step is not None and lora_init_id is None:
        raise ValueError("lora_init_step requires lora_init_id to be set")

    experiment = Experiment.from_config_file(
        config_url, base_dir=base_dir, unique_token=run_id
    )

    config = experiment.config
    console = Console()
    logger = create_logger(experiment, console, wandb_tags)

    rngs = nnx.Rngs(experiment.params_seed)
    model, tokenizer, sampling = load_base_model(config.base_model, rngs)
    # GRPO has no critic; only build the value net for the PPO loss.
    if config.loss.type == "ppo":
        model.initialize_value_net(config.value_net, rngs=rngs)
    model.initialize_lora(config.lora, rngs=rngs)

    # Optionally warm-start the LoRA parameters (only) from another experiment's
    # checkpoint. The value net is intentionally left untouched.
    if lora_init_id is not None:
        lora_exp = Experiment.load(lora_init_id, base_dir)
        with Checkpointer(lora_exp.checkpoints_url) as lora_checkpointer:
            if lora_init_step is None:
                lora_checkpointer.restore_latest(
                    {"model": model}, nnx.LoRAParam, partial=True
                )
            else:
                lora_checkpointer.restore(
                    {"model": model}, lora_init_step, nnx.LoRAParam, partial=True
                )

    checkpointer = Checkpointer(experiment.checkpoints_url)

    eval_batch_size = config.eval_envs
    env = make_env(
        config.env.name,
        eval_batch_size,
        config.group_size,
        experiment.environments_seed,
        config.env,
    )

    env_indices = np.arange(eval_batch_size, dtype=np.int32)
    obs, group_ids, metrics = env.reset(env_indices)

    assert config.policy_optimizer is not None
    policy_opt = make_optimizer(
        model,
        config.policy_optimizer,
        config.total_update_episodes,
        config.gradient_accumulations,
        nnx.LoRAParam,
    )

    agent = LocalAgent(
        model,
        tokenizer,
        config,
        env.max_turns,
        rngs.agent(),
    )

    agent.set_episode_instructions(env.instructions())

    rollout_logger = None
    if save_rollouts:
        rollout_log_size = 100
        rollout_logger = BufferedEpisodeListener(
            rollout_log_size + config.update_envs,
            rollout_log_size,
            config.max_seq_length,
            env.max_turns,
            EpisodeSaver(experiment.rollout_dir),
        )

    if config.loss.type == "ppo":
        value_opt = make_optimizer(
            model,
            config.value_optimizer,
            config.total_update_episodes,
            config.gradient_accumulations,
            ValueParam,
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
            save_periodic_checkpoints=save_checkpoints,
        )

        if value_net_id is not None:
            other_exp = Experiment.load(value_net_id, base_dir)
            with Checkpointer(other_exp.checkpoints_url) as other_checkpointer:
                trainer.restore_checkpoint(
                    checkpointer=other_checkpointer, wrt=ValueParam
                )
    else:
        trainer = GRPOTrainer(
            agent,
            policy_opt,
            rngs.trainer(),
            checkpointer,
            logger,
            config,
            episode_listener=rollout_logger,
            save_periodic_checkpoints=save_checkpoints,
        )

    trainer_listener = GroupedEpisodeListener(
        config.group_size,
        config.update_envs,
        trainer,
    )

    agent.episode_listener = trainer_listener

    rewards = np.zeros((eval_batch_size,), dtype=np.float32)
    dones = np.zeros((eval_batch_size,), dtype=np.bool_)

    logger.start()

    while trainer.progress < 1.0:
        env_indices, actions = agent.act(
            env_indices, obs, rewards, dones, group_ids, metrics
        )
        obs, rewards, dones, group_ids, metrics = env.step(env_indices, actions)

    # Always keep the final policy, even when periodic checkpoints are disabled
    # (no-op if the last periodic checkpoint already covers this step).
    trainer.save_checkpoint()

    logger.close()
    checkpointer.close()
