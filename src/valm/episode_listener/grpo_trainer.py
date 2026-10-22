import jax
import numpy as np
from flax import nnx
from valm.buffer import UpdateBatch
from valm.checkpointer import Checkpointer
from valm.config import Config, GRPOLossConfig
from valm.episode_listener.base import EpisodeListener
from valm.episode_listener.trainer import ModelProvider
from valm.logger import BaseLogger
from valm.update_step.grpo import multi_grpo_update_bucketed


class GRPOTrainer(EpisodeListener):
    """GRPO update loop: policy-only, no value network.

    Mirrors Trainer but drops the critic entirely. Advantages come from
    group-relative reward normalization (see update_step.grpo), so there is no
    value optimizer to build, checkpoint, or restore.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        policy_opt: nnx.Optimizer,
        rng_key: jax.Array,
        checkpointer: Checkpointer,
        logger: BaseLogger,
        config: Config,
        episode_listener: EpisodeListener | None = None,
        save_periodic_checkpoints: bool = True,
    ):
        assert isinstance(config.loss, GRPOLossConfig)

        self._model_provider = model_provider
        self._policy_opt_def, self._policy_opt_state = nnx.split(policy_opt)
        self._rng_key = rng_key

        self._checkpointer = checkpointer
        self._logger = logger
        self._config = config
        self._episode_listener = episode_listener
        self._save_periodic_checkpoints = save_periodic_checkpoints
        self._update_step = 0
        self._last_checkpoint_step = -1

    def save_checkpoint(self):
        if self._update_step == self._last_checkpoint_step:
            return
        policy_opt = nnx.merge(self._policy_opt_def, self._policy_opt_state)
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )
        self._checkpointer.save(
            {"policy_opt": policy_opt, "model": model},
            self._update_step,
            nnx.filterlib.Any(nnx.OptState, policy_opt.wrt),
        )
        self._last_checkpoint_step = self._update_step

    def restore_checkpoint(self):
        policy_opt: nnx.Optimizer = nnx.merge(
            self._policy_opt_def, self._policy_opt_state
        )
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )

        restore_filter = nnx.filterlib.Any(nnx.OptState, policy_opt.wrt)
        step = self._checkpointer.restore_latest(
            {"policy_opt": policy_opt, "model": model},
            restore_filter,
        )
        self._update_step = step
        self._last_checkpoint_step = step
        self._policy_opt_state = nnx.state(policy_opt)
        self._model_provider.model_state = nnx.state(model)

    @property
    def progress(self) -> float:
        return self._update_step / self._config.total_update_episodes

    def on_episodes(self, batch: UpdateBatch):
        (
            self._policy_opt_state,
            new_model_state,
            summery_metrics,
            token_metrics,
            self._rng_key,
        ) = multi_grpo_update_bucketed(
            self._policy_opt_def,
            self._policy_opt_state,
            self._model_provider.model_def,
            self._model_provider.model_state,
            self._rng_key,
            batch,
            self._config.loss,
            self._config.gradient_accumulations or 1,
            self._config.group_size,
        )

        seq_length = batch.rewards.shape[1]

        # summerize environment metrics
        env_metrics = {
            name: np.mean(np.sum(values, axis=1)).item()
            for name, values in batch.turn_metrics.items()
        }
        summery_metrics["env"] = env_metrics
        summery_metrics["turns"] = np.mean(batch.turn_counts)
        summery_metrics["truncated"] = np.mean(batch.context_length >= seq_length - 2)

        self._model_provider.model_state = new_model_state

        if self._episode_listener is not None:
            self._episode_listener.on_episodes(
                batch._replace(update_metrics=token_metrics)
            )

        self._logger.log_dict(summery_metrics, self._update_step)
        self._update_step += 1

        if (
            self._save_periodic_checkpoints
            and self._update_step % self._config.checkpoint_every == 0
        ):
            self.save_checkpoint()
