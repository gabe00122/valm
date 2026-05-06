from typing import Protocol

import jax
import numpy as np
from flax import nnx
from vaml.buffer import UpdateBatch
from vaml.checkpointer import Checkpointer
from vaml.config import Config
from vaml.episode_listener.base import EpisodeListener
from vaml.logger import BaseLogger
from vaml.model.value_network import ValueParam
from vaml.update_step import update_step


class ModelProvider(Protocol):
    model_def: nnx.GraphDef
    model_state: nnx.State


class Trainer(EpisodeListener):
    def __init__(
        self,
        model_provider: ModelProvider,
        policy_opt: nnx.Optimizer,
        value_opt: nnx.Optimizer,
        rng_key: jax.Array,
        checkpointer: Checkpointer,
        logger: BaseLogger,
        config: Config,
    ):
        self._model_provider = model_provider
        self._policy_opt_def, self._policy_opt_state = nnx.split(policy_opt)
        self._value_opt_def, self._value_opt_state = nnx.split(value_opt)
        self._rng_key = rng_key

        self._checkpointer = checkpointer
        self._logger = logger
        self._config = config
        self._update_step = 0

    def save_checkpoint(self):
        policy_opt = nnx.merge(self._policy_opt_def, self._policy_opt_state)
        value_opt = nnx.merge(self._value_opt_def, self._value_opt_state)
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )
        self._checkpointer.save(
            {"policy_opt": policy_opt, "value_opt": value_opt, "model": model},
            self._update_step,
            nnx.filterlib.Any(nnx.OptState, policy_opt.wrt, value_opt.wrt),
        )

    def restore_checkpoint(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        wrt: nnx.filterlib.Filter | None = None,
    ):
        policy_opt: nnx.Optimizer = nnx.merge(
            self._policy_opt_def, self._policy_opt_state
        )
        value_opt: nnx.Optimizer = nnx.merge(self._value_opt_def, self._value_opt_state)
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )

        restore_filter = nnx.filterlib.Any(nnx.OptState, policy_opt.wrt, value_opt.wrt)

        if checkpointer is None:
            step = self._checkpointer.restore_latest(
                {
                    "policy_opt": policy_opt,
                    "value_opt": value_opt,
                    "model": model,
                },
                restore_filter,
            )
            self._update_step = step
            self._policy_opt_state = nnx.state(policy_opt)
            self._value_opt_state = nnx.state(value_opt)
        else:
            # This should be the value_opt
            checkpointer.restore_latest({"model": model}, ValueParam)
            # self._value_opt_state = nnx.state(value_opt)

        self._model_provider.model_state = nnx.state(model)

    @property
    def progress(self) -> float:
        return self._update_step / self._config.total_update_episodes

    def on_episodes(self, batch: UpdateBatch):
        (
            self._policy_opt_state,
            self._value_opt_state,
            new_model_state,
            metrics,
            self._rng_key,
        ) = update_step(
            self._policy_opt_def,
            self._policy_opt_state,
            self._value_opt_def,
            self._value_opt_state,
            self._model_provider.model_def,
            self._model_provider.model_state,
            self._rng_key,
            batch,
            self._config.loss,
            False,
        )

        seq_length = batch.rewards.shape[1]

        # summerize environment metrics
        env_metrics = {name: np.sum(values) for name, values in batch.metrics.items()}
        metrics["env"] = env_metrics
        metrics["turns"] = np.mean(batch.turn_counts)
        metrics["truncated"] = np.mean(batch.context_length >= seq_length)

        self._model_provider.model_state = new_model_state

        self._logger.log_dict(metrics, self._update_step)
        self._update_step += 1

        if self._update_step % self._config.checkpoint_every == 0:
            self.save_checkpoint()
