import os
from pathlib import Path
from typing import NamedTuple, Protocol, override

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from vaml.agent.base import Agent
from vaml.buffer import UpdateBatch, UpdateBuffer
from vaml.chat import (
    GenerationState,
    append_prompt_tokens,
    append_user_prompts,
    create_generation_state,
    decode_responses,
    encode_input,
    generate,
)
from vaml.checkpointer import Checkpointer
from vaml.config import Config
from vaml.logger import MetricsAccumulator
from vaml.model.qwen3 import Qwen3
from vaml.model.value_network import ValueParam
from vaml.update_step import update_step
from vaml.utils.performance import PerformanceTracker
from transformers import PreTrainedTokenizerFast


class EpisodeListener(Protocol):
    def on_episodes(self, batch: UpdateBatch): ...


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
        performance: PerformanceTracker,
        logger: MetricsAccumulator,
        config: Config,
    ):
        self._model_provider = model_provider
        self._policy_opt_def, self._policy_opt_state = nnx.split(policy_opt)
        self._value_opt_def, self._value_opt_state = nnx.split(value_opt)
        self._rng_key = rng_key

        self._checkpointer = checkpointer
        self._performance = performance
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
            {
                "policy_opt": policy_opt,
                "value_opt": value_opt,
                "model": model
            },
            self._update_step,
            nnx.filterlib.Any(nnx.OptState, policy_opt.wrt, value_opt.wrt)
        )

    def restore_checkpoint(self, *, checkpointer: Checkpointer | None = None, wrt: nnx.filterlib.Filter | None = None):
        policy_opt: nnx.Optimizer = nnx.merge(self._policy_opt_def, self._policy_opt_state)
        value_opt: nnx.Optimizer = nnx.merge(self._value_opt_def, self._value_opt_state)
        model = nnx.merge(
            self._model_provider.model_def, self._model_provider.model_state
        )

        restore_filter = nnx.filterlib.Any(nnx.OptState, policy_opt.wrt, value_opt.wrt)

        if checkpointer is None:
            step = self._checkpointer.restore_latest({"policy_opt": policy_opt, "value_opt": value_opt, "model": model}, restore_filter)
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
        with self._performance.time("update_step"):
            self._policy_opt_state, self._value_opt_state, new_model_state, metrics, self._rng_key = update_step(
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

        self._model_provider.model_state = new_model_state

        # metrics["rewards"] = batch.rewards.sum() / batch.rewards.shape[0]
        metrics["performance"] = self._performance.total_time_percentages()
        self._performance.reset()

        self._logger.add(metrics)
        self._update_step += 1

        # batch.save_npz("./episode_viewer/episodes.npz")

        if self._update_step % self._config.checkpoint_every == 0:
            self.save_checkpoint()


class EpisodeSaver(EpisodeListener):
    def __init__(self, directory: str):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self.chunk_num = 0

    def on_episodes(self, batch: UpdateBatch):
        file_name = os.path.join(self._directory, f"episodes_{self.chunk_num}.npz")
        batch.save_npz(file_name, compressed=False)
        self.chunk_num += 1


class MultiEpisodeListener(EpisodeListener):
    def __init__(self, listeners: list[EpisodeListener]):
        self._listeners = listeners

    def on_episodes(self, batch: UpdateBatch):
        for listener in self._listeners:
            listener.on_episodes(batch)


class BufferedEpisodeListener(EpisodeListener):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        seq_length: int,
        listener: EpisodeListener,
    ):
        self._listener = listener
        self._buffer = UpdateBuffer(buffer_size, batch_size, seq_length)

    @property
    def size(self) -> int:
        return self._buffer.size

    def on_episodes(self, batch: UpdateBatch):
        self._buffer.store(batch)
        while self._buffer.has_batch:
            self._listener.on_episodes(self._buffer.take_batch())


class NpGenData(NamedTuple):
    kv_cache_length: jax.typing.ArrayLike
    context: jax.typing.ArrayLike
    log_probs: jax.typing.ArrayLike
    values: jax.typing.ArrayLike
    policy_mask: jax.typing.ArrayLike
    context_length: jax.typing.ArrayLike
    turn_start_positions: jax.typing.ArrayLike
    turn_finished: jax.typing.ArrayLike


def get_np_gen_data(gen: GenerationState) -> NpGenData:
    data = NpGenData(
        gen.kv_cache_length,
        gen.context,
        gen.log_probs,
        gen.values,
        gen.policy_mask,
        gen.context_length,
        gen.turn_start_positions,
        gen.turn_finished
    )
    return jax.tree.map(lambda x: np.array(x), jax.device_get(data))

class LocalAgent(Agent, ModelProvider):
    def __init__(
        self,
        model: Qwen3,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
        logger: MetricsAccumulator,
        performance_tracker: PerformanceTracker,
        rng_key: jax.Array,
    ):
        self.episode_listener: EpisodeListener | None = None

        self.model_def, self.model_state = nnx.split(model)
        self._tokenizer = tokenizer
        self._config = config
        self._logger = logger
        self._performance_tracker = performance_tracker

        self._rng_key = rng_key

        shape = (self._config.eval_envs, self._config.max_seq_length)
        kv_cache = model.initialize_carry(*shape)
        self._gen = create_generation_state(
            kv_cache, self._config.eval_envs, self._config.max_seq_length, self._rng_key
        )
        self._np_gen = get_np_gen_data(self._gen)

        self._rewards = np.zeros(
            (self._config.eval_envs, self._config.max_seq_length), dtype=np.float32
        )

        self._last_tokens = 0

    def set_episode_instructions(self, instructions: str):
        instruction_tokens = encode_input(
            self._tokenizer,
            [
                [{"role": "system", "content": instructions}]
                for _ in range(self._config.eval_envs)
            ],
            False,
        )

        self._np_gen = get_np_gen_data(self._gen)
        append_prompt_tokens(
            self._np_gen,
            np.arange(self._config.eval_envs, dtype=np.int32),
            instruction_tokens,
        )
        self._env_instruction_length = self._np_gen.context_length[0].item()

        # this can probably go away now
        self._gen = self._gen._replace(
            env_instruction_length=self._gen.context_length.copy(),
        )

    def _report_tps(self):
        current_tokens = self._gen.tokens_processed.item()
        # token_delta = current_tokens - self._last_tokens
        # self._last_tokens = current_tokens

        # self._logger.add_counts({
        #     "tokens": current_tokens
        # })

    @override
    def reset(self) -> None:
        pass

    @override
    def act(
        self,
        batch_indices: np.ndarray,
        obs: list[str],
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        kv_cache_lengths = self._np_gen.kv_cache_length
        self._rewards[batch_indices, kv_cache_lengths[batch_indices]] = rewards

        done_idx = batch_indices[np.where(dones)]

        if dones.any():
            if self.episode_listener is not None:
                self.episode_listener.on_episodes(
                    UpdateBatch(
                        self._np_gen.context[done_idx],
                        kv_cache_lengths[done_idx],
                        self._np_gen.log_probs[done_idx],
                        self._np_gen.values[done_idx],
                        self._rewards[done_idx],
                        self._np_gen.policy_mask[done_idx],
                    )
                )

            with self._performance_tracker.time("reset"):
                self._np_gen.context_length[done_idx] = self._env_instruction_length
                # force a re-revaluation
                self._np_gen.kv_cache_length[done_idx] = 0 #self._env_instruction_length
                self._rewards[done_idx] = 0.0

        with self._performance_tracker.time("encode"):
            append_user_prompts(
                self._np_gen, batch_indices, self._tokenizer, obs
            )
            context, kv_cache_length, context_length, turn_start_positions = jax.device_put(
                (self._np_gen.context, self._np_gen.kv_cache_length, self._np_gen.context_length, self._np_gen.turn_start_positions)
            )
            self._gen = self._gen._replace(
                context=context,
                turn_start_positions=turn_start_positions,
                context_length=context_length,
                kv_cache_length=kv_cache_length,
                turn_finished=jnp.zeros_like(self._gen.turn_finished),
            )
            self._report_tps()

        with self._performance_tracker.time("generate"):
            self._gen = generate(
                self.model_def, self.model_state, "simple", self._gen, 4
            )
            self._np_gen = get_np_gen_data(self._gen)

        with self._performance_tracker.time("decode"):
            response_indices, response = decode_responses(self._tokenizer, self._np_gen)

        return response_indices, response
