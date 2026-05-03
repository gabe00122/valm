from typing import NamedTuple, Protocol, Self, override

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from transformers import PreTrainedTokenizerFast
from vaml.agent.base import Agent
from vaml.buffer import UpdateBatch
from vaml.chat import (
    GenerationState,
    append_prompt_tokens,
    append_user_prompts,
    create_generation_state,
    decode_responses,
    encode_input,
    generate,
)
from vaml.config import Config
from vaml.episode_listener.base import EpisodeListener
from vaml.model.qwen3 import Qwen3


class EnvSpec(Protocol):
    max_turns: int
    metric_names: list[str]


class TurnData:
    def __init__(
        self,
        turn_counts: np.ndarray,
        turn_start_positions: np.ndarray,
        metrics: dict[str, np.ndarray],
    ):
        self._turn_counts = turn_counts
        self._turn_start_positions = turn_start_positions
        self._metrics = metrics

    @classmethod
    def create(cls, eval_envs: int, max_turns: int, metric_names: list[str]) -> Self:
        turn_counts = np.zeros((eval_envs,), dtype=np.int32)
        turn_start_positions = np.zeros((eval_envs, max_turns), dtype=np.int32)
        metrics = {
            name: np.zeros((eval_envs, max_turns), dtype=np.float32)
            for name in metric_names
        }
        return cls(turn_counts, turn_start_positions, metrics)

    def update(
        self,
        batch_idx: np.ndarray,
        turn_start_positions: np.ndarray,
        updates: dict[str, np.ndarray],
    ) -> None:
        turns = self._turn_counts[batch_idx]
        self._turn_start_positions[batch_idx, turns] = turn_start_positions

        for name, values in updates.items():
            self._metrics[name][batch_idx, turns] = values

        self._turn_counts[batch_idx] += 1

    def take(
        self, done_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        turn_counts = self._turn_counts[done_idx]
        turn_start_positions = self._turn_start_positions[done_idx]
        metrics = {name: m[done_idx] for name, m in self._metrics.items()}

        self._turn_counts[done_idx] = 0

        return turn_counts, turn_start_positions, metrics


class NpGenData(NamedTuple):
    kv_cache_length: np.ndarray
    context: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    policy_mask: np.ndarray
    context_length: np.ndarray
    turn_start_positions: np.ndarray
    turn_finished: np.ndarray


def convert_to_np(gen: GenerationState) -> NpGenData:
    data = (
        gen.kv_cache_length,
        gen.context,
        gen.log_probs,
        gen.values,
        gen.policy_mask,
        gen.context_length,
        gen.turn_start_positions,
        gen.turn_finished,
    )
    return NpGenData(**jax.tree.map(lambda x: np.array(x), jax.device_get(data)))


class LocalAgent(Agent):
    def __init__(
        self,
        model: Qwen3,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
        max_turns: int,
        metric_names: list[str],
        rng_key: jax.Array,
    ):
        self.episode_listener: EpisodeListener | None = None
        self._turn_data = TurnData.create(config.eval_envs, max_turns, metric_names)

        self.model_def, self.model_state = nnx.split(model)
        self._tokenizer = tokenizer
        self._config = config

        self._rng_key = rng_key

        shape = (self._config.eval_envs, self._config.max_seq_length)
        kv_cache = model.initialize_carry(*shape)
        self._gen = create_generation_state(
            kv_cache,
            self._config.eval_envs,
            self._config.max_seq_length,
            self._rng_key,
        )
        self._np_gen = convert_to_np(self._gen)

        self._rewards = np.zeros(
            (self._config.eval_envs, self._config.max_seq_length),
            dtype=np.float32,
        )

    def set_episode_instructions(self, instructions: str):
        instruction_tokens = encode_input(
            self._tokenizer,
            [
                [{"role": "system", "content": instructions}]
                for _ in range(self._config.eval_envs)
            ],
            False,
        )

        self._np_gen = convert_to_np(self._gen)
        append_prompt_tokens(
            self._np_gen,
            np.arange(self._config.eval_envs, dtype=np.int32),
            instruction_tokens,
        )
        self._env_instruction_length = self._np_gen.context_length[0].item()

        self._gen = self._gen._replace(
            env_instruction_length=self._gen.context_length.copy(),
        )

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
        metrics: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, list[str]]:
        lengths = self._np_gen.kv_cache_length

        # maybe we should save the rewards in the dense form and not the sparse form and
        # also the list of turn transition ids to it can be expanded into the sparse form
        self._rewards[batch_indices, lengths[batch_indices]] = rewards
        self._turn_data.update(
            batch_indices,
            self._np_gen.context_length,
            metrics,
        )

        done_idx = batch_indices[np.where(dones)]

        if dones.any():
            if self.episode_listener is not None:
                turn_counts, turn_start_positions, metrics = self._turn_data.take(
                    done_idx
                )
                self.episode_listener.on_episodes(
                    UpdateBatch(
                        context_length=lengths[done_idx],
                        context=self._np_gen.context[done_idx],
                        log_probs=self._np_gen.log_probs[done_idx],
                        values=self._np_gen.values[done_idx],
                        rewards=self._rewards[done_idx],
                        policy_mask=self._np_gen.policy_mask[done_idx],
                        turn_counts=turn_counts,
                        turn_start_positions=turn_start_positions,
                        metrics=metrics,
                    )
                )

            self._np_gen.context_length[done_idx] = self._env_instruction_length
            # force a re-revaluation
            # self._env_instruction_length
            self._np_gen.kv_cache_length[done_idx] = 0
            self._rewards[done_idx] = 0.0

        append_user_prompts(self._np_gen, batch_indices, self._tokenizer, obs)
        context, kv_cache_length, context_length, turn_start_positions = jax.device_put(
            (
                self._np_gen.context,
                self._np_gen.kv_cache_length,
                self._np_gen.context_length,
                self._np_gen.turn_start_positions,
            )
        )
        self._gen = self._gen._replace(
            context=context,
            turn_start_positions=turn_start_positions,
            context_length=context_length,
            kv_cache_length=kv_cache_length,
            turn_finished=jnp.zeros_like(self._gen.turn_finished),
        )

        self._gen = generate(self.model_def, self.model_state, "simple", self._gen, 4)
        self._np_gen = convert_to_np(self._gen)

        response_indices, response = decode_responses(self._tokenizer, self._np_gen)

        return response_indices, response
