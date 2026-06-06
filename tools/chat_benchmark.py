import qwix
import argparse
import gc
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx
from rich.console import Console
from rich.table import Table
from transformers import PreTrainedTokenizerFast
from vaml.base_model_loader import load_base_model
from vaml.chat import (
    NpGenData,
    append_prompt_tokens,
    convert_to_np,
    create_generation_state,
    decode_responses,
    encode_input,
    generate,
    update_gen_state,
)
from vaml.config import LoraConfig
from vaml.model.qwen3 import Qwen3


PromptKind = Literal["short", "long"]


SHORT_PROMPTS = [
    "Reply in one concise sentence: what makes a good benchmark?",
    "Give two practical tips for keeping latency low.",
    "Answer with a short analogy for KV cache reuse.",
    "Name one likely bottleneck in autoregressive decoding.",
    "In one sentence, explain why batching can improve throughput.",
]

LONG_PROMPTS = [
    (
        "You are reviewing an inference benchmark for a local language model. "
        "The benchmark appends user turns on the host, transfers generation state "
        "to JAX, runs a jitted token loop, transfers state back to NumPy, and then "
        "decodes completed responses. Summarize the main timing risks in three "
        "compact bullets, and keep each bullet under fifteen words."
    ),
    (
        "A team is comparing several prompt shapes for a multiturn chat workload. "
        "Some requests are tiny follow-up questions, while others include logs, "
        "instructions, and a few paragraphs of context. They care about model "
        "tokens processed per second, generated tokens per second, and how much "
        "time is spent crossing the NumPy and JAX boundary. Provide a brief "
        "measurement plan with the counters they should capture."
    ),
    (
        "Read this synthetic support-ticket transcript and answer with a short "
        "triage note. User: The service becomes slow after several turns. Agent: "
        "Please share batch size and sequence length. User: Batch size is eight, "
        "sequence length is 2048, and prompts alternate between one-line messages "
        "and long pasted traces. Agent: We should separate tokenization, transfer, "
        "jitted generation, and decode time. Write the final triage note."
    ),
    (
        "Consider a workload where every request keeps conversational history in "
        "a fixed-size context buffer. The host updates prompt tokens in NumPy, "
        "copies changed lengths and context to device arrays, and calls a jitted "
        "generate function until the model emits a stop token or reaches the "
        "sequence limit. Explain why prompt length, response length, and batch "
        "completion policy all affect observed throughput."
    ),
]


@dataclass
class TurnMetrics:
    turn: int
    prompt_kind: PromptKind
    prompt_tokens: int
    truncated_prompt_tokens: int
    model_tokens: int
    generated_tokens: int
    completed: int
    tokenize_s: float
    append_np_s: float
    np_to_jax_s: float
    jit_s: float
    metrics_sync_s: float
    jax_to_np_s: float
    decode_s: float

    @property
    def accounted_s(self) -> float:
        return (
            self.tokenize_s
            + self.append_np_s
            + self.np_to_jax_s
            + self.jit_s
            + self.metrics_sync_s
            + self.jax_to_np_s
            + self.decode_s
        )


@dataclass
class BenchmarkTotals:
    turns: int = 0
    prompt_tokens: int = 0
    truncated_prompt_tokens: int = 0
    model_tokens: int = 0
    generated_tokens: int = 0
    completed: int = 0
    wall_s: float = 0.0
    tokenize_s: float = 0.0
    append_np_s: float = 0.0
    np_to_jax_s: float = 0.0
    jit_s: float = 0.0
    metrics_sync_s: float = 0.0
    jax_to_np_s: float = 0.0
    decode_s: float = 0.0
    warmup_wall_s: float = 0.0
    warmup_jit_s: float = 0.0
    stopped_early: bool = False

    @property
    def accounted_s(self) -> float:
        return (
            self.tokenize_s
            + self.append_np_s
            + self.np_to_jax_s
            + self.jit_s
            + self.metrics_sync_s
            + self.jax_to_np_s
            + self.decode_s
        )

    @property
    def end_to_end_model_tps(self) -> float:
        return self.model_tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def jit_model_tps(self) -> float:
        return self.model_tokens / self.jit_s if self.jit_s > 0 else 0.0

    @property
    def end_to_end_generated_tps(self) -> float:
        return self.generated_tokens / self.wall_s if self.wall_s > 0 else 0.0


@dataclass
class DeviceMemorySnapshot:
    label: str
    device: str
    platform: str
    stats_available: bool
    bytes_in_use: int | None = None
    peak_bytes_in_use: int | None = None
    bytes_reserved: int | None = None
    peak_bytes_reserved: int | None = None
    bytes_limit: int | None = None
    largest_alloc_size: int | None = None
    note: str | None = None


@dataclass
class ArrayMemoryBreakdown:
    model_state_bytes: int = 0
    kv_cache_bytes: int = 0
    generation_buffer_bytes: int = 0

    @property
    def tracked_total_bytes(self) -> int:
        return (
            self.model_state_bytes + self.kv_cache_bytes + self.generation_buffer_bytes
        )


@dataclass
class MemoryBreakdown:
    array_bytes: ArrayMemoryBreakdown = field(default_factory=ArrayMemoryBreakdown)
    snapshots: list[DeviceMemorySnapshot] = field(default_factory=list)


def _block_until_ready(tree: Any) -> None:
    jax.tree.map(
        lambda x: (
            getattr(x, "value", x).block_until_ready()
            if hasattr(getattr(x, "value", x), "block_until_ready")
            else x
        ),
        tree,
    )


def _trace_annotation(name: str, profile_enabled: bool, **kwargs):
    if not profile_enabled:
        return nullcontext()
    return jax.profiler.TraceAnnotation(name, **kwargs)


def _step_annotation(name: str, step_num: int, profile_enabled: bool):
    if not profile_enabled:
        return nullcontext()
    return jax.profiler.StepTraceAnnotation(name, step_num=step_num)


def _profile_trace(
    profile_dir: str | None,
    *,
    create_perfetto_trace: bool,
    host_tracer_level: int,
    python_tracer_level: int,
):
    if profile_dir is None:
        return nullcontext()

    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = host_tracer_level
    profiler_options.python_tracer_level = python_tracer_level
    return jax.profiler.trace(
        profile_dir,
        create_perfetto_trace=create_perfetto_trace,
        profiler_options=profiler_options,
    )


def _leaf_nbytes(leaf: Any) -> int:
    value = getattr(leaf, "value", leaf)
    nbytes = getattr(value, "nbytes", None)
    return int(nbytes) if nbytes is not None else 0


def _tree_nbytes(tree: Any) -> int:
    return sum(_leaf_nbytes(leaf) for leaf in jax.tree.leaves(tree))


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    return f"{num_bytes} B"


def _read_int_stat(stats: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = stats.get(key)
        if value is not None:
            return int(value)
    return None


def _device_memory_snapshots(label: str) -> list[DeviceMemorySnapshot]:
    snapshots: list[DeviceMemorySnapshot] = []
    for device in jax.local_devices():
        stats_fn = getattr(device, "memory_stats", None)
        if not callable(stats_fn):
            snapshots.append(
                DeviceMemorySnapshot(
                    label=label,
                    device=str(device),
                    platform=getattr(device, "platform", "unknown"),
                    stats_available=False,
                    note="device does not expose memory_stats()",
                )
            )
            continue

        try:
            raw_stats = stats_fn()
        except Exception as exc:
            snapshots.append(
                DeviceMemorySnapshot(
                    label=label,
                    device=str(device),
                    platform=getattr(device, "platform", "unknown"),
                    stats_available=False,
                    note=f"memory_stats() failed: {exc}",
                )
            )
            continue

        if not raw_stats:
            snapshots.append(
                DeviceMemorySnapshot(
                    label=label,
                    device=str(device),
                    platform=getattr(device, "platform", "unknown"),
                    stats_available=False,
                    note="memory_stats() returned no counters",
                )
            )
            continue

        stats = dict(raw_stats)
        snapshots.append(
            DeviceMemorySnapshot(
                label=label,
                device=str(device),
                platform=getattr(device, "platform", "unknown"),
                stats_available=True,
                bytes_in_use=_read_int_stat(stats, "bytes_in_use"),
                peak_bytes_in_use=_read_int_stat(stats, "peak_bytes_in_use"),
                bytes_reserved=_read_int_stat(stats, "bytes_reserved"),
                peak_bytes_reserved=_read_int_stat(stats, "peak_bytes_reserved"),
                bytes_limit=_read_int_stat(
                    stats,
                    "bytes_limit",
                    "bytes_reservable_limit",
                    "bytes_reserved_limit",
                ),
                largest_alloc_size=_read_int_stat(stats, "largest_alloc_size"),
            )
        )

    return snapshots


def _generation_buffer_nbytes(gen: Any) -> int:
    return _tree_nbytes(gen._replace(kv_cache=None))


def _prompt_kind(turn: int, prompt_set: str) -> PromptKind:
    if prompt_set == "short":
        return "short"
    if prompt_set == "long":
        return "long"
    return "short" if turn % 2 == 0 else "long"


def _resolve_lora_config(args: argparse.Namespace) -> LoraConfig | None:
    if not args.lora and not args.lora_attn and not args.lora_mlp:
        return None

    if args.lora_rank < 1:
        raise ValueError("--lora-rank must be at least 1 when LoRA is enabled")

    attn = args.lora or args.lora_attn
    mlp = args.lora or args.lora_mlp
    return LoraConfig(attn=attn, mlp=mlp, rank=args.lora_rank)


def _build_prompts(batch_size: int, turn: int, prompt_kind: PromptKind) -> list[str]:
    bank = SHORT_PROMPTS if prompt_kind == "short" else LONG_PROMPTS
    prompts = []
    for batch_index in range(batch_size):
        prompt = bank[(turn * batch_size + batch_index) % len(bank)]
        prompts.append(f"{prompt}\n\nRequest id: turn={turn}, batch={batch_index}.")
    return prompts


def _token_lengths(prompt_tokens: np.ndarray | Sequence[np.ndarray]) -> list[int]:
    return [int(np.asarray(prompt).shape[0]) for prompt in prompt_tokens]


def _new_generation_state(
    model: Qwen3,
    batch_size: int,
    seq_length: int,
    rng_key: jax.Array,
):
    kv_cache = model.initialize_carry(batch_size, seq_length)
    gen = create_generation_state(kv_cache, batch_size, seq_length, rng_key)
    _block_until_ready(gen)
    return gen, convert_to_np(gen)


def _quantization_model_inputs(
    model: Qwen3,
    batch_size: int,
    seq_length: int,
    rng_key: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array, Any], dict[str, jax.Array]]:
    tokens = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    positions = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    kv_cache = model.initialize_carry(batch_size, seq_length)
    return (tokens, positions, kv_cache), {"rng_key": rng_key}


def _quantize_int8_model(
    model: Qwen3,
    *,
    batch_size: int,
    seq_length: int,
    seed: int,
) -> tuple[Qwen3, int]:
    rules = [
        qwix.QuantizationRule(
            module_path=".*",
            weight_qtype="int8",
        )
    ]
    provider = qwix.PtqProvider(rules)
    model_inputs, model_input_kwargs = _quantization_model_inputs(
        model,
        batch_size,
        seq_length,
        jax.random.PRNGKey(seed),
    )
    quantized_model = qwix.quantize_model(
        model,
        provider,
        *model_inputs,
        **model_input_kwargs,
    )
    return quantized_model, sum(getattr(provider, "_rule_matches", ()))


def _run_turn(
    *,
    tokenizer: PreTrainedTokenizerFast,
    model_def: Any,
    model_state: Any,
    gen: Any,
    np_gen: NpGenData,
    batch_indices: np.ndarray,
    batch_size: int,
    wait_for: int,
    turn: int,
    prompt_kind: PromptKind,
    decode: bool,
    profile_enabled: bool,
) -> tuple[Any, NpGenData, TurnMetrics]:
    prompts = _build_prompts(batch_size, turn, prompt_kind)
    conversation_turns = [[{"role": "user", "content": content}] for content in prompts]

    with _trace_annotation("tokenize_prompts", profile_enabled):
        start = time.perf_counter()
        prompt_tokens = encode_input(tokenizer, conversation_turns)
        tokenize_s = time.perf_counter() - start

    prompt_token_count = sum(_token_lengths(prompt_tokens))
    before_prompt_lengths = np_gen.context_length.copy()

    with _trace_annotation("append_prompts_numpy", profile_enabled):
        start = time.perf_counter()
        append_prompt_tokens(np_gen, batch_indices, prompt_tokens)
        append_np_s = time.perf_counter() - start

    after_prompt_lengths = np_gen.context_length.copy()
    appended_prompt_tokens = int(np.sum(after_prompt_lengths - before_prompt_lengths))
    truncated_prompt_tokens = max(prompt_token_count - appended_prompt_tokens, 0)

    with _trace_annotation("numpy_to_jax_transfer", profile_enabled):
        start = time.perf_counter()
        gen = update_gen_state(gen, np_gen)
        _block_until_ready(
            (
                gen.context,
                gen.kv_cache_length,
                gen.context_length,
                gen.turn_start_positions,
                gen.turn_finished,
            )
        )
        np_to_jax_s = time.perf_counter() - start

    start_tokens = int(jax.device_get(gen.tokens_processed))

    with _trace_annotation(
        "jitted_generate",
        profile_enabled,
        prompt_kind=prompt_kind,
        wait_for=wait_for,
    ):
        start = time.perf_counter()
        gen = generate(model_def, model_state, "simple", gen, wait_for)
        _block_until_ready(gen)
        jit_s = time.perf_counter() - start

    with _trace_annotation("metrics_scalar_sync", profile_enabled):
        start = time.perf_counter()
        end_tokens = int(jax.device_get(gen.tokens_processed))
        metrics_sync_s = time.perf_counter() - start

    model_tokens = end_tokens - start_tokens

    with _trace_annotation("jax_to_numpy_transfer", profile_enabled):
        start = time.perf_counter()
        np_gen = convert_to_np(gen)
        jax_to_np_s = time.perf_counter() - start

    generated_tokens = int(np.sum(np_gen.context_length - after_prompt_lengths))

    with _trace_annotation("decode_responses", profile_enabled):
        start = time.perf_counter()
        completed = 0
        if decode:
            response_indices, _ = decode_responses(tokenizer, np_gen)
            completed = int(response_indices.shape[0])
        decode_s = time.perf_counter() - start

    return (
        gen,
        np_gen,
        TurnMetrics(
            turn=turn,
            prompt_kind=prompt_kind,
            prompt_tokens=appended_prompt_tokens,
            truncated_prompt_tokens=truncated_prompt_tokens,
            model_tokens=model_tokens,
            generated_tokens=generated_tokens,
            completed=completed,
            tokenize_s=tokenize_s,
            append_np_s=append_np_s,
            np_to_jax_s=np_to_jax_s,
            jit_s=jit_s,
            metrics_sync_s=metrics_sync_s,
            jax_to_np_s=jax_to_np_s,
            decode_s=decode_s,
        ),
    )


def run_benchmark(
    *,
    model: Qwen3,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    seq_length: int,
    turns: int,
    prompt_set: str,
    wait_for: int,
    warmup_turns: int,
    seed: int,
    decode: bool,
    profile_dir: str | None,
    profile_perfetto_trace: bool,
    profile_host_tracer_level: int,
    profile_python_tracer_level: int,
) -> tuple[BenchmarkTotals, list[TurnMetrics], MemoryBreakdown]:
    model_def, model_state = nnx.split(model)
    _block_until_ready(model_state)

    memory = MemoryBreakdown(
        snapshots=_device_memory_snapshots("after_model_split"),
    )
    batch_indices = np.arange(batch_size, dtype=np.int32)
    profile_enabled = profile_dir is not None

    warmup_totals = BenchmarkTotals()
    if warmup_turns > 0:
        warmup_gen, warmup_np_gen = _new_generation_state(
            model, batch_size, seq_length, jax.random.PRNGKey(seed)
        )
        start = time.perf_counter()
        for turn in range(warmup_turns):
            prompt_kind = _prompt_kind(turn, prompt_set)
            warmup_gen, warmup_np_gen, metrics = _run_turn(
                tokenizer=tokenizer,
                model_def=model_def,
                model_state=model_state,
                gen=warmup_gen,
                np_gen=warmup_np_gen,
                batch_indices=batch_indices,
                batch_size=batch_size,
                wait_for=wait_for,
                turn=turn,
                prompt_kind=prompt_kind,
                decode=False,
                profile_enabled=False,
            )
            warmup_totals.jit_s += metrics.jit_s
            if np.all(warmup_np_gen.context_length >= seq_length):
                break
        warmup_totals.wall_s = time.perf_counter() - start
        memory.snapshots.extend(_device_memory_snapshots("after_warmup"))
        del warmup_gen, warmup_np_gen
        gc.collect()

    # Do not reuse/reset the warmup state: the model-returned KV cache can have
    # a different JIT cache signature than a fresh initialize_carry() cache.
    gen, np_gen = _new_generation_state(
        model, batch_size, seq_length, jax.random.PRNGKey(seed + 1)
    )
    memory.snapshots.extend(_device_memory_snapshots("after_state_init"))

    memory.array_bytes = ArrayMemoryBreakdown(
        model_state_bytes=_tree_nbytes(model_state),
        kv_cache_bytes=_tree_nbytes(gen.kv_cache),
        generation_buffer_bytes=_generation_buffer_nbytes(gen),
    )

    totals = BenchmarkTotals(
        warmup_wall_s=warmup_totals.wall_s,
        warmup_jit_s=warmup_totals.jit_s,
    )
    turn_metrics: list[TurnMetrics] = []

    with _profile_trace(
        profile_dir,
        create_perfetto_trace=profile_perfetto_trace,
        host_tracer_level=profile_host_tracer_level,
        python_tracer_level=profile_python_tracer_level,
    ):
        start_wall = time.perf_counter()
        for turn in range(turns):
            prompt_kind = _prompt_kind(turn, prompt_set)
            with _step_annotation("chat_benchmark_turn", turn, profile_enabled):
                gen, np_gen, metrics = _run_turn(
                    tokenizer=tokenizer,
                    model_def=model_def,
                    model_state=model_state,
                    gen=gen,
                    np_gen=np_gen,
                    batch_indices=batch_indices,
                    batch_size=batch_size,
                    wait_for=wait_for,
                    turn=turn,
                    prompt_kind=prompt_kind,
                    decode=decode,
                    profile_enabled=profile_enabled,
                )
            turn_metrics.append(metrics)

            totals.turns += 1
            totals.prompt_tokens += metrics.prompt_tokens
            totals.truncated_prompt_tokens += metrics.truncated_prompt_tokens
            totals.model_tokens += metrics.model_tokens
            totals.generated_tokens += metrics.generated_tokens
            totals.completed += metrics.completed
            totals.tokenize_s += metrics.tokenize_s
            totals.append_np_s += metrics.append_np_s
            totals.np_to_jax_s += metrics.np_to_jax_s
            totals.jit_s += metrics.jit_s
            totals.metrics_sync_s += metrics.metrics_sync_s
            totals.jax_to_np_s += metrics.jax_to_np_s
            totals.decode_s += metrics.decode_s

            if np.all(np_gen.context_length >= seq_length):
                totals.stopped_early = turn + 1 < turns
                break

    totals.wall_s = time.perf_counter() - start_wall
    memory.snapshots.extend(_device_memory_snapshots("after_benchmark"))
    return totals, turn_metrics, memory


def _add_time_row(table: Table, name: str, seconds: float, wall_s: float) -> None:
    pct = (seconds / wall_s * 100.0) if wall_s > 0 else 0.0
    table.add_row(name, f"{seconds:.6f}", f"{pct:.1f}%")


def print_memory_report(console: Console, memory: MemoryBreakdown) -> None:
    array_table = Table(title="Tracked JAX Array VRAM")
    array_table.add_column("Category")
    array_table.add_column("Bytes", justify="right")
    array_table.add_column("Size", justify="right")
    array_table.add_row(
        "model parameters/state",
        str(memory.array_bytes.model_state_bytes),
        _format_bytes(memory.array_bytes.model_state_bytes),
    )
    array_table.add_row(
        "KV cache",
        str(memory.array_bytes.kv_cache_bytes),
        _format_bytes(memory.array_bytes.kv_cache_bytes),
    )
    array_table.add_row(
        "generation buffers",
        str(memory.array_bytes.generation_buffer_bytes),
        _format_bytes(memory.array_bytes.generation_buffer_bytes),
    )
    array_table.add_row(
        "tracked total",
        str(memory.array_bytes.tracked_total_bytes),
        _format_bytes(memory.array_bytes.tracked_total_bytes),
    )
    console.print(array_table)

    snapshot_table = Table(title="Allocator VRAM Snapshots")
    snapshot_table.add_column("Point")
    snapshot_table.add_column("Device")
    snapshot_table.add_column("In Use", justify="right")
    snapshot_table.add_column("Peak", justify="right")
    snapshot_table.add_column("Reserved", justify="right")
    snapshot_table.add_column("Limit", justify="right")
    snapshot_table.add_column("Note")

    for snapshot in memory.snapshots:
        snapshot_table.add_row(
            snapshot.label,
            f"{snapshot.platform}:{snapshot.device}",
            _format_bytes(snapshot.bytes_in_use),
            _format_bytes(snapshot.peak_bytes_in_use),
            _format_bytes(snapshot.bytes_reserved),
            _format_bytes(snapshot.bytes_limit),
            snapshot.note or "",
        )
    console.print(snapshot_table)


def print_report(
    console: Console,
    *,
    model_name: str,
    lora_config: LoraConfig | None,
    int8_quantized_modules: int | None,
    batch_size: int,
    seq_length: int,
    turns_requested: int,
    prompt_set: str,
    wait_for: int,
    totals: BenchmarkTotals,
    turn_metrics: list[TurnMetrics],
    memory: MemoryBreakdown,
    profile_dir: str | None,
    profile_perfetto_trace: bool,
    profile_host_tracer_level: int,
    profile_python_tracer_level: int,
) -> None:
    lora_label = (
        "off"
        if lora_config is None
        else f"rank={lora_config.rank}, attn={lora_config.attn}, mlp={lora_config.mlp}"
    )
    int8_label = (
        "off"
        if int8_quantized_modules is None
        else f"qwix PTQ matched ops={int8_quantized_modules}"
    )
    console.print(
        f"Model: {model_name} | LoRA: {lora_label} | Int8: {int8_label} | "
        f"batch={batch_size} | seq={seq_length} | "
        f"turns={totals.turns}/{turns_requested} | prompts={prompt_set} | wait_for={wait_for}"
    )
    if totals.warmup_wall_s > 0:
        console.print(
            f"Warmup: {totals.warmup_wall_s:.3f}s wall, "
            f"{totals.warmup_jit_s:.3f}s in jitted generate"
        )

    summary = Table(title="Throughput")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Model tokens processed", f"{totals.model_tokens}")
    summary.add_row("Generated context tokens", f"{totals.generated_tokens}")
    summary.add_row("Prompt tokens appended", f"{totals.prompt_tokens}")
    summary.add_row("Truncated prompt tokens", f"{totals.truncated_prompt_tokens}")
    summary.add_row("Completed responses decoded", f"{totals.completed}")
    summary.add_row("Measured wall seconds", f"{totals.wall_s:.6f}")
    summary.add_row("End-to-end model tok/s", f"{totals.end_to_end_model_tps:.2f}")
    summary.add_row("Jitted generate model tok/s", f"{totals.jit_model_tps:.2f}")
    summary.add_row(
        "End-to-end generated tok/s", f"{totals.end_to_end_generated_tps:.2f}"
    )
    console.print(summary)

    breakdown = Table(title="Time Breakdown")
    breakdown.add_column("Section")
    breakdown.add_column("Seconds", justify="right")
    breakdown.add_column("Wall %", justify="right")
    _add_time_row(breakdown, "tokenize prompts", totals.tokenize_s, totals.wall_s)
    _add_time_row(
        breakdown, "append prompts in NumPy", totals.append_np_s, totals.wall_s
    )
    _add_time_row(breakdown, "NumPy -> JAX transfer", totals.np_to_jax_s, totals.wall_s)
    _add_time_row(breakdown, "jitted generate", totals.jit_s, totals.wall_s)
    _add_time_row(
        breakdown, "metrics scalar sync", totals.metrics_sync_s, totals.wall_s
    )
    _add_time_row(breakdown, "JAX -> NumPy transfer", totals.jax_to_np_s, totals.wall_s)
    _add_time_row(breakdown, "decode responses", totals.decode_s, totals.wall_s)
    unaccounted = max(totals.wall_s - totals.accounted_s, 0.0)
    _add_time_row(breakdown, "other Python overhead", unaccounted, totals.wall_s)
    console.print(breakdown)

    turns = Table(title="Per Turn")
    turns.add_column("Turn", justify="right")
    turns.add_column("Prompt")
    turns.add_column("Prompt Tok", justify="right")
    turns.add_column("Trunc", justify="right")
    turns.add_column("Model Tok", justify="right")
    turns.add_column("Gen Tok", justify="right")
    turns.add_column("JIT s", justify="right")
    turns.add_column("JIT tok/s", justify="right")
    turns.add_column("np->jax s", justify="right")
    turns.add_column("jax->np s", justify="right")
    for metrics in turn_metrics:
        jit_tps = metrics.model_tokens / metrics.jit_s if metrics.jit_s > 0 else 0.0
        turns.add_row(
            str(metrics.turn),
            metrics.prompt_kind,
            str(metrics.prompt_tokens),
            str(metrics.truncated_prompt_tokens),
            str(metrics.model_tokens),
            str(metrics.generated_tokens),
            f"{metrics.jit_s:.6f}",
            f"{jit_tps:.2f}",
            f"{metrics.np_to_jax_s:.6f}",
            f"{metrics.jax_to_np_s:.6f}",
        )
    console.print(turns)

    if totals.stopped_early:
        console.print(
            "Stopped early because every batch row reached the sequence limit."
        )

    print_memory_report(console, memory)

    if profile_dir is not None:
        console.print(
            f"JAX profiler trace written to {profile_dir} "
            f"(perfetto_trace={profile_perfetto_trace}, "
            f"host_tracer_level={profile_host_tracer_level}, "
            f"python_tracer_level={profile_python_tracer_level})."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the vaml.chat multiturn JAX generation path."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model path under base-models/.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA adapters on both attention and MLP projections.",
    )
    parser.add_argument(
        "--lora-attn",
        action="store_true",
        help="Enable LoRA adapters on attention projections.",
    )
    parser.add_argument(
        "--lora-mlp",
        action="store_true",
        help="Enable LoRA adapters on MLP projections.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA adapter rank used when LoRA is enabled.",
    )
    parser.add_argument(
        "--int8",
        "--quantize-int8",
        action="store_true",
        dest="int8",
        help=(
            "Quantize model operations to int8 weights and activations with Qwix PTQ."
        ),
    )
    parser.add_argument(
        "--prompt-set",
        choices=("mixed", "short", "long"),
        default="mixed",
        help="Prompt schedule. mixed alternates short and long turns.",
    )
    parser.add_argument(
        "--wait-for",
        type=int,
        default=None,
        help="Rows that must finish before generate returns. Defaults to batch size.",
    )
    parser.add_argument(
        "--warmup-turns",
        type=int,
        default=1,
        help="Warmup turns run before measurement to trigger JIT compilation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Skip response decoding in the measured loop.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable metrics after the human-readable report.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Write a JAX profiler trace for the measured turns to this directory.",
    )
    parser.add_argument(
        "--profile-perfetto-trace",
        action="store_true",
        help="Also emit a Perfetto trace file in --profile-dir.",
    )
    parser.add_argument(
        "--profile-host-tracer-level",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help=(
            "JAX host tracer level. 1 keeps user annotations with much less noise; "
            "2 is JAX's verbose default."
        ),
    )
    parser.add_argument(
        "--profile-python-tracer-level",
        type=int,
        default=0,
        choices=(0, 1),
        help="JAX Python tracer level. 0 avoids huge Python stack traces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.seq_length < 2:
        raise ValueError("--seq-length must be at least 2")
    if args.turns < 1:
        raise ValueError("--turns must be at least 1")
    if args.warmup_turns < 0:
        raise ValueError("--warmup-turns must be non-negative")
    lora_config = _resolve_lora_config(args)

    wait_for = args.wait_for if args.wait_for is not None else args.batch_size
    if wait_for < 1 or wait_for > args.batch_size:
        raise ValueError("--wait-for must be between 1 and --batch-size")

    console = Console()
    rngs = nnx.Rngs(args.seed)
    model, tokenizer, _ = load_base_model(args.model, rngs)
    if lora_config is not None:
        model.initialize_lora(lora_config, rngs=rngs)
    int8_quantized_modules = None
    if args.int8:
        model, int8_quantized_modules = _quantize_int8_model(
            model,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            seed=args.seed,
        )

    totals, turn_metrics, memory = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        turns=args.turns,
        prompt_set=args.prompt_set,
        wait_for=wait_for,
        warmup_turns=args.warmup_turns,
        seed=args.seed,
        decode=not args.no_decode,
        profile_dir=args.profile_dir,
        profile_perfetto_trace=args.profile_perfetto_trace,
        profile_host_tracer_level=args.profile_host_tracer_level,
        profile_python_tracer_level=args.profile_python_tracer_level,
    )

    print_report(
        console,
        model_name=args.model,
        lora_config=lora_config,
        int8_quantized_modules=int8_quantized_modules,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        turns_requested=args.turns,
        prompt_set=args.prompt_set,
        wait_for=wait_for,
        totals=totals,
        turn_metrics=turn_metrics,
        memory=memory,
        profile_dir=args.profile_dir,
        profile_perfetto_trace=args.profile_perfetto_trace,
        profile_host_tracer_level=args.profile_host_tracer_level,
        profile_python_tracer_level=args.profile_python_tracer_level,
    )

    if args.json:
        console.print(
            json.dumps(
                {
                    "config": {
                        "model": args.model,
                        "batch_size": args.batch_size,
                        "seq_length": args.seq_length,
                        "turns": args.turns,
                        "lora": (
                            None
                            if lora_config is None
                            else {
                                "attn": lora_config.attn,
                                "mlp": lora_config.mlp,
                                "rank": lora_config.rank,
                            }
                        ),
                        "int8": args.int8,
                        "int8_quantized_modules": int8_quantized_modules,
                        "prompt_set": args.prompt_set,
                        "wait_for": wait_for,
                        "warmup_turns": args.warmup_turns,
                        "seed": args.seed,
                        "decode": not args.no_decode,
                        "profile_dir": args.profile_dir,
                        "profile_perfetto_trace": args.profile_perfetto_trace,
                        "profile_host_tracer_level": args.profile_host_tracer_level,
                        "profile_python_tracer_level": args.profile_python_tracer_level,
                    },
                    "totals": asdict(totals)
                    | {
                        "end_to_end_model_tps": totals.end_to_end_model_tps,
                        "jit_model_tps": totals.jit_model_tps,
                        "end_to_end_generated_tps": totals.end_to_end_generated_tps,
                    },
                    "turns": [asdict(metrics) for metrics in turn_metrics],
                    "memory": asdict(memory)
                    | {
                        "array_tracked_total_bytes": (
                            memory.array_bytes.tracked_total_bytes
                        ),
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
