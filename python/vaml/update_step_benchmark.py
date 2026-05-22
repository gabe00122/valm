import argparse
import gc
import json
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from rich.console import Console
from rich.table import Table
from vaml.base_model_loader import load_base_model
from vaml.buffer import UpdateBatch
from vaml.config import Config, load_config
from vaml.model.value_network import ValueParam
from vaml.update_step import update_step
from vaml.utils.optimizer import make_optimizer


@dataclass
class StepMetrics:
    step: int
    batch_index: int
    episodes: int
    sequence_tokens: int
    value_tokens: int
    policy_tokens: int
    batch_to_device_s: float
    update_s: float
    metrics_sync_s: float
    token_metrics_sync_s: float

    @property
    def accounted_s(self) -> float:
        return (
            self.batch_to_device_s
            + self.update_s
            + self.metrics_sync_s
            + self.token_metrics_sync_s
        )


@dataclass
class BenchmarkTotals:
    steps: int = 0
    episodes: int = 0
    sequence_tokens: int = 0
    value_tokens: int = 0
    policy_tokens: int = 0
    wall_s: float = 0.0
    batch_to_device_s: float = 0.0
    update_s: float = 0.0
    metrics_sync_s: float = 0.0
    token_metrics_sync_s: float = 0.0
    warmup_wall_s: float = 0.0
    warmup_update_s: float = 0.0

    @property
    def accounted_s(self) -> float:
        return (
            self.batch_to_device_s
            + self.update_s
            + self.metrics_sync_s
            + self.token_metrics_sync_s
        )

    @property
    def end_to_end_sequence_tps(self) -> float:
        return self.sequence_tokens / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def update_sequence_tps(self) -> float:
        return self.sequence_tokens / self.update_s if self.update_s > 0 else 0.0

    @property
    def end_to_end_episode_s(self) -> float:
        return self.episodes / self.wall_s if self.wall_s > 0 else 0.0


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
    policy_optimizer_state_bytes: int = 0
    value_optimizer_state_bytes: int = 0
    rollout_batch_bytes: int = 0

    @property
    def tracked_total_bytes(self) -> int:
        return (
            self.model_state_bytes
            + self.policy_optimizer_state_bytes
            + self.value_optimizer_state_bytes
            + self.rollout_batch_bytes
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


def _load_config_file(config_path: str) -> Config:
    return load_config(Path(config_path).read_text())


def _load_update_batch_npz(path: Path) -> UpdateBatch:
    with np.load(path, allow_pickle=False) as data:
        fields: dict[str, Any] = {}
        turn_metrics: dict[str, np.ndarray] = {}
        update_metrics: dict[str, np.ndarray] = {}
        batch_fields = set(UpdateBatch._fields) - {"turn_metrics", "update_metrics"}

        for key, value in data.items():
            if key.startswith("turn_metrics_"):
                turn_metrics[key[len("turn_metrics_") :]] = value
            elif key.startswith("update_metrics_"):
                update_metrics[key[len("update_metrics_") :]] = value
            elif key in batch_fields:
                fields[key] = value

        return UpdateBatch(
            turn_metrics=turn_metrics,
            update_metrics=update_metrics,
            **fields,
        )


def _slice_mapping(
    mapping: Mapping[str, np.ndarray], batch_size: int | None
) -> dict[str, np.ndarray]:
    if batch_size is None:
        return dict(mapping)
    return {name: value[:batch_size] for name, value in mapping.items()}


def _slice_batch(
    batch: UpdateBatch,
    *,
    batch_size: int | None,
    seq_length: int | None,
) -> UpdateBatch:
    context_length = batch.context_length
    context = batch.context
    log_probs = batch.log_probs
    rewards = batch.rewards
    policy_mask = batch.policy_mask
    turn_counts = batch.turn_counts
    turn_start_positions = batch.turn_start_positions

    if batch_size is not None:
        context_length = context_length[:batch_size]
        context = context[:batch_size]
        log_probs = log_probs[:batch_size]
        rewards = rewards[:batch_size]
        policy_mask = policy_mask[:batch_size]
        turn_counts = turn_counts[:batch_size]
        turn_start_positions = turn_start_positions[:batch_size]

    if seq_length is not None:
        if seq_length < 2:
            raise ValueError("--seq-length must be at least 2")
        if seq_length > context.shape[1]:
            raise ValueError(
                f"--seq-length {seq_length} exceeds rollout sequence length "
                f"{context.shape[1]}"
            )
        context = context[:, :seq_length]
        log_probs = log_probs[:, : seq_length - 1]
        rewards = rewards[:, :seq_length]
        policy_mask = policy_mask[:, :seq_length]
        context_length = np.minimum(context_length, seq_length).astype(np.int32)
        bounds = np.arange(seq_length, dtype=np.int32)[None, :] < context_length[:, None]
        policy_mask = policy_mask & bounds

    if context.shape[1] < 2:
        raise ValueError("rollout sequence length must be at least 2")

    return UpdateBatch(
        context_length=context_length.astype(np.int32, copy=False),
        context=context.astype(np.int32, copy=False),
        log_probs=log_probs.astype(np.float32, copy=False),
        rewards=rewards.astype(np.float32, copy=False),
        policy_mask=policy_mask.astype(np.bool_, copy=False),
        turn_counts=turn_counts.astype(np.int32, copy=False),
        turn_start_positions=turn_start_positions.astype(np.int32, copy=False),
        turn_metrics=_slice_mapping(batch.turn_metrics, batch_size),
        update_metrics=_slice_mapping(batch.update_metrics, batch_size),
    )


def _load_rollout_batches(
    *,
    rollout_npz: list[str] | None,
    rollout_dir: str | None,
    batch_size: int | None,
    seq_length: int | None,
) -> list[UpdateBatch]:
    paths: list[Path] = []
    if rollout_dir is not None:
        paths.extend(sorted(Path(rollout_dir).glob("*.npz")))
    if rollout_npz is not None:
        paths.extend(Path(path) for path in rollout_npz)

    batches = []
    for path in paths:
        batches.append(
            _slice_batch(
                _load_update_batch_npz(path),
                batch_size=batch_size,
                seq_length=seq_length,
            )
        )
    return batches


def _make_synthetic_batch(
    *,
    batch_size: int,
    seq_length: int,
    max_turns: int,
    vocab_size: int,
    context_length: int,
    seed: int,
) -> UpdateBatch:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if seq_length < 2:
        raise ValueError("--seq-length must be at least 2")
    if max_turns < 1:
        raise ValueError("--max-turns must be at least 1")
    if context_length < 2 or context_length > seq_length:
        raise ValueError("--context-length must be between 2 and --seq-length")
    if vocab_size < 2:
        raise ValueError("model vocabulary size must be at least 2")

    rng = np.random.default_rng(seed)
    context = rng.integers(
        0,
        vocab_size,
        size=(batch_size, seq_length),
        dtype=np.int32,
    )
    context_length_arr = np.full((batch_size,), context_length, dtype=np.int32)
    bounds = np.arange(seq_length, dtype=np.int32)[None, :] < context_length
    context = np.where(bounds, context, 0).astype(np.int32)

    policy_mask = bounds.repeat(batch_size, axis=0)
    policy_mask[:, 0] = False

    log_probs = np.full(
        (batch_size, seq_length - 1),
        -np.log(float(vocab_size)),
        dtype=np.float32,
    )

    rewards = np.zeros((batch_size, seq_length), dtype=np.float32)
    rewards[:, context_length - 1] = rng.uniform(-0.1, 1.0, size=batch_size).astype(
        np.float32
    )

    turn_counts = np.full((batch_size,), max_turns, dtype=np.int32)
    turn_start_positions = np.zeros((batch_size, max_turns), dtype=np.int32)
    for row in range(batch_size):
        turn_start_positions[row] = np.linspace(
            0,
            max(context_length - 1, 0),
            max_turns,
            dtype=np.int32,
        )

    return UpdateBatch(
        context_length=context_length_arr,
        context=context,
        log_probs=log_probs,
        rewards=rewards,
        policy_mask=policy_mask,
        turn_counts=turn_counts,
        turn_start_positions=turn_start_positions,
        turn_metrics={
            "reward": rewards[:, context_length - 1 : context_length],
        },
    )


def _batch_to_device(batch: UpdateBatch) -> UpdateBatch:
    return jax.tree.map(lambda x: jnp.asarray(x), batch)


def _batch_counts(batch: UpdateBatch) -> tuple[int, int, int, int]:
    batch_size, seq_length = batch.context.shape
    bounds = np.arange(seq_length, dtype=np.int32)[None, :] < batch.context_length[:, None]
    policy_mask = batch.policy_mask[:, :-1] & bounds[:, :-1]
    return (
        batch_size,
        batch_size * seq_length,
        int(bounds.sum()),
        int(policy_mask.sum()),
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return _json_scalar(value)


def _flatten_summary_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    rows = []
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(_flatten_summary_metrics(value, prefix=full_key))
        else:
            rows.append((full_key, value))
    return rows


def _run_one_step(
    *,
    policy_opt_def: Any,
    policy_opt_state: Any,
    value_opt_def: Any,
    value_opt_state: Any,
    model_def: Any,
    model_state: Any,
    rng_key: jax.Array,
    batch: UpdateBatch,
    config: Config,
    value_only: bool,
    sync_token_metrics: bool,
    step: int,
    batch_index: int,
    profile_enabled: bool,
) -> tuple[Any, Any, Any, jax.Array, StepMetrics, dict[str, Any]]:
    with _trace_annotation("update_step_batch_to_device", profile_enabled):
        start = time.perf_counter()
        device_batch = _batch_to_device(batch)
        _block_until_ready(device_batch)
        batch_to_device_s = time.perf_counter() - start

    with _trace_annotation("jitted_update_step", profile_enabled):
        start = time.perf_counter()
        (
            policy_opt_state,
            value_opt_state,
            model_state,
            summary_metrics,
            token_metrics,
            rng_key,
        ) = update_step(
            policy_opt_def,
            policy_opt_state,
            value_opt_def,
            value_opt_state,
            model_def,
            model_state,
            rng_key,
            device_batch,
            config.loss,
            value_only,
        )
        _block_until_ready((policy_opt_state, value_opt_state, model_state, rng_key))
        update_s = time.perf_counter() - start

    with _trace_annotation("summary_metrics_sync", profile_enabled):
        start = time.perf_counter()
        summary_host = jax.device_get(summary_metrics)
        metrics_sync_s = time.perf_counter() - start

    with _trace_annotation("token_metrics_sync", profile_enabled):
        start = time.perf_counter()
        if sync_token_metrics:
            jax.device_get(token_metrics)
        token_metrics_sync_s = time.perf_counter() - start

    episodes, sequence_tokens, value_tokens, policy_tokens = _batch_counts(batch)
    metrics = StepMetrics(
        step=step,
        batch_index=batch_index,
        episodes=episodes,
        sequence_tokens=sequence_tokens,
        value_tokens=value_tokens,
        policy_tokens=policy_tokens,
        batch_to_device_s=batch_to_device_s,
        update_s=update_s,
        metrics_sync_s=metrics_sync_s,
        token_metrics_sync_s=token_metrics_sync_s,
    )
    return (
        policy_opt_state,
        value_opt_state,
        model_state,
        rng_key,
        metrics,
        _jsonify(summary_host),
    )


def run_benchmark(
    *,
    model: Any,
    config: Config,
    batches: list[UpdateBatch],
    steps: int,
    warmup_steps: int,
    seed: int,
    total_optimizer_steps: int,
    value_only: bool,
    sync_token_metrics: bool,
    profile_dir: str | None,
    profile_perfetto_trace: bool,
    profile_host_tracer_level: int,
    profile_python_tracer_level: int,
) -> tuple[BenchmarkTotals, list[StepMetrics], MemoryBreakdown, dict[str, Any]]:
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if not batches:
        raise ValueError("at least one rollout batch is required")

    rng_key = jax.random.PRNGKey(seed)

    if not value_only:
        if config.policy_optimizer is None:
            raise ValueError(
                "config.policy_optimizer is required unless --value-only is set"
            )
        policy_opt = make_optimizer(
            model,
            config.policy_optimizer,
            total_optimizer_steps,
            nnx.LoRAParam,
        )
        policy_opt_def, policy_opt_state = nnx.split(policy_opt)
    else:
        policy_opt_def = None
        policy_opt_state = None

    value_opt = make_optimizer(
        model,
        config.value_optimizer,
        total_optimizer_steps,
        ValueParam,
    )
    value_opt_def, value_opt_state = nnx.split(value_opt)
    model_def, model_state = nnx.split(model)
    _block_until_ready((policy_opt_state, value_opt_state, model_state))

    memory = MemoryBreakdown(
        array_bytes=ArrayMemoryBreakdown(
            model_state_bytes=_tree_nbytes(model_state),
            policy_optimizer_state_bytes=_tree_nbytes(policy_opt_state),
            value_optimizer_state_bytes=_tree_nbytes(value_opt_state),
            rollout_batch_bytes=_tree_nbytes(batches[0]),
        ),
        snapshots=_device_memory_snapshots("after_model_and_optimizer_split"),
    )

    warmup_totals = BenchmarkTotals()
    if warmup_steps > 0:
        start_wall = time.perf_counter()
        for warmup_step in range(warmup_steps):
            batch_index = warmup_step % len(batches)
            (
                policy_opt_state,
                value_opt_state,
                model_state,
                rng_key,
                metrics,
                _,
            ) = _run_one_step(
                policy_opt_def=policy_opt_def,
                policy_opt_state=policy_opt_state,
                value_opt_def=value_opt_def,
                value_opt_state=value_opt_state,
                model_def=model_def,
                model_state=model_state,
                rng_key=rng_key,
                batch=batches[batch_index],
                config=config,
                value_only=value_only,
                sync_token_metrics=sync_token_metrics,
                step=warmup_step,
                batch_index=batch_index,
                profile_enabled=False,
            )
            warmup_totals.update_s += metrics.update_s
        warmup_totals.wall_s = time.perf_counter() - start_wall
        memory.snapshots.extend(_device_memory_snapshots("after_warmup"))
        gc.collect()

    totals = BenchmarkTotals(
        warmup_wall_s=warmup_totals.wall_s,
        warmup_update_s=warmup_totals.update_s,
    )
    step_metrics: list[StepMetrics] = []
    last_summary_metrics: dict[str, Any] = {}
    profile_enabled = profile_dir is not None

    with _profile_trace(
        profile_dir,
        create_perfetto_trace=profile_perfetto_trace,
        host_tracer_level=profile_host_tracer_level,
        python_tracer_level=profile_python_tracer_level,
    ):
        start_wall = time.perf_counter()
        for step in range(steps):
            batch_index = step % len(batches)
            with _step_annotation("update_step_benchmark", step, profile_enabled):
                (
                    policy_opt_state,
                    value_opt_state,
                    model_state,
                    rng_key,
                    metrics,
                    last_summary_metrics,
                ) = _run_one_step(
                    policy_opt_def=policy_opt_def,
                    policy_opt_state=policy_opt_state,
                    value_opt_def=value_opt_def,
                    value_opt_state=value_opt_state,
                    model_def=model_def,
                    model_state=model_state,
                    rng_key=rng_key,
                    batch=batches[batch_index],
                    config=config,
                    value_only=value_only,
                    sync_token_metrics=sync_token_metrics,
                    step=step,
                    batch_index=batch_index,
                    profile_enabled=profile_enabled,
                )

            step_metrics.append(metrics)
            totals.steps += 1
            totals.episodes += metrics.episodes
            totals.sequence_tokens += metrics.sequence_tokens
            totals.value_tokens += metrics.value_tokens
            totals.policy_tokens += metrics.policy_tokens
            totals.batch_to_device_s += metrics.batch_to_device_s
            totals.update_s += metrics.update_s
            totals.metrics_sync_s += metrics.metrics_sync_s
            totals.token_metrics_sync_s += metrics.token_metrics_sync_s

    totals.wall_s = time.perf_counter() - start_wall
    memory.snapshots.extend(_device_memory_snapshots("after_benchmark"))
    return totals, step_metrics, memory, last_summary_metrics


def _add_time_row(table: Table, name: str, seconds: float, wall_s: float) -> None:
    pct = (seconds / wall_s * 100.0) if wall_s > 0 else 0.0
    table.add_row(name, f"{seconds:.6f}", f"{pct:.1f}%")


def print_memory_report(console: Console, memory: MemoryBreakdown) -> None:
    array_table = Table(title="Tracked JAX Array VRAM")
    array_table.add_column("Category")
    array_table.add_column("Bytes", justify="right")
    array_table.add_column("Size", justify="right")
    array_table.add_row(
        "model state",
        str(memory.array_bytes.model_state_bytes),
        _format_bytes(memory.array_bytes.model_state_bytes),
    )
    array_table.add_row(
        "policy optimizer state",
        str(memory.array_bytes.policy_optimizer_state_bytes),
        _format_bytes(memory.array_bytes.policy_optimizer_state_bytes),
    )
    array_table.add_row(
        "value optimizer state",
        str(memory.array_bytes.value_optimizer_state_bytes),
        _format_bytes(memory.array_bytes.value_optimizer_state_bytes),
    )
    array_table.add_row(
        "rollout batch",
        str(memory.array_bytes.rollout_batch_bytes),
        _format_bytes(memory.array_bytes.rollout_batch_bytes),
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
    config_path: str,
    model_name: str,
    value_only: bool,
    batch_count: int,
    batch_size: int,
    seq_length: int,
    steps_requested: int,
    warmup_steps: int,
    sync_token_metrics: bool,
    totals: BenchmarkTotals,
    step_metrics: list[StepMetrics],
    memory: MemoryBreakdown,
    summary_metrics: Mapping[str, Any],
    profile_dir: str | None,
    profile_perfetto_trace: bool,
    profile_host_tracer_level: int,
    profile_python_tracer_level: int,
) -> None:
    console.print(
        f"Model: {model_name} | config={config_path} | mode="
        f"{'value-only' if value_only else 'policy+value'} | "
        f"batch={batch_size} | seq={seq_length} | "
        f"steps={totals.steps}/{steps_requested} | rollout_batches={batch_count}"
    )
    if totals.warmup_wall_s > 0:
        console.print(
            f"Warmup: {warmup_steps} steps, {totals.warmup_wall_s:.3f}s wall, "
            f"{totals.warmup_update_s:.3f}s in jitted update_step"
        )

    summary = Table(title="Throughput")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Update steps", f"{totals.steps}")
    summary.add_row("Episodes", f"{totals.episodes}")
    summary.add_row("Full sequence tokens", f"{totals.sequence_tokens}")
    summary.add_row("Context value tokens", f"{totals.value_tokens}")
    summary.add_row("Policy loss tokens", f"{totals.policy_tokens}")
    summary.add_row("Measured wall seconds", f"{totals.wall_s:.6f}")
    summary.add_row(
        "End-to-end sequence tok/s", f"{totals.end_to_end_sequence_tps:.2f}"
    )
    summary.add_row("Jitted update sequence tok/s", f"{totals.update_sequence_tps:.2f}")
    summary.add_row("End-to-end episodes/s", f"{totals.end_to_end_episode_s:.2f}")
    console.print(summary)

    breakdown = Table(title="Time Breakdown")
    breakdown.add_column("Section")
    breakdown.add_column("Seconds", justify="right")
    breakdown.add_column("Wall %", justify="right")
    _add_time_row(
        breakdown, "batch host -> device", totals.batch_to_device_s, totals.wall_s
    )
    _add_time_row(breakdown, "jitted update_step", totals.update_s, totals.wall_s)
    _add_time_row(breakdown, "summary metrics sync", totals.metrics_sync_s, totals.wall_s)
    if sync_token_metrics:
        _add_time_row(
            breakdown,
            "token metrics sync",
            totals.token_metrics_sync_s,
            totals.wall_s,
        )
    unaccounted = max(totals.wall_s - totals.accounted_s, 0.0)
    _add_time_row(breakdown, "other Python overhead", unaccounted, totals.wall_s)
    console.print(breakdown)

    steps = Table(title="Per Step")
    steps.add_column("Step", justify="right")
    steps.add_column("Batch", justify="right")
    steps.add_column("Seq Tok", justify="right")
    steps.add_column("Value Tok", justify="right")
    steps.add_column("Policy Tok", justify="right")
    steps.add_column("H2D s", justify="right")
    steps.add_column("Update s", justify="right")
    steps.add_column("Update tok/s", justify="right")
    steps.add_column("Metric s", justify="right")
    for metrics in step_metrics:
        update_tps = (
            metrics.sequence_tokens / metrics.update_s if metrics.update_s > 0 else 0.0
        )
        steps.add_row(
            str(metrics.step),
            str(metrics.batch_index),
            str(metrics.sequence_tokens),
            str(metrics.value_tokens),
            str(metrics.policy_tokens),
            f"{metrics.batch_to_device_s:.6f}",
            f"{metrics.update_s:.6f}",
            f"{update_tps:.2f}",
            f"{metrics.metrics_sync_s:.6f}",
        )
    console.print(steps)

    if summary_metrics:
        metrics_table = Table(title="Last Summary Metrics")
        metrics_table.add_column("Metric")
        metrics_table.add_column("Value", justify="right")
        for key, value in _flatten_summary_metrics(summary_metrics):
            if isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            metrics_table.add_row(key, rendered)
        console.print(metrics_table)

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
        description="Benchmark the vaml.update_step JAX training path."
    )
    parser.add_argument(
        "--config",
        default="configs/test.json",
        help="Training config used for model heads, optimizers, and loss.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override config.base_model with a model path under base-models/.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Synthetic rollout context length. Defaults to --seq-length.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Synthetic rollout turn_start_positions width.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Warmup update steps run before measurement to trigger JIT compilation.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--total-optimizer-steps",
        type=int,
        default=None,
        help="Optimizer schedule length. Defaults to config.total_update_episodes.",
    )
    parser.add_argument(
        "--value-only",
        action="store_true",
        help="Benchmark the value-only update path used by train_value.",
    )
    parser.add_argument(
        "--rollout-npz",
        action="append",
        default=None,
        help="Use a rollout .npz file instead of synthetic data. Can be repeated.",
    )
    parser.add_argument(
        "--rollout-dir",
        default=None,
        help="Use all .npz rollout files in this directory instead of synthetic data.",
    )
    parser.add_argument(
        "--sync-token-metrics",
        action="store_true",
        help="Also copy token-aligned update metrics back to host each step.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable metrics after the human-readable report.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Write a JAX profiler trace for measured steps to this directory.",
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
    config = _load_config_file(args.config)

    batch_size = args.batch_size if args.batch_size is not None else config.update_envs
    seq_length = args.seq_length if args.seq_length is not None else config.max_seq_length
    context_length = args.context_length if args.context_length is not None else seq_length
    seed = args.seed if args.seed is not None else int(config.seed)
    total_optimizer_steps = (
        args.total_optimizer_steps
        if args.total_optimizer_steps is not None
        else config.total_update_episodes
    )

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if total_optimizer_steps < 1:
        raise ValueError("--total-optimizer-steps must be at least 1")

    console = Console()
    rngs = nnx.Rngs(seed)
    model_name = args.model if args.model is not None else config.base_model
    model, _, _ = load_base_model(model_name, rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)
    if not args.value_only:
        model.initialize_lora(config.lora, rngs=rngs)

    batches = _load_rollout_batches(
        rollout_npz=args.rollout_npz,
        rollout_dir=args.rollout_dir,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    if not batches:
        vocab_size = int(model.embeddings.embedding.shape[0])
        batches = [
            _make_synthetic_batch(
                batch_size=batch_size,
                seq_length=seq_length,
                max_turns=args.max_turns,
                vocab_size=vocab_size,
                context_length=context_length,
                seed=seed,
            )
        ]

    first_batch_size, first_seq_length = batches[0].context.shape
    totals, step_metrics, memory, summary_metrics = run_benchmark(
        model=model,
        config=config,
        batches=batches,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=seed,
        total_optimizer_steps=total_optimizer_steps,
        value_only=args.value_only,
        sync_token_metrics=args.sync_token_metrics,
        profile_dir=args.profile_dir,
        profile_perfetto_trace=args.profile_perfetto_trace,
        profile_host_tracer_level=args.profile_host_tracer_level,
        profile_python_tracer_level=args.profile_python_tracer_level,
    )

    print_report(
        console,
        config_path=args.config,
        model_name=model_name,
        value_only=args.value_only,
        batch_count=len(batches),
        batch_size=first_batch_size,
        seq_length=first_seq_length,
        steps_requested=args.steps,
        warmup_steps=args.warmup_steps,
        sync_token_metrics=args.sync_token_metrics,
        totals=totals,
        step_metrics=step_metrics,
        memory=memory,
        summary_metrics=summary_metrics,
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
                        "config": args.config,
                        "model": model_name,
                        "batch_size": first_batch_size,
                        "seq_length": first_seq_length,
                        "steps": args.steps,
                        "warmup_steps": args.warmup_steps,
                        "seed": seed,
                        "total_optimizer_steps": total_optimizer_steps,
                        "value_only": args.value_only,
                        "rollout_npz": args.rollout_npz,
                        "rollout_dir": args.rollout_dir,
                        "sync_token_metrics": args.sync_token_metrics,
                        "profile_dir": args.profile_dir,
                        "profile_perfetto_trace": args.profile_perfetto_trace,
                        "profile_host_tracer_level": args.profile_host_tracer_level,
                        "profile_python_tracer_level": args.profile_python_tracer_level,
                    },
                    "totals": asdict(totals)
                    | {
                        "end_to_end_sequence_tps": totals.end_to_end_sequence_tps,
                        "update_sequence_tps": totals.update_sequence_tps,
                        "end_to_end_episode_s": totals.end_to_end_episode_s,
                    },
                    "steps": [asdict(metrics) for metrics in step_metrics],
                    "last_summary_metrics": summary_metrics,
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
