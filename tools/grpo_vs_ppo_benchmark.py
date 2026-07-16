#!/usr/bin/env python
"""Benchmark the GRPO update step against the PPO update step.

One "batch" is a single policy update over ``update_envs`` episodes, driven
through the exact code path the training loop uses:

  * GRPO -> ``valm.update_step.grpo.multi_grpo_update_bucketed`` (policy only)
  * PPO  -> ``valm.update_step.ppo.multi_update_step_bucket``    (policy + value)

Both apply gradient accumulation internally (``gradient_accumulations`` chunks
of the batch). We feed real Wordle rollouts saved under ``old-results/rollouts``
and report throughput as batches per second.

Run with the repo venv, e.g.::

    .venv/bin/python tools/grpo_vs_ppo_benchmark.py --config configs/test.json

The script defaults every dimension (model, update_envs, gradient
accumulation, optimizers, value net) from the supplied config so the numbers
reflect a realistic update.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the package importable when run straight from a checkout (mirrors the
# pytest `pythonpath = ["python"]` setting).
REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import jax  # noqa: E402
from flax import nnx  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from valm.buffer import UpdateBatch  # noqa: E402
from valm.base_model_loader import load_base_model  # noqa: E402
from valm.config import Config, GRPOLossConfig, load_config  # noqa: E402
from valm.model.value_network import ValueParam  # noqa: E402
from valm.update_step.grpo import multi_grpo_update_bucketed  # noqa: E402
from valm.update_step.ppo import multi_update_step_bucket  # noqa: E402
from valm.utils.optimizer import make_optimizer  # noqa: E402


@dataclass
class AlgoResult:
    name: str
    measured_steps: int
    wall_s: float
    per_step_s: list[float] = field(default_factory=list)

    @property
    def batches_per_s(self) -> float:
        return self.measured_steps / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def ms_per_batch(self) -> float:
        return 1000.0 * self.wall_s / self.measured_steps if self.measured_steps else 0.0

    def stats_ms(self) -> tuple[float, float, float, float]:
        if not self.per_step_s:
            return (0.0, 0.0, 0.0, 0.0)
        ms = [s * 1000.0 for s in self.per_step_s]
        std = statistics.pstdev(ms) if len(ms) > 1 else 0.0
        return (statistics.mean(ms), std, min(ms), max(ms))


def load_benchmark_config(path: Path) -> tuple[Config, GRPOLossConfig]:
    """Load a training config, tolerating ``test.json`` whose loss block has no
    discriminator tag, and derive a matching GRPO loss for the same clip range.
    """
    raw = json.loads(path.read_text())
    loss = dict(raw.get("loss", {}))

    # PPO drives the full update (policy + critic); fill any missing PPO knobs
    # with the standard defaults so the config validates regardless of which
    # flavor of loss block the file shipped with.
    ppo_fields = {
        "type": "ppo",
        "gae_lambda": loss.get("gae_lambda", 0.95),
        "gae_discount": loss.get("gae_discount", 1.0),
        "turn_lambda": loss.get("turn_lambda", 0.97),
        "turn_discount": loss.get("turn_discount", 0.97),
        "pg_clip_high": loss.get("pg_clip_high", 0.28),
        "pg_clip_low": loss.get("pg_clip_low", 0.2),
        "entropy_coef": loss.get("entropy_coef"),
        "is_correction": loss.get("is_correction", True),
    }
    raw["loss"] = ppo_fields
    config = load_config(json.dumps(raw))

    grpo_loss = GRPOLossConfig(
        pg_clip_low=ppo_fields["pg_clip_low"],
        pg_clip_high=ppo_fields["pg_clip_high"],
        entropy_coef=ppo_fields["entropy_coef"],
        is_correction=ppo_fields["is_correction"],
    )
    return config, grpo_loss


def load_realistic_batches(
    rollout_dir: Path, update_envs: int, n_batches: int
) -> list[UpdateBatch]:
    """Slice the first ``update_envs`` episodes out of each of ``n_batches``
    rollout .npz files. Saved metrics are dropped so each batch looks like a
    fresh rollout handed to the trainer.
    """
    paths = sorted(rollout_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz rollouts found in {rollout_dir}")

    batches: list[UpdateBatch] = []
    for path in paths:
        if len(batches) >= n_batches:
            break
        with np.load(path, allow_pickle=False) as d:
            if d["context"].shape[0] < update_envs:
                continue
            sl = slice(update_envs)
            batches.append(
                UpdateBatch(
                    context_length=d["context_length"][sl].astype(np.int32),
                    context=d["context"][sl].astype(np.int32),
                    log_probs=d["log_probs"][sl].astype(np.float32),
                    rewards=d["rewards"][sl].astype(np.float32),
                    policy_mask=d["policy_mask"][sl].astype(np.bool_),
                    turn_counts=d["turn_counts"][sl].astype(np.int32),
                    turn_start_positions=d["turn_start_positions"][sl].astype(np.int32),
                )
            )

    if not batches:
        raise ValueError(
            f"no rollout file in {rollout_dir} has at least {update_envs} episodes"
        )
    return batches


def _block(tree) -> None:
    jax.block_until_ready(tree)


def run_grpo(
    *,
    model_def,
    model_state,
    policy_def,
    policy_state,
    rng_key,
    batches: list[UpdateBatch],
    grpo_loss: GRPOLossConfig,
    steps: int,
    group_size: int,
    measured_steps: int,
    console: Console,
):
    def one(batch, state):
        policy_state_, model_state_, _, _, rng_ = multi_grpo_update_bucketed(
            policy_def,
            state[0],
            model_def,
            state[1],
            state[2],
            batch,
            grpo_loss,
            steps,
            group_size,
        )
        _block((policy_state_, model_state_))
        return (policy_state_, model_state_, rng_)

    state = (policy_state, model_state, rng_key)

    console.print("[grpo] warmup (compiling)...")
    for batch in batches:
        state = one(batch, state)

    console.print(f"[grpo] measuring {measured_steps} batches...")
    per_step = []
    t0 = time.perf_counter()
    for i in range(measured_steps):
        s = time.perf_counter()
        state = one(batches[i % len(batches)], state)
        per_step.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0

    result = AlgoResult("GRPO", measured_steps, wall, per_step)
    # Return the (mutated) model state so PPO can continue from it.
    return result, state[1]


def run_ppo(
    *,
    model_def,
    model_state,
    policy_def,
    policy_state,
    value_def,
    value_state,
    rng_key,
    batches: list[UpdateBatch],
    ppo_loss,
    steps: int,
    measured_steps: int,
    console: Console,
):
    def one(batch, state):
        (
            policy_state_,
            value_state_,
            model_state_,
            _,
            _,
            rng_,
        ) = multi_update_step_bucket(
            policy_def,
            state[0],
            value_def,
            state[1],
            model_def,
            state[2],
            state[3],
            batch,
            ppo_loss,
            steps,
            False,
        )
        _block((policy_state_, value_state_, model_state_))
        return (policy_state_, value_state_, model_state_, rng_)

    state = (policy_state, value_state, model_state, rng_key)

    console.print("[ppo] warmup (compiling)...")
    for batch in batches:
        state = one(batch, state)

    console.print(f"[ppo] measuring {measured_steps} batches...")
    per_step = []
    t0 = time.perf_counter()
    for i in range(measured_steps):
        s = time.perf_counter()
        state = one(batches[i % len(batches)], state)
        per_step.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0

    return AlgoResult("PPO", measured_steps, wall, per_step), state[2]


def describe_buckets(batches: list[UpdateBatch], chunk_size: int) -> tuple[int, int]:
    """Report the largest and smallest sequence bucket actually compiled, using
    the same power-of-two rule as ``bucket_chunk``.
    """
    lo, hi = 1 << 30, 0
    for batch in batches:
        n = batch.context.shape[0]
        for start in range(0, n, chunk_size):
            sub = batch.context_length[start : start + chunk_size]
            length = int(sub.max())
            bucket = max(128, 1 << length.bit_length())
            lo = min(lo, bucket)
            hi = max(hi, bucket)
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/test.json")
    parser.add_argument("--rollout-dir", default="old-results/rollouts")
    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="Distinct realistic rollout batches to cycle through.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Measured update steps (batches) per algorithm.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="GRPO group size. Defaults to config.group_size.",
    )
    parser.add_argument(
        "--algos",
        default="grpo,ppo",
        help="Comma list of algorithms to run (grpo, ppo).",
    )
    args = parser.parse_args()

    console = Console()
    config, grpo_loss = load_benchmark_config(Path(args.config))
    ppo_loss = config.loss

    assert config.policy_optimizer is not None, "config.policy_optimizer is required"

    update_envs = config.update_envs
    grad_accum = config.gradient_accumulations or 1
    group_size = args.group_size if args.group_size is not None else config.group_size

    if update_envs % grad_accum != 0:
        raise ValueError(
            f"update_envs ({update_envs}) must be divisible by "
            f"gradient_accumulations ({grad_accum})"
        )
    if update_envs % group_size != 0:
        raise ValueError(
            f"update_envs ({update_envs}) must be divisible by group_size ({group_size})"
        )

    chunk_size = update_envs // grad_accum

    console.print(
        f"[bold]GRPO vs PPO update-step benchmark[/bold]\n"
        f"config={args.config}  model={config.base_model}\n"
        f"update_envs={update_envs}  gradient_accumulations={grad_accum} "
        f"(chunk={chunk_size} episodes)  group_size={group_size}\n"
        f"lora_rank={config.lora.rank}  max_seq_length={config.max_seq_length}"
    )

    batches = load_realistic_batches(
        Path(args.rollout_dir), update_envs, args.batches
    )
    seq_lo, seq_hi = describe_buckets(batches, chunk_size)
    console.print(
        f"loaded {len(batches)} realistic batches from {args.rollout_dir} | "
        f"sequence bucket(s): "
        + (f"{seq_lo}" if seq_lo == seq_hi else f"{seq_lo}..{seq_hi}")
    )

    rngs = nnx.Rngs(int(config.seed))
    console.print("loading base model + value net + lora...")
    model, _, _ = load_base_model(config.base_model, rngs)
    model.initialize_value_net(config.value_net, rngs=rngs)
    model.initialize_lora(config.lora, rngs=rngs)

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]

    # Build every optimizer from the live model module, then split into
    # (graphdef, state) the way the trainers do.
    grpo_policy = make_optimizer(
        model, config.policy_optimizer, config.total_update_episodes,
        grad_accum, nnx.LoRAParam,
    )
    ppo_policy = make_optimizer(
        model, config.policy_optimizer, config.total_update_episodes,
        grad_accum, nnx.LoRAParam,
    )
    value_opt = make_optimizer(
        model, config.value_optimizer, config.total_update_episodes,
        grad_accum, ValueParam,
    )

    grpo_policy_def, grpo_policy_state = nnx.split(grpo_policy)
    ppo_policy_def, ppo_policy_state = nnx.split(ppo_policy)
    value_def, value_state = nnx.split(value_opt)
    model_def, model_state = nnx.split(model)
    _block((grpo_policy_state, ppo_policy_state, value_state, model_state))

    rng_key = jax.random.PRNGKey(int(config.seed))
    results: list[AlgoResult] = []

    if "grpo" in algos:
        grpo_result, model_state = run_grpo(
            model_def=model_def,
            model_state=model_state,
            policy_def=grpo_policy_def,
            policy_state=grpo_policy_state,
            rng_key=rng_key,
            batches=batches,
            grpo_loss=grpo_loss,
            steps=grad_accum,
            group_size=group_size,
            measured_steps=args.steps,
            console=console,
        )
        results.append(grpo_result)

    if "ppo" in algos:
        ppo_result, model_state = run_ppo(
            model_def=model_def,
            model_state=model_state,
            policy_def=ppo_policy_def,
            policy_state=ppo_policy_state,
            value_def=value_def,
            value_state=value_state,
            rng_key=rng_key,
            batches=batches,
            ppo_loss=ppo_loss,
            steps=grad_accum,
            measured_steps=args.steps,
            console=console,
        )
        results.append(ppo_result)

    table = Table(title="Update-step throughput (realistic Wordle rollouts)")
    table.add_column("Algorithm")
    table.add_column("Batches/s", justify="right")
    table.add_column("ms/batch", justify="right")
    table.add_column("Episodes/s", justify="right")
    table.add_column("mean±std ms", justify="right")
    table.add_column("min/max ms", justify="right")
    for r in results:
        mean, std, lo, hi = r.stats_ms()
        table.add_row(
            r.name,
            f"{r.batches_per_s:.3f}",
            f"{r.ms_per_batch:.1f}",
            f"{r.batches_per_s * update_envs:.1f}",
            f"{mean:.0f}±{std:.0f}",
            f"{lo:.0f}/{hi:.0f}",
        )
    console.print(table)

    if len(results) == 2:
        grpo = next(r for r in results if r.name == "GRPO")
        ppo = next(r for r in results if r.name == "PPO")
        if ppo.batches_per_s > 0:
            console.print(
                f"[bold]GRPO is {grpo.batches_per_s / ppo.batches_per_s:.2f}x "
                f"PPO's batches/s[/bold] "
                f"(GRPO {grpo.batches_per_s:.3f} vs PPO {ppo.batches_per_s:.3f})"
            )


if __name__ == "__main__":
    main()
