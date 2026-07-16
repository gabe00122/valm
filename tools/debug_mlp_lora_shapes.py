"""Debug MlpLayer LoRA shapes against real Qwen3 config files.

Examples:
    uv run python tools/debug_mlp_lora_shapes.py
    uv run python tools/debug_mlp_lora_shapes.py \
        --config base-models/Qwen/Qwen3-4B-Instruct-2507/config.json \
        --ranks 1,8,64 \
        --hlo-dir /tmp/mlp-hlo
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _configure_jax_platform() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("cpu", "gpu", "tpu"))
    args, _ = parser.parse_known_args()
    if args.platform is not None:
        os.environ["JAX_PLATFORM_NAME"] = args.platform


_configure_jax_platform()

import jax  # noqa: E402
from flax import nnx  # noqa: E402
from jax import numpy as jnp  # noqa: E402
from valm.config import LLMConfig, LoraConfig  # noqa: E402
from valm.model.mlp import MlpLayer  # noqa: E402


@dataclass(frozen=True)
class DebugCase:
    name: str
    rank: int | None
    expects_lora_params: bool
    expects_lora_used: bool
    toggle_off_after_init: bool = False


@dataclass(frozen=True)
class AotResult:
    out_shape: tuple[int, ...]
    out_dtype: str
    hlo_text: str
    compiled_text: str | None
    cost_analysis: Any | None
    memory_analysis: Any | None


def _forward(model_def: nnx.graph.GraphDef, model_state: nnx.State, x: jax.Array):
    model: MlpLayer = nnx.merge(model_def, model_state)
    return model(x)


def _required(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None:
        raise ValueError(f"Missing required config field: {key}")
    return value


def load_qwen3_config(path: Path) -> tuple[LLMConfig, dict[str, Any]]:
    data = json.loads(path.read_text())
    hidden_size = _required(data, "hidden_size")
    num_attention_heads = _required(data, "num_attention_heads")

    return (
        LLMConfig(
            embed=hidden_size,
            mlp_ffw_size=_required(data, "intermediate_size"),
            q_heads=num_attention_heads,
            kv_heads=_required(data, "num_key_value_heads"),
            num_layers=_required(data, "num_hidden_layers"),
            head_dim=data.get("head_dim") or hidden_size // num_attention_heads,
            vocab_size=data.get("vocab_size", -1),
            norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=data.get("rope_theta", 500000.0),
        ),
        data,
    )


def find_local_qwen3_configs(root: Path, *, dedupe_shapes: bool) -> list[Path]:
    paths: list[Path] = []
    seen_shapes: set[tuple[Any, ...]] = set()

    for path in sorted(root.glob("**/config.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        if data.get("model_type") != "qwen3":
            continue

        shape_key = (
            data.get("hidden_size"),
            data.get("intermediate_size"),
            data.get("num_attention_heads"),
            data.get("num_key_value_heads"),
            data.get("num_hidden_layers"),
            data.get("head_dim"),
            data.get("vocab_size"),
        )
        if dedupe_shapes and shape_key in seen_shapes:
            continue

        seen_shapes.add(shape_key)
        paths.append(path)

    return paths


def parse_ranks(value: str) -> list[int]:
    ranks: list[int] = []
    if value.strip() == "":
        return ranks

    for part in value.split(","):
        rank = int(part.strip())
        if rank <= 0:
            raise argparse.ArgumentTypeError(
                "LoRA ranks must be positive; use the off case for no LoRA."
            )
        ranks.append(rank)

    return ranks


def input_dtype(name: str):
    if name in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if name in {"f32", "float32"}:
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}")


def shape_text(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def dtype_nbytes(dtype: Any) -> int:
    return jnp.dtype(dtype).itemsize


def bytes_text(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def path_text(path: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in path)


def param_shapes(mlp: MlpLayer) -> dict[str, tuple[tuple[int, ...], str, int, str]]:
    shapes: dict[str, tuple[tuple[int, ...], str, int, str]] = {}
    for path, value in nnx.iter_graph(mlp):
        if not isinstance(value, nnx.Param):
            continue

        shape = tuple(int(dim) for dim in value.shape)
        dtype = str(value.dtype)
        size = 1
        for dim in shape:
            size *= dim

        shapes[path_text(path)] = (
            shape,
            dtype,
            size * dtype_nbytes(value.dtype),
            type(value).__name__,
        )

    return dict(sorted(shapes.items()))


def expected_shapes(config: LLMConfig, *, expects_lora_params: bool, rank: int | None):
    embed = config.embed
    ffw = config.mlp_ffw_size
    expected: dict[str, tuple[int, ...]] = {
        "up_proj.kernel": (embed, ffw * 2),
        "down_proj.kernel": (ffw, embed),
    }

    if expects_lora_params:
        assert rank is not None
        expected.update(
            {
                "up_proj_lora.lora_a": (embed, rank),
                "up_proj_lora.lora_b": (rank, ffw * 2),
                "down_proj_lora.lora_a": (ffw, rank),
                "down_proj_lora.lora_b": (rank, embed),
            }
        )

    return expected


def build_mlp(config: LLMConfig, case: DebugCase, *, seed: int) -> MlpLayer:
    mlp = MlpLayer(config, rngs=nnx.Rngs(seed))
    if case.rank is not None:
        mlp.initialize_lora(
            LoraConfig(mlp=True, attn=False, rank=case.rank),
            rngs=nnx.Rngs(seed + 1),
        )
        if case.toggle_off_after_init:
            mlp.initialize_lora(
                LoraConfig(mlp=False, attn=False, rank=case.rank),
                rngs=nnx.Rngs(seed + 2),
            )
    return mlp


def validate_shapes(
    config: LLMConfig,
    case: DebugCase,
    actual: dict[str, tuple[tuple[int, ...], str, int, str]],
    mlp: MlpLayer,
) -> list[str]:
    errors: list[str] = []
    expected = expected_shapes(
        config, expects_lora_params=case.expects_lora_params, rank=case.rank
    )

    for name, expected_shape in expected.items():
        if name not in actual:
            errors.append(f"missing param {name}; expected {expected_shape}")
            continue

        actual_shape = actual[name][0]
        if actual_shape != expected_shape:
            errors.append(
                f"{name}: expected {expected_shape}, observed {actual_shape}"
            )

    for name in actual:
        if name not in expected:
            errors.append(f"unexpected param {name}: observed {actual[name][0]}")

    if mlp._use_lora != case.expects_lora_used:
        errors.append(
            f"_use_lora expected {case.expects_lora_used}, observed {mlp._use_lora}"
        )

    return errors


def abstract_state(state: nnx.State) -> nnx.State:
    return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)


def run_aot(
    mlp: MlpLayer,
    config: LLMConfig,
    *,
    batch: int,
    seq: int,
    dtype: Any,
    compile_lowered: bool,
) -> AotResult:
    model_def, model_state = nnx.split(mlp)
    x_spec = jax.ShapeDtypeStruct((batch, seq, config.embed), dtype)

    traced = jax.jit(_forward, static_argnames=("model_def",)).trace(
        model_def,
        abstract_state(model_state),
        x_spec,
    )
    lowered = traced.lower()
    compiled_text = None
    cost_analysis = None
    memory_analysis = None

    if compile_lowered:
        compiled = lowered.compile()
        compiled_text = compiled.as_text()
        cost_analysis = compiled.cost_analysis()
        memory_analysis_fn = getattr(compiled, "memory_analysis", None)
        if memory_analysis_fn is not None:
            memory_analysis = memory_analysis_fn()

    out_info = traced.out_info
    return AotResult(
        out_shape=tuple(int(dim) for dim in out_info.shape),
        out_dtype=str(out_info.dtype),
        hlo_text=lowered.as_text(debug_info=True),
        compiled_text=compiled_text,
        cost_analysis=cost_analysis,
        memory_analysis=memory_analysis,
    )


def run_eager(
    mlp: MlpLayer,
    config: LLMConfig,
    *,
    batch: int,
    seq: int,
    dtype: Any,
):
    x = jnp.zeros((batch, seq, config.embed), dtype=dtype)
    y = mlp(x)
    return tuple(int(dim) for dim in y.shape), str(y.dtype)


def interesting_hlo_lines(hlo_text: str, *, limit: int) -> list[str]:
    needles = ("stablehlo.dot_general", "stablehlo.slice", "stablehlo.add")
    lines: list[str] = []
    for line in hlo_text.splitlines():
        if any(needle in line for needle in needles):
            lines.append(line.strip())
        if len(lines) >= limit:
            break
    return lines


def interesting_compiled_lines(compiled_text: str, *, limit: int) -> list[str]:
    needles = ("ENTRY ", "ROOT ", "fusion", "custom-call", "dot(")
    lines: list[str] = []
    for line in compiled_text.splitlines():
        if any(needle in line for needle in needles):
            lines.append(line.strip())
        if len(lines) >= limit:
            break
    return lines


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def print_case(
    config_path: Path,
    config: LLMConfig,
    data: dict[str, Any],
    case: DebugCase,
    actual: dict[str, tuple[tuple[int, ...], str, int, str]],
    errors: list[str],
    aot: AotResult | None,
    eager: tuple[tuple[int, ...], str] | None,
    *,
    show_interesting_hlo: bool,
    interesting_limit: int,
    hlo_lines: int,
    compiled_lines: int,
    show_interesting_compiled: bool,
    hlo_dir: Path | None,
) -> None:
    print(f"\n=== {config_path} :: {case.name} ===")
    print(
        "config: "
        f"model_type={data.get('model_type')}, "
        f"hidden={config.embed}, "
        f"intermediate={config.mlp_ffw_size}, "
        f"layers={config.num_layers}, "
        f"heads={config.q_heads}/{config.kv_heads}, "
        f"head_dim={config.head_dim}"
    )
    print(
        "expected flow: "
        f"input [B,S,{config.embed}] -> "
        f"up_proj [B,S,{config.mlp_ffw_size * 2}] -> "
        f"split gate/up [B,S,{config.mlp_ffw_size}] -> "
        f"down_proj [B,S,{config.embed}]"
    )

    print("params:")
    for name, (shape, dtype, num_bytes, param_type) in actual.items():
        print(
            f"  {name:<24} {param_type:<10} "
            f"{shape_text(shape):<18} {dtype:<8} {bytes_text(num_bytes)}"
        )

    if eager is not None:
        print(f"eager output: shape={eager[0]}, dtype={eager[1]}")

    if aot is not None:
        print(f"aot output: shape={aot.out_shape}, dtype={aot.out_dtype}")
        if aot.cost_analysis is not None:
            print(f"cost analysis: {aot.cost_analysis}")
        if aot.memory_analysis is not None:
            print(f"memory analysis: {aot.memory_analysis}")

        if hlo_dir is not None:
            hlo_dir.mkdir(parents=True, exist_ok=True)
            file_prefix = safe_filename(f"{config_path.parent.name}_{case.name}")
            hlo_path = hlo_dir / f"{file_prefix}.stablehlo"
            hlo_path.write_text(aot.hlo_text)
            print(f"stablehlo file: {hlo_path}")
            if aot.compiled_text is not None:
                compiled_path = hlo_dir / f"{file_prefix}.compiled.txt"
                compiled_path.write_text(aot.compiled_text)
                print(f"compiled.as_text file: {compiled_path}")

        if hlo_lines > 0:
            print(f"stablehlo first {hlo_lines} lines:")
            for line in aot.hlo_text.splitlines()[:hlo_lines]:
                print(f"  {line}")

        if aot.compiled_text is not None and compiled_lines > 0:
            print(f"compiled.as_text first {compiled_lines} lines:")
            for line in aot.compiled_text.splitlines()[:compiled_lines]:
                print(f"  {line}")

        if show_interesting_hlo:
            print("stablehlo shape ops:")
            for line in interesting_hlo_lines(aot.hlo_text, limit=interesting_limit):
                print(f"  {line}")

        if aot.compiled_text is not None and show_interesting_compiled:
            print("compiled.as_text ops:")
            for line in interesting_compiled_lines(
                aot.compiled_text, limit=interesting_limit
            ):
                print(f"  {line}")

    if errors:
        print("errors:")
        for error in errors:
            print(f"  {error}")
    else:
        print("shape check: ok")


def make_cases(args: argparse.Namespace) -> list[DebugCase]:
    cases: list[DebugCase] = []
    if not args.no_off:
        cases.append(
            DebugCase(
                name="off",
                rank=None,
                expects_lora_params=False,
                expects_lora_used=False,
            )
        )

    for rank in args.ranks:
        cases.append(
            DebugCase(
                name=f"rank_{rank}",
                rank=rank,
                expects_lora_params=True,
                expects_lora_used=True,
            )
        )
        if args.toggle_off_after_rank:
            cases.append(
                DebugCase(
                    name=f"rank_{rank}_then_off",
                    rank=rank,
                    expects_lora_params=True,
                    expects_lora_used=False,
                    toggle_off_after_init=True,
                )
            )

    return cases


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect python/valm/model/mlp.py shapes for Qwen3 MLP LoRA on/off cases."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="HF config.json path. May be passed more than once.",
    )
    parser.add_argument(
        "--base-models-root",
        type=Path,
        default=Path("base-models"),
        help="Root used to auto-discover Qwen3 config.json files.",
    )
    parser.add_argument(
        "--no-dedupe-shapes",
        action="store_true",
        help="Inspect every discovered Qwen3 config, even when shapes are identical.",
    )
    parser.add_argument(
        "--ranks",
        type=parse_ranks,
        default=parse_ranks("1,8,64"),
        help="Comma-separated positive LoRA ranks to inspect.",
    )
    parser.add_argument(
        "--no-off",
        action="store_true",
        help="Do not include the no-LoRA case.",
    )
    parser.add_argument(
        "--toggle-off-after-rank",
        action="store_true",
        help="Also inspect rank_N_then_off cases where LoRA params remain present.",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "bfloat16", "f32", "float32"),
        default="bf16",
        help="Input dtype for AOT/eager shape checks.",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu", "tpu"),
        help="Set JAX_PLATFORM_NAME before importing JAX.",
    )
    parser.add_argument(
        "--skip-aot",
        action="store_true",
        help="Only inspect constructed parameter shapes.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the lowered function and print cost/memory analysis when available.",
    )
    parser.add_argument(
        "--run-eager",
        action="store_true",
        help="Run an actual forward pass. This can be slow for 4B configs on CPU.",
    )
    parser.add_argument(
        "--hlo-dir",
        type=Path,
        help="Write full StableHLO text for each case to this directory.",
    )
    parser.add_argument(
        "--hlo-lines",
        type=int,
        default=0,
        help="Print the first N StableHLO lines for each case.",
    )
    parser.add_argument(
        "--compiled-lines",
        type=int,
        default=0,
        help="Print the first N compiled.as_text() lines for each compiled case.",
    )
    parser.add_argument(
        "--no-interesting-hlo",
        action="store_true",
        help="Do not print filtered dot/slice/add StableHLO lines.",
    )
    parser.add_argument(
        "--no-interesting-compiled",
        action="store_true",
        help="Do not print filtered compiled.as_text() lines when --compile is set.",
    )
    parser.add_argument(
        "--interesting-lines",
        type=int,
        default=24,
        help="Maximum filtered StableHLO shape-op lines to print per case.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config_paths = args.config
    if config_paths is None:
        config_paths = find_local_qwen3_configs(
            args.base_models_root, dedupe_shapes=not args.no_dedupe_shapes
        )

    if not config_paths:
        print(
            "No Qwen3 configs found. Pass --config path/to/config.json.",
            file=sys.stderr,
        )
        return 2

    cases = make_cases(args)
    dtype = input_dtype(args.dtype)
    all_errors: list[str] = []

    print(f"jax: {jax.__version__}, backend: {jax.default_backend()}")
    print(f"configs: {len(config_paths)}, cases per config: {len(cases)}")

    for config_path in config_paths:
        config, data = load_qwen3_config(config_path)
        if data.get("model_type") != "qwen3":
            print(
                f"warning: {config_path} has model_type={data.get('model_type')!r}",
                file=sys.stderr,
            )

        for case_index, case in enumerate(cases):
            mlp = build_mlp(config, case, seed=args.seed + case_index * 1000)
            actual = param_shapes(mlp)
            errors = validate_shapes(config, case, actual, mlp)
            aot = None
            eager = None

            try:
                if args.run_eager:
                    eager = run_eager(
                        mlp,
                        config,
                        batch=args.batch,
                        seq=args.seq,
                        dtype=dtype,
                    )
                if not args.skip_aot:
                    aot = run_aot(
                        mlp,
                        config,
                        batch=args.batch,
                        seq=args.seq,
                        dtype=dtype,
                        compile_lowered=args.compile,
                    )
                    expected_out = (args.batch, args.seq, config.embed)
                    if aot.out_shape != expected_out:
                        errors.append(
                            f"AOT output expected {expected_out}, observed {aot.out_shape}"
                        )
            except Exception as exc:  # noqa: BLE001 - this is a debug script.
                errors.append(f"AOT/eager failure: {type(exc).__name__}: {exc}")

            print_case(
                config_path,
                config,
                data,
                case,
                actual,
                errors,
                aot,
                eager,
                show_interesting_hlo=not args.no_interesting_hlo,
                interesting_limit=args.interesting_lines,
                hlo_lines=args.hlo_lines,
                compiled_lines=args.compiled_lines,
                show_interesting_compiled=not args.no_interesting_compiled,
                hlo_dir=args.hlo_dir,
            )
            all_errors.extend(f"{config_path}::{case.name}: {e}" for e in errors)

            del mlp
            gc.collect()
            jax.clear_caches()

    if all_errors:
        print("\nSummary: failures")
        for error in all_errors:
            print(f"  {error}")
        return 1

    print("\nSummary: all shape checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
