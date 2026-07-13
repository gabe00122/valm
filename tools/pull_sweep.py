"""Pull hyper-sweep runs from wandb into results/hyper_sweep/."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import wandb

OUT = Path("/home/gabrielk/Projects/llm_rl/results/hyper_sweep")
OUT.mkdir(parents=True, exist_ok=True)

RUN_IDS = {
    "vt1": "v4h3rji4",
    "vt2": "wok2i4kw",
    "vt3": "k2axuy2f",
    "mse1": "e77m8t6d",
    "mse2": "vqprxw3q",
    "mse3": "ut400823",
    "mc1": "n4hnx6yc",
    "mc2": "ameplfbn",
    "mc3": "3gspr8f1",
    "last1": "br44nt3h",
    "last2": "py3yekqp",
    "last3": "bhtth7je",
    # real mse config (mse head, λ=0.95) — relaunched after the mislaunch
    "msefix1": "6sflifhk",
    "msefix2": "yt55yhwv",
    "msefix3": "oaotjilk",
    # grpo baseline: cold = from scratch, warm = lora warm-start
    "grpo-cold1": "vpvmotyj",
    "grpo-cold2": "zgxlcxns",
    "grpo-cold3": "qzn2xvuj",
    "grpo-warm": "uplmzr8a",
    "grpo-warm2": "q0es9y64",
    "grpo-warm3": "x6ya7etb",
}

api = wandb.Api(timeout=60)


def pull(name: str, run_id: str) -> dict:
    run = api.run(f"gabe-keith/vaml/{run_id}")
    rows = list(run.scan_history())
    df = pd.DataFrame(rows)
    df = df.sort_values("_step").reset_index(drop=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.config.json").write_text(json.dumps(run.config, indent=2, default=str))
    return {
        "name": name,
        "id": run_id,
        "state": run.state,
        "created_at": str(run.created_at),
        "last_step": int(df["_step"].max()),
        "rows": len(df),
        "word_found_last": float(df["env.word_found"].dropna().iloc[-1]) if "env.word_found" in df else None,
    }


with ThreadPoolExecutor(max_workers=6) as ex:
    manifest = list(ex.map(lambda kv: pull(*kv), RUN_IDS.items()))

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
for m in manifest:
    print(f"{m['name']:>6}  {m['state']:>8}  steps={m['last_step']:>6}  word_found(last)={m['word_found_last']}")
