"""Corrected win-rate plot + training diagnostics, mse runs regrouped as mc."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("/home/gabrielk/Projects/llm_rl/hyper_sweep")

GROUPS = {  # true configs: "mse"-named runs actually ran mc.json; msefix* are the real mse relaunch
    "vt  (λ=0.95, value transformer)": ("#2a78d6", ["vt1", "vt2", "vt3"]),
    "mse  (λ=0.95, mse scalar head)": ("#1baf7a", ["msefix1", "msefix2", "msefix3"]),
    "mc  (λ=1, incl. runs named mse*)": ("#eda100", ["mc1", "mc2", "mc3", "mse1", "mse2", "mse3"]),
    "last  (λ=0.95, last-latent only)": ("#008300", ["last1", "last2", "last3"]),
    "grpo-warm  (no critic, lora warm-start)": ("#8a63d2", ["grpo-warm", "grpo-warm2", "grpo-warm3"]),
    "grpo-cold  (no critic, from scratch)": ("#d4526e", ["grpo-cold1", "grpo-cold2", "grpo-cold3"]),
}
RUNNING = set()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ROLL = 500

dfs = {}
for _, runs in GROUPS.values():
    for name in runs:
        dfs[name] = pd.read_csv(DATA / f"{name}.csv", usecols=lambda c: c in (
            "_step", "env.word_found", "entropy", "explained_variance", "advantage.std",
            "turns", "rewards.mean"))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlim(0, 26000)
    ax.set_xticks(range(0, 26000, 5000))
    ax.set_xticklabels([f"{t//1000}k" if t else "0" for t in range(0, 26000, 5000)])


# ---------- figure 1: corrected win-rate chart ----------
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.set_facecolor(SURFACE)

ends = []
for label, (color, runs) in GROUPS.items():
    for name in runs:
        df = dfs[name]
        wf = df.dropna(subset=["env.word_found"])
        y = wf["env.word_found"].rolling(ROLL, min_periods=50).mean()
        ax.plot(wf["_step"], y, color=color, lw=1.8, alpha=0.95, solid_capstyle="round")
        ends.append([float(wf["_step"].iloc[-1]), float(y.iloc[-1]), name, color, name in RUNNING])

for x, y, name, color, running in ends:
    mfc = SURFACE if running else color
    ax.plot([x], [y], marker="o", ms=6 if running else 5, mfc=mfc, mec=color, mew=1.6, zorder=5)

placed = []
for x, y, name, color, running in sorted(ends, key=lambda e: e[1]):
    ly = y
    for px, py in placed:
        if abs(x - px) < 2600 and abs(ly - py) < 0.042:
            ly = py + 0.042
    placed.append((x, ly))
    ax.annotate(name, (x, y), xytext=(x + 350, ly), color=color, fontsize=9.5,
                fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=3)
                if abs(ly - y) > 0.02 else None)

style(ax)
ax.set_xlim(0, 27800)
ax.set_ylim(-0.03, 1.06)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("update batches", color=INK2, fontsize=10)
ax.set_title("Wordle win rate by value-head variant",
             color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=44)
ax.text(0, 1.10, f"rolling mean over {ROLL} episodes · all runs finished · mc: 3 of 6 seeds never learned",
        transform=ax.get_xaxis_transform(), color=INK2, fontsize=9.5, va="bottom", clip_on=False)
ax.text(0, 1.045, "runs named mse* ran the mc config (msefix* = real-mse relaunch) · grpo-cold: 0 of 3 seeds learned",
        transform=ax.get_xaxis_transform(), color=INK2, fontsize=9.5, va="bottom", clip_on=False)
handles = [plt.Line2D([], [], color=c, lw=2.4, label=g) for g, (c, _) in GROUPS.items()]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.47, 0.08),
          frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.4)
fig.tight_layout()
fig.savefig(DATA / "learning_curves.png", facecolor=SURFACE, bbox_inches="tight")

# ---------- figure 2: diagnostics small multiples ----------
panels = [
    ("entropy", "policy entropy", None),
    ("explained_variance", "explained variance (clipped at −2)", (-2.1, 1.1)),
    ("advantage.std", "advantage std", None),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=150)
fig.set_facecolor(SURFACE)

for ax, (col, title, ylim) in zip(axes, panels):
    for label, (color, runs) in GROUPS.items():
        for name in runs:
            if col not in dfs[name]:  # grpo has no critic → no explained_variance
                continue
            df = dfs[name].dropna(subset=[col])
            y = df[col].rolling(300, min_periods=30).mean()
            if ylim:
                y = y.clip(lower=ylim[0] + 0.05)
            ax.plot(df["_step"], y, color=color, lw=1.4, alpha=0.85)
    style(ax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left")
    ax.set_xlabel("update batches", color=INK2, fontsize=9)

axes[1].axhline(0, color=BASELINE, lw=0.9, ls=(0, (3, 3)))
fig.suptitle("Training diagnostics", color=INK, fontsize=12.5, fontweight="bold",
             x=0.005, y=0.99, ha="left")
fig.text(0.005, 0.905,
         "rolling mean over 300 episodes · dead mc seeds: entropy → 0 by ~3k · grpo has no critic (no explained variance)",
         color=INK2, fontsize=9.5, ha="left")
handles = [plt.Line2D([], [], color=c, lw=2.4, label=g) for g, (c, _) in GROUPS.items()]
fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
           labelcolor=INK2, handlelength=1.4, ncol=2, bbox_to_anchor=(0.995, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(DATA / "diagnostics.png", facecolor=SURFACE, bbox_inches="tight")

# ---------- figure 3: group summary dot plot (seeds, range, mean) ----------
STAT_GROUPS = {  # label -> (color, runs); mse3 is an mc seed swapped in for dead mc1
    "vt\nsidecar + HL-Gauss": ("#2a78d6", ["vt1", "vt2", "vt3"]),
    "mse\nsidecar + scalar head": ("#1baf7a", ["msefix1", "msefix2", "msefix3"]),
    "mc\nλ=1, 3 survivors of 6": ("#eda100", ["mse3", "mc2", "mc3"]),
    "last\nlatent probe": ("#008300", ["last1", "last2", "last3"]),
    "grpo-warm\nno critic, warm-start": ("#8a63d2", ["grpo-warm", "grpo-warm2", "grpo-warm3"]),
}


def milestones(name):
    wf = dfs[name].dropna(subset=["env.word_found"])
    y = wf["env.word_found"].rolling(ROLL, min_periods=50).mean().reset_index(drop=True)
    s = wf["_step"].reset_index(drop=True)
    def cross(t):
        hit = y[y >= t]
        return int(s[hit.index[0]]) if len(hit) else None
    return cross(0.5), cross(0.9), float(y.iloc[-1])

stats = {g: [milestones(r) for r in runs] for g, (_, runs) in STAT_GROUPS.items()}

panels3 = [
    (0, "episodes to 50% win rate", (0, 10000), 2500),
    (1, "episodes to 90% win rate", (0, 22000), 5000),
    (2, f"final win rate (rolling {ROLL})", (0.9, 1.0), 0.025),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), dpi=150, sharey=True)
fig.set_facecolor(SURFACE)

ys = range(len(STAT_GROUPS) - 1, -1, -1)  # top row = first group
for ax, (idx, title, xlim, tick) in zip(axes, panels3):
    for y0, (g, (color, _)) in zip(ys, STAT_GROUPS.items()):
        vals = [r[idx] for r in stats[g] if r[idx] is not None]
        if len(vals) > 1:
            ax.plot([min(vals), max(vals)], [y0, y0], color=color, lw=1.6, alpha=0.45,
                    solid_capstyle="round", zorder=2)
            mean = sum(vals) / len(vals)
            ax.plot([mean], [y0], marker="|", ms=15, mew=2.2, color=color, zorder=3)
        ax.plot(vals, [y0] * len(vals), "o", ms=6.5, color=color, mec=SURFACE, mew=1.1, zorder=4)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlim(*xlim)
    ticks = [xlim[0] + i * tick for i in range(int(round((xlim[1] - xlim[0]) / tick)) + 1)]
    ax.set_xticks(ticks)
    if idx < 2:
        ax.set_xticklabels([f"{t / 1000:g}k" if t else "0" for t in ticks])
    else:
        ax.set_xticklabels([f"{t * 100:g}%" for t in ticks])
    ax.set_ylim(-0.6, len(STAT_GROUPS) - 0.4)
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", loc="left")

axes[0].set_yticks(list(ys))
axes[0].set_yticklabels(list(STAT_GROUPS), fontsize=9.5, color=INK2, ha="right")
axes[2].annotate("3 dead seeds at ≈0% (off scale)", (0.901, 2.42), color=MUTED, fontsize=8.5, va="center")
fig.suptitle("Group summary — one dot per seed, | = mean", color=INK, fontsize=12.5,
             fontweight="bold", x=0.005, y=0.99, ha="left")
fig.text(0.005, 0.885, "mc stats condition on survival (3 of 6 seeds collapsed) · grpo-cold omitted: 0 of 3 seeds ever learned",
         color=INK2, fontsize=9.5, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(DATA / "group_stats.png", facecolor=SURFACE, bbox_inches="tight")

# ---------- figure 4: group mean with seed min–max band ----------
import numpy as np

BAND_GROUPS = {  # mc restricted to survivors: a mean over a bimodal 6-seed set describes no real run
    "value transformer": ("#2a78d6", ["vt1", "vt2", "vt3"]),
    "mean squared error": ("#1baf7a", ["msefix1", "msefix2", "msefix3"]),
    "monte carlo": ("#eda100", ["mse3", "mc2", "mc3"]),
    "last latent only": ("#008300", ["last1", "last2", "last3"]),
    "grpo warm start": ("#8a63d2", ["grpo-warm", "grpo-warm2", "grpo-warm3"]),
    "grpo cold start": ("#d4526e", ["grpo-cold1", "grpo-cold2", "grpo-cold3"]),
}

grid = np.arange(0, 24950, 50)


def on_grid(name, col="env.word_found", roll=ROLL):
    wf = dfs[name].dropna(subset=[col])
    y = wf[col].rolling(roll, min_periods=50).mean()
    m = y.notna()
    return np.interp(grid, wf["_step"][m].to_numpy(), y[m].to_numpy(),
                     left=np.nan, right=np.nan)


fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.set_facecolor(SURFACE)

for label, (color, runs) in BAND_GROUPS.items():
    arr = np.vstack([on_grid(r) for r in runs])
    valid = ~np.isnan(arr).any(axis=0)
    x, a = grid[valid], arr[:, valid]
    ax.fill_between(x, a.min(0), a.max(0), color=color, alpha=0.16, lw=0)
    ax.plot(x, a.mean(0), color=color, lw=2, solid_capstyle="round")

style(ax)
ax.set_ylim(-0.03, 1.06)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("update batches", color=INK2, fontsize=10)
ax.set_title("Wordle win rate — group mean and seed range",
             color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
handles = [plt.Line2D([], [], color=c, lw=2.4, label=g) for g, (c, _) in BAND_GROUPS.items()]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.10),
          frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.4)
fig.tight_layout()
fig.savefig(DATA / "mean_bands.png", facecolor=SURFACE, bbox_inches="tight")

# ---------- figure 5: each variant vs the value transformer, one panel apiece ----------
PANELS5 = [  # (panel title, [(group key, legend label), ...]) — vt's head is hl gauss, so that panel contrasts head types
    ("mean squared error", [("value transformer", "hl gauss"), ("mean squared error", "mean squared error")]),
    ("last latent only", [("value transformer", "value transformer"), ("last latent only", "last latent only")]),
    ("monte carlo", [("value transformer", "value transformer"), ("monte carlo", "monte carlo")]),
    ("grpo", [("value transformer", "value transformer"), ("grpo warm start", "grpo warm start"),
              ("grpo cold start", "grpo cold start")]),
]

def panel_fig(col, suptitle, fname, percent=False, legend_loc="lower right"):
    locs = [legend_loc] * 4 if isinstance(legend_loc, str) else legend_loc
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=150, sharex=True, sharey=True)
    fig.set_facecolor(SURFACE)
    for ax, loc, (title, groups) in zip(axes.flat, locs, PANELS5):
        for key, label in groups:
            color, runs = BAND_GROUPS[key]
            arr = np.vstack([on_grid(r, col) for r in runs])
            valid = ~np.isnan(arr).any(axis=0)
            x, a = grid[valid], arr[:, valid]
            ax.fill_between(x, a.min(0), a.max(0), color=color, alpha=0.16, lw=0)
            ax.plot(x, a.mean(0), color=color, lw=2, solid_capstyle="round", label=label)
        style(ax)
        if percent:
            ax.set_ylim(-0.03, 1.06)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(f"{title} vs {groups[0][1]}", color=INK, fontsize=11,
                     fontweight="bold", loc="left")
        ax.legend(loc=loc, frameon=False, fontsize=9, labelcolor=INK2,
                  handlelength=1.4)
    for ax in axes[1]:
        ax.set_xlabel("update batches", color=INK2, fontsize=10)
    fig.suptitle(suptitle, color=INK, fontsize=13, fontweight="bold",
                 x=0.005, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(DATA / fname, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {fname}")


# ---------- figure 6: explained variance, vt vs last, mean + seed band ----------
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.set_facecolor(SURFACE)

for key in ("value transformer", "last latent only"):
    color, runs = BAND_GROUPS[key]
    arr = np.vstack([on_grid(r, "explained_variance", roll=300) for r in runs])
    valid = ~np.isnan(arr).any(axis=0)
    x, a = grid[valid], arr[:, valid]
    ax.fill_between(x, a.min(0), a.max(0), color=color, alpha=0.16, lw=0)
    ax.plot(x, a.mean(0), color=color, lw=2, solid_capstyle="round", label=key)

style(ax)
ax.set_ylim(0, 1.02)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("update batches", color=INK2, fontsize=10)
ax.set_title("Explained variance — value transformer vs last latent only",
             color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=24)
ax.text(0, 1.03, "rolling mean over 300 episodes · line = seed mean, band = seed min–max",
        transform=ax.get_xaxis_transform(), color=INK2, fontsize=9.5, va="bottom", clip_on=False)
ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK2, handlelength=1.4)
fig.tight_layout()
fig.savefig(DATA / "ev_mean_bands.png", facecolor=SURFACE, bbox_inches="tight")
print("saved ev_mean_bands.png")

panel_fig("env.word_found", "Wordle win rate — group mean and seed range", "vs_vt.png",
          percent=True)
panel_fig("turns", "Guesses per game — group mean and seed range", "vs_vt_turns.png",
          legend_loc=["upper right", "upper right", "upper right", "center right"])
panel_fig("rewards.mean", "Mean episode reward — group mean and seed range", "vs_vt_reward.png",
          legend_loc=["lower right", "lower right", "lower right", "center right"])
print("saved learning_curves.png + diagnostics.png + group_stats.png + mean_bands.png")
