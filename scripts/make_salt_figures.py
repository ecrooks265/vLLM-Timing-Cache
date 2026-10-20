#!/usr/bin/env python
"""Figures for the salting experiments (05 propagation, 06 granularity).

    python scripts/make_salt_figures.py            # newest salt_*.json under runs/
    python scripts/make_salt_figures.py runs/<ts>  # a specific run dir

Writes into <run>/figures/:
  defense_salt     AP + ground-truth hit-rate per salt granularity (06)
  salt_propagation cross-salt block-reuse hit-rate (05)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
C_AP, C_GT, C_LEAK, C_SAFE = "#0072B2", "#E69F00", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#dddddd"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def newest(base: Path, name: str):
    hits = sorted(base.glob(f"*/{name}"))
    return hits[-1] if hits else None


def save(fig, outdir, name):
    fig.tight_layout()
    outdir.mkdir(exist_ok=True)
    png = outdir / f"{name}.png"
    fig.savefig(png); fig.savefig(outdir / f"{name}.svg"); plt.close(fig)
    return str(png.relative_to(ROOT))


SHORT = {"no-salt (global cache)": "no salt\n(global)",
         "shared salt (per-org / coarse)": "shared salt\n(per-org)",
         "distinct salts (per-session)": "distinct salts\n(per-session)"}


def fig_defense(jpath: Path):
    d = json.loads(jpath.read_text()); conds = d["conditions"]
    labels = [SHORT.get(c["condition"], c["condition"]) for c in conds]
    ap = [c["average_precision"] for c in conds]
    gt = [c["gt_hit_rate"] for c in conds]
    won = [c["attack_succeeds"] for c in conds]
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w/2, ap, w, color=C_AP, label="detector AP")
    ax.bar(x + w/2, gt, w, color=C_GT, label="ground-truth hit-rate")
    ax.axhline(0.70, color="#D55E00", lw=1.4, ls="--")
    ax.text(len(conds)-0.5, 0.715, "AP target 0.70", color="#D55E00", fontsize=9, ha="right", va="bottom")
    for xi, w_ in zip(x, won):
        ax.text(xi, 1.12, "ATTACK SUCCEEDS" if w_ else "DEFEATED",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=C_LEAK if w_ else C_SAFE)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.2)
    ax.set_ylabel("score (0–1)"); ax.set_title(f"Cache-salting defense — {d['model'].split('/')[-1]}")
    ax.legend(frameon=False, loc="center right")
    return save(fig, jpath.parent / "figures", "defense_salt")


def fig_propagation(jpath: Path):
    d = json.loads(jpath.read_text()); m = d["modes"]
    order = [("no_salt", "no salt"), ("same_salt", "same salt"), ("diff_salt", "diff salt")]
    labels = [lbl for _, lbl in order]
    rates = [m[k]["hit_rate"] for k, _ in order]
    colors = [C_GT, C_SAFE, C_LEAK if m["diff_salt"]["hit_rate"] > 0.5 else C_SAFE]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, rates, 0.55, color=colors)
    for xi, k in zip(x, [k for k, _ in order]):
        ax.text(xi, min(rates[xi], 1.0) + 0.02,
                f"{rates[xi]:.2f}\n(Δhits {m[k]['mean_hits_delta']:.0f})",
                ha="center", va="bottom", fontsize=9, color=INK)
    verdict = "CROSS-SALT LEAK" if d.get("cross_salt_leak") else "salt isolates (secure)"
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.25)
    ax.set_ylabel("cache hit-rate (ground truth)")
    ax.set_title(f"Salt propagation probe — {verdict}")
    return save(fig, jpath.parent / "figures", "salt_propagation")


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    made = []
    if arg:
        run = Path(arg); run = run if run.is_absolute() else ROOT / run
        g, s = run / "salt_granularity.json", run / "salt_propagation.json"
    else:
        g = newest(ROOT / "runs", "salt_granularity.json")
        s = newest(ROOT / "runs", "salt_propagation.json")
    if g and g.exists():
        made.append(fig_defense(g))
    if s and s.exists():
        made.append(fig_propagation(s))
    if not made:
        print("no salt_granularity.json / salt_propagation.json found under runs/"); return 1
    print(f"[*] wrote {len(made)} salting figures:")
    for f in made:
        print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
