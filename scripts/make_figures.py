#!/usr/bin/env python
"""Generate replication figures from a cache-timing-lab run directory.

Reads <run>/requests.jsonl (+ optional calibration.json) and writes publication-
style figures into <run>/figures/. Self-contained: recomputes stats from the raw
per-request records so a figure never disagrees with its data.

    python scripts/make_figures.py                     # newest run in runs/
    python scripts/make_figures.py runs/20261010-xxxxxx

Figures (per cache level, using GROUND-TRUTH hit/miss labels):
  ecdf_<level>      TTFT empirical CDF, hit vs miss  (paper's CDF view)
  hist_<level>      TTFT histogram, hit vs miss
  pr_<level>        precision-recall of the timing detector (AP annotated)
  audit_ap          average precision per cache level (bar)
  calibration       AP vs cached-prefix length (if calibration.json present)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cachelab.stats import ks_hit_vs_miss, average_precision, best_threshold_classifier  # noqa: E402

# Okabe-Ito colorblind-safe pair (fixed order: hit, miss)
C_HIT, C_MISS = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#dddddd"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def newest_run(base: Path) -> Path:
    runs = sorted([p for p in base.glob("*") if (p / "requests.jsonl").exists()])
    if not runs:
        sys.exit(f"no run with requests.jsonl under {base}")
    return runs[-1]


def load_records(run: Path) -> list[dict]:
    recs = []
    for line in (run / "requests.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def by_level(recs: list[dict]) -> dict[str, dict]:
    """Group TTFTs (ms) by cache level and ground-truth label."""
    out: dict[str, dict] = {}
    for r in recs:
        lv = r["cache_level"]
        t = r.get("client_ttft")
        gt = r.get("gt_prefix_cache_hit")
        if t is None or gt is None:
            continue
        d = out.setdefault(lv, {"hit": [], "miss": [], "t": [], "y": []})
        ms = t * 1000.0
        (d["hit"] if gt else d["miss"]).append(ms)
        d["t"].append(ms); d["y"].append(bool(gt))
    return out


def ecdf(ax, data, color, label):
    x = np.sort(np.asarray(data, float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, lw=2, label=f"{label} (n={len(x)})")


def fig_ecdf(level, d, outdir):
    if not d["hit"] or not d["miss"]:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ecdf(ax, d["miss"], C_MISS, "cache miss")
    ecdf(ax, d["hit"], C_HIT, "cache hit")
    ax.set_xlabel("time-to-first-token (ms)"); ax.set_ylabel("empirical CDF")
    ax.set_title(f"TTFT distribution — {level}")
    ax.legend(frameon=False, loc="lower right")
    return save(fig, outdir, f"ecdf_{level}")


def fig_hist(level, d, outdir):
    if not d["hit"] or not d["miss"]:
        return None
    allv = np.array(d["hit"] + d["miss"]); lo, hi = np.percentile(allv, [1, 99])
    bins = np.linspace(lo, hi, 40)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(d["miss"], bins=bins, color=C_MISS, alpha=0.6, label=f"cache miss (n={len(d['miss'])})")
    ax.hist(d["hit"], bins=bins, color=C_HIT, alpha=0.6, label=f"cache hit (n={len(d['hit'])})")
    ax.axvline(np.median(d["miss"]), color=C_MISS, lw=1.5, ls="--")
    ax.axvline(np.median(d["hit"]), color=C_HIT, lw=1.5, ls="--")
    ax.set_xlabel("time-to-first-token (ms)"); ax.set_ylabel("requests")
    ax.set_title(f"TTFT histogram — {level}")
    ax.legend(frameon=False)
    return save(fig, outdir, f"hist_{level}")


def fig_pr(level, d, outdir):
    t = np.asarray(d["t"], float); y = np.asarray(d["y"], bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    # detector: smaller TTFT => more likely hit. Sweep thresholds -> precision/recall.
    order = np.argsort(t); ts = t[order]; ys = y[order]
    P = y.sum(); precs, recs = [], []
    for k in range(1, len(ts) + 1):
        tp = ys[:k].sum(); precs.append(tp / k); recs.append(tp / P)
    ap = average_precision(d["t"], d["y"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(recs, precs, color=C_HIT, lw=2)
    ax.axhline(P / len(y), color=MUTED, lw=1, ls=":", label=f"baseline={P/len(y):.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.02)
    ax.set_title(f"timing detector PR — {level}  (AP={ap:.3f})")
    ax.legend(frameon=False, loc="lower left")
    return save(fig, outdir, f"pr_{level}")


def fig_audit_ap(levels, outdir):
    names = list(levels.keys())
    aps, kss = [], []
    for lv in names:
        d = levels[lv]
        aps.append(average_precision(d["t"], d["y"]) if d["t"] else 0.0)
        try:
            kss.append(ks_hit_vs_miss(d["hit"], d["miss"]).pvalue if d["hit"] and d["miss"] else 1.0)
        except Exception:
            kss.append(1.0)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = np.arange(len(names))
    ax.bar(x, aps, color=[C_HIT if a >= 0.7 else MUTED for a in aps], width=0.55)
    ax.axhline(0.70, color=C_MISS, lw=1.6, ls="--")
    ax.text(-0.45, 0.715, "target AP = 0.70", color=C_MISS, fontsize=9, va="bottom", ha="left")
    for xi, a, pk in zip(x, aps, kss):
        ax.text(xi, min(a, 1.0) - 0.06, f"AP = {a:.3f}\nKS p = {pk:.1e}",
                ha="center", va="top", fontsize=9.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.08); ax.set_ylabel("average precision")
    ax.set_title("Side-channel audit — detector AP by cache level")
    return save(fig, outdir, "audit_ap")


def fig_calibration(run, outdir):
    cj = run / "calibration.json"
    if not cj.exists():
        return None
    curve = json.loads(cj.read_text()).get("calibration_curve", [])
    if not curve:
        return None
    L = [c["prompt_length"] for c in curve]; A = [c["best_average_precision"] for c in curve]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(L, A, "-o", color=C_HIT, lw=2, ms=7)
    ax.axhline(0.70, color=C_MISS, lw=1.5, ls="--", label="target AP=0.70")
    ax.set_xscale("log"); ax.set_xlabel("cached-prefix length (~tokens)"); ax.set_ylabel("best average precision")
    ax.set_ylim(0, 1.05); ax.set_title("Calibration — signal vs prefix length")
    ax.legend(frameon=False, loc="lower right")
    return save(fig, outdir, "calibration")


def save(fig, outdir, name) -> str:
    fig.tight_layout()
    png = outdir / f"{name}.png"; svg = outdir / f"{name}.svg"
    fig.savefig(png); fig.savefig(svg); plt.close(fig)
    return str(png.relative_to(ROOT))


def write_summary_md(run, levels, figs, outdir):
    lines = [f"# Figures — run `{run.name}`", "",
             "Ground-truth-labeled timing audit (client-side TTFT).", "",
             "| cache level | n | median miss (ms) | median hit (ms) | Δ (ms) | AP | KS p |",
             "|---|---|---|---|---|---|---|"]
    for lv, d in levels.items():
        if not d["hit"] or not d["miss"]:
            continue
        mm, mh = np.median(d["miss"]), np.median(d["hit"])
        ap = average_precision(d["t"], d["y"])
        try:
            p = ks_hit_vs_miss(d["hit"], d["miss"]).pvalue
        except Exception:
            p = float("nan")
        lines.append(f"| {lv} | {len(d['t'])} | {mm:.2f} | {mh:.2f} | {mm-mh:+.2f} | {ap:.3f} | {p:.2e} |")
    lines += ["", "## Files", ""] + [f"- `{f}`" for f in figs if f]
    (outdir / "figures.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run = Path(arg) if arg else newest_run(ROOT / "runs")
    if not run.is_absolute():
        run = ROOT / run
    outdir = run / "figures"; outdir.mkdir(exist_ok=True)
    recs = load_records(run)
    levels = by_level(recs)
    figs = []
    for lv, d in levels.items():
        figs += [fig_ecdf(lv, d, outdir), fig_hist(lv, d, outdir), fig_pr(lv, d, outdir)]
    figs.append(fig_audit_ap(levels, outdir))
    figs.append(fig_calibration(run, outdir))
    write_summary_md(run, levels, figs, outdir)
    made = [f for f in figs if f]
    print(f"[*] wrote {len(made)} figures to {outdir}")
    for f in made:
        print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
