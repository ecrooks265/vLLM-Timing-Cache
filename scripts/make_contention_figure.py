#!/usr/bin/env python3
"""
Plot Exp 15 (batching / compute-contention channel) results.

Reads runs/batching_contention.json (idle_vs_busy OR sweep mode) and writes:
  docs/contention_idle_vs_busy.png  (ECDF of attacker probe TTFT, idle vs busy)
  docs/contention_sweep.png         (dose-response: victim prompt size vs probe delay)

Only matplotlib required. Run after experiments/15_batching_contention.py.

    python experiments/15_batching_contention.py                 # -> idle_vs_busy json
    python scripts/make_contention_figure.py
    python experiments/15_batching_contention.py --sweep         # overwrites json (sweep)
    python scripts/make_contention_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON = Path("runs/batching_contention.json")
DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)


def ecdf(xs):
    xs = sorted(xs)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def plot_idle_vs_busy(d):
    idle = d["idle_samples_ms"]
    busy = d["busy_samples_ms"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for xs, lab, c in [(idle, "idle (no victim)", "#2b8cbe"),
                       (busy, f"busy ({d['victims']} victims)", "#e34a33")]:
        x, y = ecdf(xs)
        ax.step(x, y, where="post", label=lab, color=c, lw=2)
    ax.set_xlabel("attacker probe TTFT (ms)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Exp 15 — salting-immune contention channel\n"
                 f"AUC P(busy>idle) = {d['auc_busy_gt_idle']:.3f} "
                 f"(95% CI {d['auc_ci95'][0]:.3f}–{d['auc_ci95'][1]:.3f}), "
                 f"Δmedian {d['delta_ms']:+.1f} ms")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = DOCS / "contention_idle_vs_busy.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_sweep(d):
    # keep only points where the victim actually sustained load; a skipped or
    # degenerate size (prompt > ctx -> all 400s -> no load) is not a channel point.
    valid = [r for r in d["rows"]
             if r.get("delta_ms") is not None and r.get("load_ok", True)]
    skipped = [r for r in d["rows"]
               if r.get("delta_ms") is None or not r.get("load_ok", True)]
    xs = [r["victim_words"] for r in valid]
    dy = [r["delta_ms"] for r in valid]
    au = [r["auc"] for r in valid]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(xs, dy, "o-", color="#e34a33", lw=2, label="Δ probe median (ms)")
    ax1.set_xlabel("victim prompt length (words)")
    ax1.set_ylabel("Δ attacker probe median TTFT (ms)", color="#e34a33")
    ax1.tick_params(axis="y", labelcolor="#e34a33")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, au, "s--", color="#2b8cbe", lw=1.5, label="AUC vs idle")
    ax2.axhline(0.5, color="gray", ls=":", lw=1)
    ax2.set_ylabel("AUC P(busy>idle)", color="#2b8cbe")
    ax2.set_ylim(0.4, 1.0)
    ax2.tick_params(axis="y", labelcolor="#2b8cbe")
    sub = ""
    if skipped:
        ws = ", ".join(str(r["victim_words"]) for r in skipped)
        sub = f"\n(excluded {ws} words: prompt > ctx, victim rejected -> no load)"
    ax1.set_title("Exp 15 — dose-response: victim workload size leaks via probe latency\n"
                  f"(baseline idle median {d['baseline_ms']:.1f} ms, {d['victims']} victims)" + sub)
    fig.tight_layout()
    out = DOCS / "contention_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    if not JSON.exists():
        print(f"no {JSON}; run experiments/15_batching_contention.py first", file=sys.stderr)
        sys.exit(1)
    d = json.loads(JSON.read_text())
    if d.get("mode") == "sweep":
        plot_sweep(d)
    else:
        plot_idle_vs_busy(d)


if __name__ == "__main__":
    main()
