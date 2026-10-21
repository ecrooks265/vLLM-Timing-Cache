#!/usr/bin/env python
"""Plot the per-block SNR sweep (runs/snr_sweep.csv) -> runs/snr_sweep_fig/ and docs/."""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator

csv_path = Path("runs/snr_sweep.csv")
if not csv_path.exists():
    sys.exit("run experiments/11_block_snr.py against each model first")
rows = sorted(csv.DictReader(open(csv_path)), key=lambda r: float(r["params_B"]))
x = [float(r["params_B"]) for r in rows]
auc = [float(r["block_auc"]) for r in rows]
delta = [float(r["delta_ms"]) for r in rows]
noise = [float(r["noise_ms"]) for r in rows]
labels = [f"{v:g}B" for v in x]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("model size (parameters)"); ax.grid(alpha=0.3)
ax1.axhspan(0.45, 0.55, color="gray", alpha=0.15, label="chance band")
ax1.axhline(0.5, ls="--", c="gray", lw=1)
ax1.plot(x, auc, "o-", c="#c0392b", ms=9, lw=2)
ax1.set_ylim(0, 1); ax1.set_ylabel("per-block AUC")
ax1.set_title("Per-block detectability vs model size"); ax1.legend(loc="upper right")
ax2.axhline(0, ls="--", c="gray", lw=1)
ax2.errorbar(x, delta, yerr=noise, fmt="o-", c="#2c3e50", ms=9, lw=2, capsize=5)
ax2.set_ylabel("block delta: miss - hit (ms)")
ax2.set_title("Per-block TTFT delta vs model size (+/- noise)")
fig.suptitle("Per-block cache-timing signal is noise at every tested model size (batch=1)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("runs/snr_sweep_fig"); out.mkdir(exist_ok=True)
for p in (out, Path("docs")):
    fig.savefig(p / "snr_sweep.png", dpi=140, bbox_inches="tight")
    fig.savefig(p / "snr_sweep.svg", bbox_inches="tight")
print("wrote docs/snr_sweep.png and runs/snr_sweep_fig/")
