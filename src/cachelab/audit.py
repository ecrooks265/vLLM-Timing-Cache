"""Side-channel audit at per-user and global cache-sharing levels.

For each level we collect labeled hit/miss TTFT samples and compute:
  - KS test (is there any detectable timing difference?)
  - average precision + best-threshold precision/recall of the timing detector
    against ground-truth labels.

Success criterion (from the proposal): KS p < alpha AND average precision >=
precision_target (paper reported 0.71-0.90 on vulnerable providers).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .client import TimingClient
from .config import Config
from .io_utils import JsonlWriter
from .stats import ks_hit_vs_miss, average_precision, best_threshold_classifier
from .timing import collect_hit_miss_samples, Sample


@dataclass
class LevelResult:
    cache_level: str
    timing_source: str
    ks_statistic: float
    ks_pvalue: float
    ks_significant: bool
    average_precision: float
    best_precision: float
    best_recall: float
    best_f1: float
    n_samples: int
    success: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _pick_ttft(s: Sample, source: str) -> Optional[float]:
    if source == "server":
        return s.server_ttft
    return s.client_ttft   # default to client; "both" handled by caller


def _score(samples: list[Sample], source: str, cfg: Config, level: str) -> LevelResult:
    ttfts, truth = [], []
    for s in samples:
        v = _pick_ttft(s, source)
        if v is None or s.gt_hit is None:
            continue
        ttfts.append(v)
        truth.append(s.gt_hit)

    # split by ground-truth label for the KS test
    hit = [t for t, y in zip(ttfts, truth) if y]
    miss = [t for t, y in zip(ttfts, truth) if not y]

    ks = ks_hit_vs_miss(hit, miss, alpha=cfg.stats.ks_alpha) if (hit and miss) else None
    ap = average_precision(ttfts, truth) if ttfts else 0.0
    clf = best_threshold_classifier(ttfts, truth) if ttfts else None

    success = bool(
        ks is not None and ks.significant and ap >= cfg.stats.precision_target
    )
    return LevelResult(
        cache_level=level,
        timing_source=source,
        ks_statistic=ks.statistic if ks else float("nan"),
        ks_pvalue=ks.pvalue if ks else float("nan"),
        ks_significant=ks.significant if ks else False,
        average_precision=ap,
        best_precision=clf.precision if clf else 0.0,
        best_recall=clf.recall if clf else 0.0,
        best_f1=clf.f1 if clf else 0.0,
        n_samples=len(ttfts),
        success=success,
    )


def audit_level(
    cfg: Config,
    client: TimingClient,
    base_prompt: str,
    cache_level: str,
    writer: Optional[JsonlWriter] = None,
) -> list[LevelResult]:
    """Run one cache level end-to-end and score it.

    Per-user vs global is modeled by which `user` ids prime and probe:
      - per_user: attacker and victim are the SAME user  -> hit expected only
        within a user's own cache (benchmark that the signal exists at all).
      - global:   attacker and victim are DIFFERENT users -> a hit proves the
        cache is shared across users (the real-world threat).
    """
    if cache_level == "per_user":
        victim_user = attacker_user = "user_a"
    elif cache_level == "global":
        victim_user, attacker_user = "victim_user", "attacker_user"
    else:
        raise ValueError(f"unsupported cache_level: {cache_level}")

    samples = collect_hit_miss_samples(
        cfg, client, base_prompt,
        attacker_user=attacker_user, victim_user=victim_user,
        writer=writer, cache_level=cache_level,
    )

    sources = ["client", "server"] if cfg.timing.timing_source == "both" else [cfg.timing.timing_source]
    return [_score(samples, src, cfg, cache_level) for src in sources]
