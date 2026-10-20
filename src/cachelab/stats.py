"""Statistical tests and scoring.

This is the part that turns raw TTFT samples into the numbers the proposal
promises: a KS test for "is there a detectable timing difference", and
precision/recall of the *timing-based* hit predictor against *ground-truth*
hit labels.

Implemented (not stubbed) so it can be unit-tested without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class KSResult:
    statistic: float
    pvalue: float
    n_hit: int
    n_miss: int
    significant: bool          # pvalue < alpha
    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


def ks_hit_vs_miss(
    hit_ttfts: Sequence[float],
    miss_ttfts: Sequence[float],
    alpha: float = 1e-8,
) -> KSResult:
    """Two-sample KS test between cache-hit and cache-miss TTFT distributions.

    This matches the audit paper's formulation: the null hypothesis is that the
    two TTFT distributions are identical (i.e. no observable caching signal).
    A small p-value is evidence of a timing side channel.
    """
    hit = np.asarray(hit_ttfts, dtype=float)
    miss = np.asarray(miss_ttfts, dtype=float)
    if len(hit) == 0 or len(miss) == 0:
        raise ValueError("need non-empty hit and miss samples")

    res = stats.ks_2samp(hit, miss, alternative="two-sided", method="auto")
    return KSResult(
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        n_hit=len(hit),
        n_miss=len(miss),
        significant=bool(res.pvalue < alpha),
        alpha=alpha,
    )


@dataclass
class ClassifierScore:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def best_threshold_classifier(
    ttfts: Sequence[float],
    truth_is_hit: Sequence[bool],
) -> ClassifierScore:
    """Pick the TTFT threshold that best separates hits from misses.

    A request is predicted a *hit* if its TTFT is below the threshold (hits are
    faster). We sweep candidate thresholds (the midpoints between sorted TTFTs)
    and return the one maximizing F1. Precision here is the real quantity the
    proposal reports — and because `truth_is_hit` comes from ground truth, it is
    honest, unlike a black-box audit that has to infer the label from timing.
    """
    t = np.asarray(ttfts, dtype=float)
    y = np.asarray(truth_is_hit, dtype=bool)
    if len(t) != len(y):
        raise ValueError("ttfts and truth_is_hit must be same length")
    if len(t) == 0:
        raise ValueError("empty input")

    order = np.argsort(t)
    ts = t[order]
    candidates = np.concatenate([[ts[0] - 1e-9], (ts[:-1] + ts[1:]) / 2.0, [ts[-1] + 1e-9]])

    best: ClassifierScore | None = None
    for thr in candidates:
        pred_hit = t < thr
        tp = int(np.sum(pred_hit & y))
        fp = int(np.sum(pred_hit & ~y))
        tn = int(np.sum(~pred_hit & ~y))
        fn = int(np.sum(~pred_hit & y))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = (tp + tn) / len(t)
        score = ClassifierScore(float(thr), precision, recall, f1, acc, tp, fp, tn, fn)
        if best is None or score.f1 > best.f1:
            best = score
    assert best is not None
    return best


def average_precision(
    ttfts: Sequence[float],
    truth_is_hit: Sequence[bool],
) -> float:
    """Average precision of the timing-based hit detector.

    Scores requests by -TTFT (faster = more likely a hit) and computes AP
    against ground-truth labels. This is the headline metric in the proposal;
    the paper reported AP in the 0.71–0.90 range across vulnerable providers.
    """
    t = np.asarray(ttfts, dtype=float)
    y = np.asarray(truth_is_hit, dtype=bool)
    if len(t) == 0:
        raise ValueError("empty input")
    if y.sum() == 0:
        return 0.0

    scores = -t
    order = np.argsort(scores)[::-1]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted)
    precision_at_k = tp_cum / (np.arange(len(y_sorted)) + 1)
    # AP = mean precision at the ranks where a true positive is retrieved
    ap = float(np.sum(precision_at_k * y_sorted) / y.sum())
    return ap
