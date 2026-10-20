"""Unit tests for the stats module — runnable without a GPU."""
import numpy as np

from cachelab.stats import (
    ks_hit_vs_miss, average_precision, best_threshold_classifier,
)


def test_ks_detects_clear_separation():
    rng = np.random.default_rng(0)
    hits = rng.normal(0.05, 0.005, 200)    # fast (cache hit)
    miss = rng.normal(0.25, 0.005, 200)    # slow (cache miss)
    res = ks_hit_vs_miss(hits, miss, alpha=1e-8)
    assert res.significant
    assert res.statistic > 0.9


def test_ks_no_signal_when_identical():
    rng = np.random.default_rng(1)
    a = rng.normal(0.1, 0.01, 300)
    b = rng.normal(0.1, 0.01, 300)
    res = ks_hit_vs_miss(a, b, alpha=1e-8)
    assert not res.significant


def test_average_precision_perfect_separation():
    # hits strictly faster than misses -> AP should be 1.0
    ttfts = [0.01, 0.02, 0.03, 0.30, 0.31, 0.32]
    truth = [True, True, True, False, False, False]
    assert abs(average_precision(ttfts, truth) - 1.0) < 1e-9


def test_best_threshold_classifier_recovers_split():
    ttfts = [0.01, 0.02, 0.03, 0.30, 0.31, 0.32]
    truth = [True, True, True, False, False, False]
    clf = best_threshold_classifier(ttfts, truth)
    assert clf.precision == 1.0
    assert clf.recall == 1.0
    assert clf.f1 == 1.0


def test_average_precision_handles_no_positives():
    assert average_precision([0.1, 0.2], [False, False]) == 0.0
