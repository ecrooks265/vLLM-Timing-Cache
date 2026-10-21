"""Calibrated timing-oracle simulator for prefix-cache recovery (EarlyBird).

Runs the token-by-token recovery at block_size=1 -- which the ROCm/WSL2 stack
cannot do -- by drawing probe TTFTs from the REAL hit/miss distributions measured
on the live server (runs/<ts>/requests.jsonl). The attack logic mirrors
attack.py; only the timing oracle is simulated, and it is seeded from
ground-truth-labeled measurements, so the result is calibrated, not invented.
"""
from __future__ import annotations
import json, random, statistics, string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_VOCAB = list(string.ascii_letters + string.digits + " .,!?-_@:/")


def load_ttft_distributions(requests_jsonl):
    """(hit_ttfts, miss_ttfts) in seconds from a ground-truth-labeled run."""
    hits, misses = [], []
    for line in Path(requests_jsonl).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t, gt = r.get("client_ttft"), r.get("gt_prefix_cache_hit")
        if t is None or gt is None:
            continue
        (hits if gt else misses).append(float(t))
    return hits, misses


class TimingOracle:
    """Draws a probe TTFT by bootstrap-resampling the empirical hit/miss samples."""
    def __init__(self, hit_ttfts, miss_ttfts, seed=1337):
        if not hit_ttfts or not miss_ttfts:
            raise ValueError("need both hit and miss samples to calibrate the oracle")
        self.hit, self.miss = list(hit_ttfts), list(miss_ttfts)
        self.rng = random.Random(seed)

    def probe(self, is_hit):
        return self.rng.choice(self.hit if is_hit else self.miss)

    def summary(self):
        return {"hit_median_ms": round(statistics.median(self.hit) * 1000, 2),
                "miss_median_ms": round(statistics.median(self.miss) * 1000, 2),
                "delta_ms": round((statistics.median(self.miss) - statistics.median(self.hit)) * 1000, 2),
                "n_hit": len(self.hit), "n_miss": len(self.miss)}


class SimPrefixCache:
    """vLLM prefix caching at a block size, holding one victim secret.

    'Tokens' are characters (ascii stand-in for the model vocab, as in attack.py).
    Only whole matching blocks count as cached, which is what makes block_size>1
    seal the per-character signal.
    """
    def __init__(self, victim_secret, block_size=1):
        self.secret = victim_secret
        self.block = max(1, int(block_size))

    def cached_prefix_len(self, s):
        maxb = min(len(s), len(self.secret)) // self.block
        n = 0
        for b in range(maxb):
            seg = slice(b * self.block, (b + 1) * self.block)
            if s[seg] == self.secret[seg]:
                n += 1
            else:
                break
        return n * self.block

    def probe_is_hit(self, known, cand):
        """known+cand is a HIT iff it exposes a new fully-cached block beyond known."""
        return self.cached_prefix_len(known + cand) > self.cached_prefix_len(known)


@dataclass
class SimStep:
    position: int
    guessed: Optional[str]
    correct: str
    is_correct: bool


@dataclass
class SimResult:
    block_size: int
    known_prefix: str
    target: str
    recovered: str
    steps: list = field(default_factory=list)

    @property
    def n_correct(self):
        return sum(1 for s in self.steps if s.is_correct)

    @property
    def accuracy(self):
        return self.n_correct / len(self.target) if self.target else 0.0

    def to_dict(self):
        return {"block_size": self.block_size, "known_prefix": self.known_prefix,
                "target": self.target, "recovered": self.recovered,
                "n_target": len(self.target), "n_correct": self.n_correct,
                "accuracy": round(self.accuracy, 4)}


def simulate_recovery(oracle, victim_secret, *, prefix_fraction=0.5, max_recover=48,
                      block_size=1, vocab=None, repeats=5, calib_trials=40, seed=0):
    """Greedy EarlyBird recovery against the simulated, calibrated oracle."""
    rng = random.Random(seed)
    vocab = vocab or DEFAULT_VOCAB
    cache = SimPrefixCache(victim_secret, block_size)
    split = int(len(victim_secret) * prefix_fraction)
    known = victim_secret[:split]
    target = victim_secret[split:split + max_recover]

    hit_ref = statistics.median(oracle.probe(True) for _ in range(calib_trials))
    miss_ref = statistics.median(oracle.probe(False) for _ in range(calib_trials))
    threshold = (hit_ref + miss_ref) / 2.0 if miss_ref > hit_ref else None

    result = SimResult(block_size, known, target, "")
    current = known
    for pos in range(len(target)):
        truth = target[pos]
        best, best_t = None, None
        if threshold is not None:
            order = list(vocab)
            rng.shuffle(order)
            for cand in order:
                hit = cache.probe_is_hit(current, cand)
                med = statistics.median(oracle.probe(hit) for _ in range(repeats))
                if med < threshold and (best_t is None or med < best_t):
                    best, best_t = cand, med
        is_correct = (best == truth)
        result.steps.append(SimStep(pos, best, truth, is_correct))
        if best is None:
            break
        current += best
        result.recovered += best
    return result
