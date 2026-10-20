"""Ground-truth hit/miss labeling.

Wraps server.read_cache_counters into a context manager so a single measured
request is bracketed by a before/after snapshot and gets an authoritative label.

This is the module that distinguishes this project from the original black-box
audit: the label here is TRUE cache state, so the precision we later report is a
measurement, not an inference.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from .config import Config
from .server import (
    CacheCounters,
    read_cache_counters,
    ground_truth_hit,
    try_reset_prefix_cache,
)


@dataclass
class GroundTruthLabel:
    hit: Optional[bool]
    cached_tokens_delta: Optional[int]
    queries_delta: float
    hits_delta: float


@contextmanager
def labeled_request(cfg: Config) -> Iterator[list[Optional[GroundTruthLabel]]]:
    """Context manager that snapshots cache counters before and after.

    Usage:
        with labeled_request(cfg) as box:
            measurement = client.measure(...)
        label = box[0]          # GroundTruthLabel, available after the block

    We hand back a 1-element list so the caller can read the label *after* the
    with-block populates it (the after-snapshot happens on __exit__).
    """
    box: list[Optional[GroundTruthLabel]] = [None]
    before = read_cache_counters(cfg)
    try:
        yield box
    finally:
        after = read_cache_counters(cfg)
        d = after.delta(before)
        box[0] = GroundTruthLabel(
            hit=ground_truth_hit(cfg, before, after),
            cached_tokens_delta=None,   # TODO: wire if a cached-tokens metric is exposed
            queries_delta=d.queries,
            hits_delta=d.hits,
        )


def reset_prefix_cache(cfg: Config) -> bool:
    """Best-effort cache reset between miss-condition trials.

    To generate a guaranteed MISS, the prefix must not already be cached. The
    cleanest way is a unique prefix per miss trial (see audit.py). vLLM also has
    an admin reset endpoint on some builds; we leave a hook here.

    Returns True if a reset endpoint was hit, False if we rely on unique prefixes.

    Note: on a GLOBAL-cache audit the unique-prefix strategy is what guarantees a
    miss; the server reset is only useful for isolating back-to-back HIT trials or
    for sanity-resetting between prompts. Prefer unique prefixes for the miss
    condition and use this reset between distinct prompt_ids.
    """
    return try_reset_prefix_cache(cfg)
