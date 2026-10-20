"""vLLM server control and ground-truth cache-metrics reader.

Two responsibilities:
  1. Build the exact vLLM launch command (portable local <-> GCP).
  2. Read vLLM's Prometheus /metrics endpoint to get REAL cache state, which is
     what makes this a white-box experiment.

The launch here is intentionally a command builder + health check rather than a
Python-embedded engine: running vLLM as its own OpenAI-compatible server is how
it behaves in production, so the timing we measure is realistic.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .config import Config


def build_launch_command(cfg: Config) -> list[str]:
    """Return the argv to start the vLLM OpenAI-compatible server."""
    s, m = cfg.server, cfg.model
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", m.name,
        "--dtype", m.dtype,
        "--max-model-len", str(m.max_model_len),
        "--host", s.host,
        "--port", str(s.port),
        "--block-size", str(s.block_size),
        "--gpu-memory-utilization", str(s.gpu_memory_utilization),
    ]
    if s.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if s.enforce_eager:
        cmd.append("--enforce-eager")
    return cmd


def wait_until_healthy(cfg: Config, timeout_s: float = 600.0) -> bool:
    """Poll the server's /health until it responds or we time out."""
    url = f"http://{cfg.server.host}:{cfg.server.port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


# ─── Ground-truth cache metrics ──────────────────────────────────────────────

# vLLM exposes prefix-cache counters in its Prometheus metrics. The metric
# names changed between engines:
#   V0 (<=0.6.x):  vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total
#   V1 (modern):   vllm:gpu_prefix_cache_queries_total / vllm:gpu_prefix_cache_hits_total
# The optional `gpu_` group below matches both so the same reader works whether
# you run the pinned 0.6.x server or a modern ROCm V1 build. `_total` is also
# optional for forward/backward compatibility.
_QUERIES_RE = re.compile(
    r"^vllm:(?:gpu_)?prefix_cache_queries(?:_total)?\{[^}]*\}\s+([0-9.eE+-]+)", re.M
)
_HITS_RE = re.compile(
    r"^vllm:(?:gpu_)?prefix_cache_hits(?:_total)?\{[^}]*\}\s+([0-9.eE+-]+)", re.M
)


@dataclass
class CacheCounters:
    queries: float
    hits: float

    def delta(self, other: "CacheCounters") -> "CacheCounters":
        return CacheCounters(self.queries - other.queries, self.hits - other.hits)

    @property
    def hit_occurred(self) -> bool:
        """True if at least one block hit was recorded in this (delta) window."""
        return self.hits > 0


def read_cache_counters(cfg: Config) -> CacheCounters:
    """Scrape vLLM /metrics and return prefix-cache query/hit counters.

    Snapshot before and after a request, then diff, to get a per-request
    ground-truth hit/miss label. This is the measurement a black-box attacker
    cannot make — and the reason we run white-box.
    """
    resp = requests.get(cfg.server.metrics_url, timeout=5)
    resp.raise_for_status()
    text = resp.text

    def _sum(pattern: re.Pattern) -> float:
        vals = [float(x) for x in pattern.findall(text)]
        return sum(vals) if vals else 0.0

    return CacheCounters(queries=_sum(_QUERIES_RE), hits=_sum(_HITS_RE))


def try_reset_prefix_cache(cfg: Config) -> bool:
    """Call vLLM's admin prefix-cache reset endpoint if the build exposes it.

    Modern vLLM (V1) serves `POST /reset_prefix_cache`. Returns True on a 200,
    False if the endpoint is absent (older builds) or errors — callers then fall
    back to the unique-prefix strategy for guaranteeing a MISS.
    """
    url = f"http://{cfg.server.host}:{cfg.server.port}/reset_prefix_cache"
    try:
        r = requests.post(url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def ground_truth_hit(cfg: Config, before: CacheCounters, after: CacheCounters) -> Optional[bool]:
    """Label a request hit/miss from the counter delta around it.

    Returns None if the counters didn't move at all (can't tell — e.g. metric
    names differ on this vLLM build and need adjusting).
    """
    d = after.delta(before)
    if d.queries <= 0:
        return None
    return d.hits > 0
