"""TTFT sampling harness.

Collects paired hit/miss TTFT samples under a given cache-sharing regime, each
carrying a ground-truth label. Downstream, stats.py turns these into KS tests
and precision numbers.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

from .client import TimingClient
from .config import Config
from .groundtruth import labeled_request
from .io_utils import RequestRecord, JsonlWriter


@dataclass
class Sample:
    client_ttft: Optional[float]
    server_ttft: Optional[float]
    intended_condition: str       # "hit" | "miss"
    gt_hit: Optional[bool]        # ground truth


def _prime_cache(client: TimingClient, cfg: Config, prompt: str, user: Optional[str]) -> None:
    """Send a prompt once to ensure it is cached (for a subsequent HIT trial)."""
    client.measure(cfg.model.name, prompt, max_tokens=1, user=user)
    time.sleep(cfg.timing.inter_request_delay)


def collect_hit_miss_samples(
    cfg: Config,
    client: TimingClient,
    base_prompt: str,
    *,
    attacker_user: Optional[str],
    victim_user: Optional[str],
    writer: Optional[JsonlWriter] = None,
    cache_level: str = "per_user",
    prompt_id: str = "p0",
) -> list[Sample]:
    """Collect NUMSAMPLES hit and miss trials.

    HIT trial:  victim primes `base_prompt`, then attacker queries the SAME prompt.
                If the cache is shared at this level, attacker sees a cache hit.
    MISS trial: attacker queries a UNIQUE prompt (base_prompt + random nonce) that
                was never primed, guaranteeing a miss.

    Whether a HIT actually occurs depends on the cache-sharing level — that is
    exactly what we are testing (per_user vs global).
    """
    samples: list[Sample] = []
    n = cfg.timing.num_samples
    warmup = cfg.timing.num_warmup
    delay = cfg.timing.inter_request_delay

    for i in tqdm(range(n + warmup), desc=f"{cache_level}:{prompt_id}"):
        is_warmup = i < warmup
        intended = "hit" if (i % 2 == 0) else "miss"

        if intended == "hit":
            # victim primes, attacker reads
            _prime_cache(client, cfg, base_prompt, user=victim_user)
            probe_prompt = base_prompt
            probe_user = attacker_user
        else:
            # unique prompt guarantees a miss
            nonce = uuid.uuid4().hex
            probe_prompt = f"{nonce} {base_prompt}"
            probe_user = attacker_user

        with labeled_request(cfg) as box:
            m = client.measure(cfg.model.name, probe_prompt, max_tokens=1, user=probe_user)
        label = box[0]
        gt_hit = label.hit if label else None

        if not is_warmup and m.ok:
            s = Sample(m.client_ttft, m.server_ttft, intended, gt_hit)
            samples.append(s)
            if writer:
                writer.write(RequestRecord(
                    idx=i, cache_level=cache_level, condition=intended,
                    prompt_id=prompt_id, client_ttft=m.client_ttft,
                    server_ttft=m.server_ttft, gt_prefix_cache_hit=gt_hit,
                    meta={"queries_delta": label.queries_delta if label else None,
                          "hits_delta": label.hits_delta if label else None},
                ).to_dict())

        time.sleep(delay)

    return samples
