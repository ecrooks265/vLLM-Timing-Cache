"""Typed configuration loaded from YAML.

Keeping config in one typed object means every experiment reads the same knobs
and a run can be reproduced from the dumped config in its output dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    name: str
    dtype: str = "bfloat16"
    max_model_len: int = 4096


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    block_size: int = 1
    enable_prefix_caching: bool = True
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.90
    expose_metrics: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def metrics_url(self) -> str:
        return f"http://{self.host}:{self.port}/metrics"


class TimingCfg(BaseModel):
    num_samples: int = 250
    num_warmup: int = 10
    timing_source: Literal["client", "server", "both"] = "both"
    inter_request_delay: float = 0.05


class AttackCfg(BaseModel):
    prefix_fraction: float = 0.95
    max_tokens_to_recover: int = 32
    beam_width: int = 1
    candidate_vocab: str = "ascii"
    victim_prompt_file: str = "config/victim_prompts.txt"
    prompt_min_words: int = 0   # pad base prompt to >= this many words (longer cached prefix = cleaner hit/miss separation)


class StatsCfg(BaseModel):
    ks_alpha: float = 1.0e-8
    precision_target: float = 0.70


class ExternalCfg(BaseModel):
    enabled: bool = False
    provider: Optional[str] = None
    base_url: Optional[str] = None


class RunCfg(BaseModel):
    out_dir: str = "runs"
    seed: int = 1337


class Config(BaseModel):
    model: ModelCfg
    server: ServerCfg
    timing: TimingCfg
    attack: AttackCfg
    stats: StatsCfg
    external: ExternalCfg = Field(default_factory=ExternalCfg)
    run: RunCfg = Field(default_factory=RunCfg)
    cache_levels: list[str] = ["per_user", "global"]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
