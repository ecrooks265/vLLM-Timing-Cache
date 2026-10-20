"""Run directories, JSONL logging, and result schemas."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional


def new_run_dir(base: str | Path) -> Path:
    """Create runs/<timestamp>/ and return it."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = Path(base) / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


class JsonlWriter:
    """Append-only JSONL logger for per-request records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        self._f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class RequestRecord:
    """One measured request."""
    idx: int
    cache_level: str              # "per_user" | "global"
    condition: str                # "hit" | "miss" (intended)
    prompt_id: str
    client_ttft: Optional[float] = None
    server_ttft: Optional[float] = None
    # Ground truth, read from vLLM metrics (the whole point of white-box):
    gt_prefix_cache_hit: Optional[bool] = None
    gt_cached_tokens: Optional[int] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
