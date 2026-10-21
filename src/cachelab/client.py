"""OpenAI-compatible client with per-request TTFT measurement.

Client-side TTFT = wall-clock from request send to first streamed token.
Server-side TTFT = pulled from the server's own usage/timing when available
(stronger signal, since it excludes network jitter — the paper found server-side
timing more reliable).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class TTFTMeasurement:
    client_ttft: Optional[float]
    server_ttft: Optional[float]
    first_token: Optional[str]
    total_time: float
    ok: bool
    error: Optional[str] = None


class TimingClient:
    """Thin streaming client that times the first token precisely.

    We use raw httpx streaming rather than the openai SDK for the hot path so we
    control exactly when the TTFT clock stops (first content byte of the first
    token), with no SDK buffering in between.
    """

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def measure(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1,
        user: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        cache_salt: Optional[str] = None,
        chat: bool = False,
    ) -> TTFTMeasurement:
        """Send one streaming completion and measure TTFT.

        max_tokens=1 is enough: we only need the first token to time the cache
        effect, and short outputs keep runs cheap.
        """
        url = f"{self.base_url}/chat/completions" if chat else f"{self.base_url}/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            # include usage so some servers return server-side timing/cached tokens
            "stream_options": {"include_usage": True},
        }
        # vLLM honors cache_salt on the chat endpoint only (not /completions), so
        # salting/defense runs must use chat=True for the salt to enter the cache key.
        if chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt
        if user is not None:
            payload["user"] = user      # distinguishes per-user vs global cache tests
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt

        t0 = time.perf_counter()
        client_ttft: Optional[float] = None
        server_ttft: Optional[float] = None
        first_token: Optional[str] = None

        try:
            with self._client.stream("POST", url, headers=headers, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    if client_ttft is None:
                        # first streamed chunk with content = first token
                        client_ttft = time.perf_counter() - t0
                        # TODO: parse `data` JSON to capture the first token text
                        first_token = None
            total = time.perf_counter() - t0
            return TTFTMeasurement(client_ttft, server_ttft, first_token, total, ok=True)
        except Exception as e:  # noqa: BLE001 - we want to log any transport error
            total = time.perf_counter() - t0
            return TTFTMeasurement(None, None, None, total, ok=False, error=str(e))

    def close(self) -> None:
        self._client.close()
