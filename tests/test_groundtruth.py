"""Unit tests for the cache-counter delta logic — no server required."""
from cachelab.server import CacheCounters, _QUERIES_RE, _HITS_RE


# Sample /metrics text from each engine. If these stop matching, ground-truth
# labeling silently returns None and every precision number becomes meaningless,
# so pin the parsing here.
_V0_METRICS = """
vllm:prefix_cache_queries_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1234.0
vllm:prefix_cache_hits_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 567.0
"""
_V1_METRICS = """
vllm:gpu_prefix_cache_queries_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 1234.0
vllm:gpu_prefix_cache_hits_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 567.0
"""


def test_regex_matches_v0_metric_names():
    assert _QUERIES_RE.findall(_V0_METRICS) == ["1234.0"]
    assert _HITS_RE.findall(_V0_METRICS) == ["567.0"]


def test_regex_matches_v1_gpu_metric_names():
    assert _QUERIES_RE.findall(_V1_METRICS) == ["1234.0"]
    assert _HITS_RE.findall(_V1_METRICS) == ["567.0"]


def test_delta_reports_hit():
    before = CacheCounters(queries=100, hits=40)
    after = CacheCounters(queries=101, hits=41)
    d = after.delta(before)
    assert d.queries == 1
    assert d.hits == 1
    assert d.hit_occurred


def test_delta_reports_miss():
    before = CacheCounters(queries=100, hits=40)
    after = CacheCounters(queries=101, hits=40)   # query happened, no new hit
    d = after.delta(before)
    assert d.queries == 1
    assert d.hits == 0
    assert not d.hit_occurred


def test_no_query_movement():
    before = CacheCounters(queries=100, hits=40)
    after = CacheCounters(queries=100, hits=40)
    d = after.delta(before)
    assert d.queries == 0
    assert not d.hit_occurred
