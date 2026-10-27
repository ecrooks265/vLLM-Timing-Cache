#!/usr/bin/env python3
"""Offline unit tests for the parts of modal_exp3_ttft.py CLIENT_SRC that do NOT need a
GPU/server. Catches harness bugs for free before spending Modal credit."""
import re, sys, types, json
import numpy as np

src = open("modal_exp3_ttft.py").read()
client = re.search(r"CLIENT_SRC = r'''(.*?)'''", src, re.S).group(1)

# stub the network + openai imports so we can execCLIENT_SRC offline
fake_requests = types.ModuleType("requests")
def _post(url, json=None, timeout=None):
    text = json.get("prompt", "")
    n = len(text.split())          # fake tokenizer: token == whitespace word
    if url.endswith("/tokenize"):
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"count": n})
    elif url.endswith("/detokenize"):
        return types.SimpleNamespace(raise_for_status=lambda: None,
                                     json=lambda: {"prompt": " ".join("w%d" % t for t in json["tokens"])})
    raise AssertionError(url)
fake_requests.post = _post
sys.modules["requests"] = fake_requests

fake_openai = types.ModuleType("openai")
class _OpenAI:
    def __init__(self, *a, **k): self.chat = types.SimpleNamespace(completions=None)
fake_openai.OpenAI = _OpenAI
sys.modules["openai"] = fake_openai

g = {}
exec(client, g)

fails = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  " + str(detail)) if detail else ""))
    if not cond: fails.append(name)

print("== _sum_metric ==")
prom = """# HELP vllm:prefix_cache_hits Prefix cache hits
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{model_name="qwen",engine="0"} 128.0
vllm:prefix_cache_hits_total{model_name="qwen",engine="1"} 32.0
vllm:prefix_cache_queries_total{model_name="qwen",engine="0"} 4096.0
vllm:gpu_prefix_cache_hits_total{model_name="qwen"} 999.0
vllm:num_requests_running{model_name="qwen"} 1.0
"""
check("hits summed across engines", g["_sum_metric"](prom, "vllm:prefix_cache_hits") == 160.0,
      g["_sum_metric"](prom, "vllm:prefix_cache_hits"))
check("queries parsed", g["_sum_metric"](prom, "vllm:prefix_cache_queries") == 4096.0)
check("bare (no _total) variant parsed",
      g["_sum_metric"]("vllm:prefix_cache_hits 7.0\n", "vllm:prefix_cache_hits") == 7.0)
check("does not swallow the deprecated gpu_ name",
      g["_sum_metric"](prom, "vllm:prefix_cache_hits") == 160.0)
check("comments ignored", g["_sum_metric"](prom, "vllm:num_requests_running") == 1.0)

print("== separation ==")
rng = np.random.default_rng(0)
same_a = rng.normal(100, 5, 200); same_b = rng.normal(100, 5, 200)
s = g["separation"](same_a, same_b, "hit", "miss")
check("identical distributions -> auc ~0.5", abs(s["auc"] - 0.5) < 0.08, s["auc"])
check("identical distributions -> ks_p large", s["ks_p"] > 0.05, s["ks_p"])
check("identical distributions -> overlap near but below 1 (noise floor)",
      s["kde_overlap"] > 0.7, s["kde_overlap"])
d = g["separation"](rng.normal(50, 5, 200), rng.normal(120, 5, 200), "hit", "miss")
check("disjoint distributions -> auc ~0", d["auc"] < 0.01, d["auc"])
check("disjoint distributions -> ks_stat ~1", d["ks_stat"] > 0.99, d["ks_stat"])
check("disjoint distributions -> ks_p tiny", d["ks_p"] < 1e-6, d["ks_p"])
check("delta_p50 is miss-minus-hit positive", d["delta_p50_ms"] > 0, d["delta_p50_ms"])

print("== gt ==")
rows = [{"ttft_ms": 100.0, "cached_tokens": 64.0, "queried_tokens": 70.0},
        {"ttft_ms": 110.0, "cached_tokens": 48.0, "queried_tokens": 70.0},
        {"ttft_ms": 120.0, "cached_tokens": 0.0, "queried_tokens": 70.0}]
t = g["gt"](rows)
check("gt n", t["n"] == 3)
check("gt cached p50", t["cached_tokens_p50"] == 48.0, t["cached_tokens_p50"])
check("gt frac hit", abs(t["frac_trials_with_cache_hit"] - 2/3) < 1e-9, t["frac_trials_with_cache_hit"])
check("gt p50 latency", t["p50_ms"] == 110.0, t["p50_ms"])
check("gt tolerates rows without counters (metrics scrape failed)",
      g["gt"]([{"ttft_ms": 5.0}])["p50_ms"] == 5.0)

print("== make_prefix (fake tokenizer: 1 token per whitespace word) ==")
for want in (16, 64, 1024):
    text, n = g["make_prefix"]("m", want, seed=want)
    check("make_prefix converges to exactly %d tokens" % want, n == want, "got %d" % n)

print("== probe() end-to-end against a fake engine ==")
class _Chunk: pass
class _Stream:
    def __init__(self): self._done = False
    def __iter__(self): return self
    def __next__(self):
        if self._done: raise StopIteration
        self._done = True; return _Chunk()
    def close(self): pass
class _Completions:
    def __init__(self): self.calls = []
    def create(self, **kw):
        self.calls.append(kw); return _Stream()
class _Chat: 
    def __init__(self): self.completions = _Completions()
class _Cli:
    def __init__(self): self.chat = _Chat()

state = {"hits": 0.0, "q": 0.0, "n": 0}
def _get(url, timeout=None):
    # each /metrics scrape advances the counter, so two scrapes around one request
    # bracket exactly that request's cached tokens
    state["n"] += 1
    body = "vllm:prefix_cache_hits_total %.1f\n" % (state["base"] + state["step"] * state["n"])
    body += "vllm:prefix_cache_queries_total 0.0\n"
    return types.SimpleNamespace(text=body)
fake_requests.get = _get
state["base"] = 100.0; state["step"] = 0.0
cli = _Cli()
state["step"] = 0.0
r = g["probe"](cli, "m", "hello", None)
check("probe returns ttft_ms", "ttft_ms" in r and r["ttft_ms"] > 0, r.get("ttft_ms"))
state["step"] = 64.0          # engine gains 64 cached tokens per scrape-pair, i.e. per request
r2 = g["probe"](cli, "m", "hello", "s1")
check("probe computes per-request cache delta",
      r2.get("cached_tokens") == 64.0, r2.get("cached_tokens"))
check("probe passes cache_salt through extra_body",
      cli.chat.completions.calls[-1].get("extra_body") == {"cache_salt": "s1"},
      cli.chat.completions.calls[-1].get("extra_body"))
check("probe omits extra_body when salt is None",
      cli.chat.completions.calls[0].get("extra_body") == {},
      cli.chat.completions.calls[0].get("extra_body"))

print("== recover_fields: live secret recovery, against a SIMULATED cache oracle ==")
# Simulate vLLM's block-chained prefix cache: a probe's cost falls with how many leading
# WORDS it shares with the victim's cached prompt (1 word == 1 token here). Crucially the
# simulated template puts a long KNOWN run after each field, exactly like the real design,
# so guessing a field correctly extends the matched prefix by that whole run.
HEAD_W, V = 8, 6
RUN_W = [40, 20, 10]                                   # amplifier length after field f
cands = [[chr(ord('a') + f) + str(i) for i in range(V)] for f in range(3)]

def build(known, upto):
    parts = [["HEAD"] * HEAD_W, [known[0]], ["RUN0"] * RUN_W[0]]
    if upto >= 2:
        parts += [[known[1]], ["RUN1"] * RUN_W[1]]
    if upto >= 3:
        parts += [[known[2]], ["RUN2"] * RUN_W[2]]
    return parts

def flatten(parts):
    out = []
    for p in parts:
        out += p
    return out

def make_oracle(secret, noise=0.0, seed=0):
    victim = flatten(build(secret, 3))
    rng = np.random.default_rng(seed)
    def measure(text_parts):
        p = flatten(text_parts)
        shared = 0
        for a, b in zip(p, victim):
            if a != b:
                break
            shared += 1
        return 200.0 - 1.0 * shared + (rng.normal(0, noise) if noise else 0.0)
    return measure

secret = [cands[0][3], cands[1][1], cands[2][5]]
rec, det = g["recover_fields"](build, cands, V, make_oracle(secret))
check("noise-free oracle recovers all 3 fields", rec == secret, rec)
# a correct guess captures the secret's own token PLUS the whole amplifier run
check("noise-free margins equal 1 + the amplifier length (41/21/11 ms)",
      [round(d["margin_ms"]) for d in det] == [r + 1 for r in RUN_W],
      [round(d["margin_ms"]) for d in det])

# with realistic noise the same oracle is still usable because the amplifier dominates
rec_n, det_n = g["recover_fields"](build, cands, V, make_oracle(secret, noise=6.0, seed=7))
check("noisy oracle (sigma = 6 ms) still recovers all fields", rec_n == secret, rec_n)

# A blind oracle (salt ON: every probe is an identical miss and the differences are pure
# noise) must recover only by chance. The noise has to be FRESH per call: a deterministic
# per-text score would make argmin pick the same (arbitrary) candidate every trial, which
# is not chance but a reproducible artifact.
_blind_rng = np.random.default_rng(0)
def measure_blind(text_parts):
    return 200.0 + float(_blind_rng.normal(0, 6))
hits = 0
for s in range(40):
    rb, _ = g["recover_fields"](build, cands, V, measure_blind)
    hits += sum(1 for f in range(3) if rb[f] == secret[f])
rate = hits / 120.0
check("a blind oracle recovers at ~chance (the salt-ON negative control)",
      rate < 3.0 / V, "field accuracy %.3f (chance %.3f)" % (rate, 1.0 / V))

# the one-shot property: probing a candidate twice must not be how the attack works
calls = {"n": 0}
def counting_measure(t):
    calls["n"] += 1
    return 200.0
g["recover_fields"](build, cands, V, counting_measure)
check("exactly one probe per candidate (V*n queries, no repeats)",
      calls["n"] == V * 3, calls["n"])

print()
if fails:
    print("FAILED: %d -> %s" % (len(fails), fails)); sys.exit(1)
print("ALL OFFLINE HARNESS TESTS PASSED")
