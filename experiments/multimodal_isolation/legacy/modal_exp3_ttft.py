#!/usr/bin/env python3
"""
Experiment 3 — TTFT/TTFA prefix-cache timing channel on self-hosted vLLM (needs GPU).

Run:  modal run modal_exp3_ttft.py            # text channel  (L4  + Qwen2.5-3B-Instruct)
      modal run modal_exp3_ttft.py::mm         # image channel (A100 + Qwen2.5-VL-3B-Instruct)
      modal run modal_exp3_ttft.py::audio      # audio channel (A100 + Qwen2-Audio-7B-Instruct)
      modal run modal_exp3_ttft.py::smoke      # cheap end-to-end validation of the harness

SCOPE / ETHICS (read once): this measures the timing side channel on OUR OWN vLLM
instance where we control both "victim" and "probe", so we have ground truth. It
uses the same observable and detector family as Gu et al. (ICML'25, arXiv:2502.07776)
— TTFT with max_tokens=1 plus a two-sample KS test — but it is the EXTENSION half of the
project, not the reproduction: the paper's own experiment (a black-box audit of commercial
providers) is reproduced separately by another teammate. Our contribution is what that
paper left open: the image and audio channels, the recovery question it said it could not
answer, and the measured cost of the defence it proposed but never evaluated (cache_salt,
which is the production implementation of its "per-user caching" mitigation).
It is a measurement/defense study, not an attack on any third party.

FIXES vs the first draft of this file (all four were real bugs):
  1. cache_salt ON was *vacuous*: the "victim" warm-up and the "attacker" probe used the
     SAME salt string, so salt-ON was bit-identical to salt-OFF and could never show the
     channel closing. Now the victim uses its own salt and the attacker a different one.
  2. Ground truth was a single *cumulative* gauge scraped once per prefix length. Now every
     single request is bracketed by reads of the server's own counters
     (vllm:prefix_cache_hits / vllm:prefix_cache_queries), so each trial records exactly how
     many of ITS tokens were served from cache. HIT/MISS is verified per trial, not assumed.
     (gpu_prefix_cache_hit_rate is deprecated in 0.11.0 — see vllm/v1/metrics/loggers.py.)
  3. Prefixes were "N random words", which is not N tokens and not block-aligned (block=16).
     Now the prefix is built and re-checked through the server's own /tokenize so it is
     EXACTLY the requested token count, i.e. block-aligned. Requires /tokenize+/detokenize
     (both exist in vllm/entrypoints/openai/api_server.py).
  4. `pkill -f "vllm serve"` could silently fail (no procps) and the next server would fail
     to bind :8000. Now the Popen handle is terminated explicitly and startup failure raises
     with the server log tail.

Verified against real vLLM v0.11.0 source: cache_salt is a per-request field
(vllm/entrypoints/openai/protocol.py:322) gated on engine V1, added to block-0 extra_keys
(vllm/v1/core/kv_cache_utils.py:536) so it re-keys the whole parent chain. The block hash is
SHA-256 by default (prefix_caching_hash_algo="sha256"), so the axis that matters is NOT hash
strength but cache_salt OFF vs ON, and text vs image vs audio.
"""
import modal, json, os

app = modal.App("pcache-timing")

# Modal's own python base + pip install vLLM 0.11.0. (We first tried
# `Image.from_registry("vllm/vllm-openai:v0.11.0")`, but Modal cannot determine the
# Python version inside that image -- "unable to determine the version of Python
# installed in the Image" -- and adding `add_python` re-layers the whole 15 GB image,
# which did not finish in >25 min. This path is the documented one and pins the same
# vLLM version.)
#
# PINS THAT MATTER: vLLM 0.11.0 declares only `transformers >= 4.55.2` (no upper bound,
# requirements/common.txt:10), so a bare install pulls transformers 5.x and the server
# dies at startup with "Qwen2Tokenizer has no attribute all_special_tokens_extended".
# 4.55.2 is exactly what vLLM 0.11.0's own requirements/test.txt:1154 pins, so we pin it.
# openai is capped below 2 so we get the 1.x client API this client is written against.
#
# `vllm[audio]` (= librosa, soundfile, mistral_common[audio]; setup.py:665) is required or
# every audio request returns HTTP 500 "Please install vllm[audio] for audio support".
IMG = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("vllm[audio]==0.11.0", "transformers==4.55.2", "openai>=1.99.1,<2",
                    "numpy", "scipy", "requests", "pillow", "soundfile")
       # Instrumented copies of two stock vLLM files, applied at runtime by
       # apply_vllm_patch(). They only ADD hit/miss counters; no behaviour change.
       .add_local_file("_vllm_patch/encoder_cache_manager.py",
                       "/opt/pcache_patch/encoder_cache_manager.py")
       .add_local_file("_vllm_patch/cache.py", "/opt/pcache_patch/cache.py"))
OUT = modal.Volume.from_name("pcache-out", create_if_missing=True)

# ---------------------------------------------------------------- shared client
CLIENT_SRC = r'''
import base64, io, json, os, subprocess, time
import numpy as np
import requests
from scipy.stats import ks_2samp, mannwhitneyu, gaussian_kde
from openai import OpenAI
_trap = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

BASE_URL = "http://localhost:8000"

def _tail(path, n=40):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except Exception:
        return "<no log>"

def serve(model, extra=(), log="/tmp/vllm.log"):
    """Start `vllm serve` with prefix caching and block until healthy."""
    f = open(log, "wb")
    p = subprocess.Popen(["vllm", "serve", model, "--enable-prefix-caching",
                          "--port", "8000", "--max-model-len", "8192", *extra],
                         stdout=f, stderr=subprocess.STDOUT)
    for _ in range(900):
        if p.poll() is not None:
            raise RuntimeError("vllm serve exited early; log tail:\n" + _tail(log))
        try:
            if requests.get(BASE_URL + "/health", timeout=2).status_code == 200:
                return p
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("vllm serve never became healthy; log tail:\n" + _tail(log))

def stop(p):
    try:
        p.terminate(); p.wait(timeout=60)
    except Exception:
        try: p.kill()
        except Exception: pass
    time.sleep(5)

def vllm_meta():
    m = {}
    try:
        import vllm
        m["vllm_version"] = vllm.__version__
    except Exception as e:
        m["vllm_version"] = "unknown: %s" % e
    try:
        from vllm.config import CacheConfig
        m["block_size"] = int(CacheConfig().block_size)
    except Exception:
        m["block_size"] = 16
    return m

def _sum_metric(txt, name):
    tot = 0.0
    for ln in txt.splitlines():
        if not ln or ln[0] == "#":
            continue
        if ln.startswith(name):
            rest = ln[len(name):]
            if rest.startswith("_total") or rest.startswith("{") or rest.startswith(" "):
                try: tot += float(rest.rsplit(" ", 1)[-1])
                except Exception: pass
    return tot

def counters():
    """(cached_tokens, queried_tokens) cumulative, straight from the engine."""
    try:
        t = requests.get(BASE_URL + "/metrics", timeout=10).text
    except Exception:
        return None
    return (_sum_metric(t, "vllm:prefix_cache_hits"),
            _sum_metric(t, "vllm:prefix_cache_queries"))

def probe(cli, model, content, salt=None, max_tokens=1):
    """One timed request. TTFT = create -> first streamed chunk, measured client-side.
    Bracketed by engine-counter reads so we know how many tokens of THIS request were
    cache hits (ground truth). salt=None disables cache_salt; any string enables it."""
    extra = {} if salt is None else {"cache_salt": salt}
    c0 = counters()
    t0 = time.perf_counter(); first = None
    st = cli.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens, temperature=0, stream=True, extra_body=extra)
    try:
        for _ in st:
            first = time.perf_counter(); break
    finally:
        try: st.close()
        except Exception: pass
    c1 = counters()
    if first is None:
        raise RuntimeError("stream produced no chunk")
    d = {"ttft_ms": (first - t0) * 1000.0}
    if c0 is not None and c1 is not None:
        d["cached_tokens"] = c1[0] - c0[0]
        d["queried_tokens"] = c1[1] - c0[1]
    return d

# ---- block-aligned prompt construction, through the server's own tokenizer ----
def _tokenize(model, text):
    r = requests.post(BASE_URL + "/tokenize",
                      json={"model": model, "prompt": text, "add_special_tokens": False}, timeout=60)
    r.raise_for_status(); return int(r.json()["count"])

PCACHE_STATS_DIR = "/tmp/pcache_stats"

def apply_vllm_patch():
    """Copy the instrumented files over the INSTALLED vLLM so the server subprocess
    imports them. Returns {relative_path: contains_instrumentation}."""
    import vllm, shutil
    base = os.path.dirname(vllm.__file__)
    report = {}
    for name, rel in (("encoder_cache_manager.py", "v1/core/encoder_cache_manager.py"),
                      ("cache.py", "multimodal/cache.py")):
        src = "/opt/pcache_patch/" + name
        dst = os.path.join(base, rel)
        shutil.copyfile(src, dst)
        report[rel] = "_pc_bump" in open(dst, encoding="utf-8").read()
    return report

def clear_pcache_stats():
    try:
        for f in os.listdir(PCACHE_STATS_DIR):
            os.remove(os.path.join(PCACHE_STATS_DIR, f))
    except Exception:
        pass

def pcache_stats():
    """Sum the per-process counters written by the patch. Unlike the Prometheus
    counters these are DIRECT observations of encoder/processor cache hits."""
    d = {}
    try:
        for f in os.listdir(PCACHE_STATS_DIR):
            if f.startswith("stats.") and f.endswith(".json"):
                with open(os.path.join(PCACHE_STATS_DIR, f)) as fh:
                    for k, v in json.load(fh).items():
                        d[k] = d.get(k, 0) + v
    except Exception:
        pass
    return d

def probe_direct(cli, model, content, salt=None, max_tokens=1):
    """A request bracketed by DIRECT cache counters instead of /metrics: gives the
    encoder- and processor-cache outcome of exactly this request."""
    extra = {} if salt is None else {"cache_salt": salt}
    c0 = pcache_stats()
    t0 = time.perf_counter(); first = None
    st = cli.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens, temperature=0, stream=True, extra_body=extra)
    try:
        for _ in st:
            first = time.perf_counter(); break
    finally:
        try: st.close()
        except Exception: pass
    c1 = pcache_stats()
    if first is None:
        raise RuntimeError("stream produced no chunk")
    return {"ttft_ms": (first - t0) * 1000.0,
            "encoder_cache_hit": c1.get("encoder_cache_hit", 0) - c0.get("encoder_cache_hit", 0),
            "encoder_cache_miss": c1.get("encoder_cache_miss", 0) - c0.get("encoder_cache_miss", 0),
            "processor_cache_hit": c1.get("processor_cache_hit", 0) - c0.get("processor_cache_hit", 0),
            "processor_cache_miss": c1.get("processor_cache_miss", 0) - c0.get("processor_cache_miss", 0)}


def _tokenize_ids(model, text):
    """Token IDs, not just the count -- needed to VERIFY that two prompts share exactly
    the intended number of leading tokens (Gu et al. control this precisely)."""
    r = requests.post(BASE_URL + "/tokenize",
                      json={"model": model, "prompt": text, "add_special_tokens": False}, timeout=120)
    r.raise_for_status(); return [int(x) for x in r.json()["tokens"]]

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def letter_prompt(model, n_tokens, seed, prefix_letters=None):
    """The SAME prompt distribution Gu et al. use: n_tokens space-separated single English
    letters, uniform over upper+lowercase. That distribution is chosen (there and here)
    because a whitespace-pretokenising BPE maps each letter to one token, so token
    alignment is exact and a shared letter prefix is a shared token prefix. We still
    verify the token count and the shared prefix against the server's own tokenizer.

    prefix_letters, if given, fixes the leading letters and forces position len(prefix)
    to a different letter, so the two prompts share EXACTLY that many tokens."""
    rng = np.random.default_rng(seed)
    if prefix_letters is not None:
        k = len(prefix_letters)
        letters = list(prefix_letters)
        alt = LETTERS[(LETTERS.index(prefix_letters[-1]) + 1) % len(LETTERS)] if k else "a"
        letters.append(alt)                                  # first differing position
        letters += [LETTERS[i] for i in rng.integers(0, len(LETTERS), size=max(n_tokens - k - 1, 0))]
    else:
        letters = [LETTERS[i] for i in rng.integers(0, len(LETTERS), size=n_tokens)]
    text = " ".join(letters)
    n = _tokenize(model, text)
    for _ in range(40):
        if n == n_tokens:
            break
        if n < n_tokens:
            letters += [LETTERS[i] for i in rng.integers(0, len(LETTERS), size=n_tokens - n)]
        else:
            letters = letters[:max(1, len(letters) - (n - n_tokens))]
        text = " ".join(letters)
        n = _tokenize(model, text)
    return text, n


def _detokenize(model, ids):
    r = requests.post(BASE_URL + "/detokenize",
                      json={"model": model, "tokens": ids}, timeout=60)
    r.raise_for_status(); return r.json()["prompt"]

def make_prefix(model, n_tokens, seed):
    """Random text that tokenizes to EXACTLY n_tokens (so n_tokens=16*k is block-aligned).
    Verified with /tokenize rather than assumed from a word count."""
    rng = np.random.default_rng(seed)
    ids = [int(x) for x in rng.integers(1000, 20000, size=n_tokens)]
    text = _detokenize(model, ids)
    n = _tokenize(model, text)
    for _ in range(60):
        if n == n_tokens:
            break
        if n < n_tokens:
            ids = ids + [int(x) for x in rng.integers(1000, 20000, size=n_tokens - n)]
        else:
            ids = ids[:max(1, len(ids) - (n - n_tokens))]
        text = _detokenize(model, ids)
        n = _tokenize(model, text)
    return text, n

# ---- statistics ----
def separation(a, b, a_name="hit", b_name="miss"):
    """How separable are two latency samples? This IS the attacker's detector quality.
    a = 'target content is cached' case, b = control. ks_stat~0 / auc~0.5 / overlap~1
    means the attacker cannot tell them apart (channel closed)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = {}
    ks = ks_2samp(a, b)
    # Gu et al. use a ONE-SIDED test ("since we expect cache hits to be faster under the
    # alternative", their Sec. 3.2, scipy alternative="greater"). We report both so our
    # numbers sit directly beside the team's reproduction of their protocol.
    try:
        ksg = ks_2samp(a, b, alternative="greater")
        out["ks_p_one_sided_greater"] = float(ksg.pvalue)
    except Exception:
        out["ks_p_one_sided_greater"] = float("nan")
    out["ks_stat"] = float(ks.statistic); out["ks_p"] = float(ks.pvalue)
    try:
        u = mannwhitneyu(a, b, alternative="two-sided")
        # P(random a > random b); 0.5 = indistinguishable
        out["auc"] = float(u.statistic) / float(len(a) * len(b))
    except Exception:
        out["auc"] = float("nan")
    lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
    xs = np.linspace(lo, hi, 512)
    try:
        oh = gaussian_kde(a)(xs); om = gaussian_kde(b)(xs)
        out["kde_overlap"] = float(_trap(np.minimum(oh, om), xs) /
                                   max(_trap(np.maximum(oh, om), xs), 1e-12))
    except Exception:
        out["kde_overlap"] = float("nan")
    # Gu et al.'s headline metric: AVERAGE PRECISION from the precision-recall curve,
    # cache hits (the 'a' arm) as the positive class, score = -latency. Reported here so
    # every experiment carries the base paper's own number.
    try:
        sc = np.r_[-a, -b]; yy = np.r_[np.ones(len(a)), np.zeros(len(b))]
        o = np.argsort(-sc); yy = yy[o]
        tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
        prec = tp / np.maximum(tp + fp, 1e-12); rec = tp / len(a)
        out["average_precision"] = float(np.sum(np.diff(np.r_[0.0, rec]) * np.r_[1.0, prec][1:]))
    except Exception:
        out["average_precision"] = float("nan")
    out[a_name + "_p50_ms"] = float(np.median(a))
    out[b_name + "_p50_ms"] = float(np.median(b))
    out[a_name + "_p95_ms"] = float(np.percentile(a, 95))
    out[b_name + "_p95_ms"] = float(np.percentile(b, 95))
    out["delta_p50_ms"] = out[b_name + "_p50_ms"] - out[a_name + "_p50_ms"]
    return out

def gt(rows):
    """Ground truth for a list of probe() results: did the engine really serve these
    tokens from cache, and how many? Also keeps the RAW per-trial latencies so the
    "channel closed" claims can be bootstrapped afterwards instead of resting on a
    p-value that merely failed to reach significance (see analyze_rigor.py)."""
    ct = [r.get("cached_tokens") for r in rows if r.get("cached_tokens") is not None]
    qt = [r.get("queried_tokens") for r in rows if r.get("queried_tokens") is not None]
    t = [r["ttft_ms"] for r in rows]
    d = {"n": len(rows), "p50_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
         "raw_ttft_ms": [float(x) for x in t]}
    if ct:
        d["cached_tokens_p50"] = float(np.median(ct))
        d["cached_tokens_max"] = float(np.max(ct))
        d["frac_trials_with_cache_hit"] = float(np.mean([c > 0 for c in ct]))
    if qt:
        d["queried_tokens_p50"] = float(np.median(qt))
        rates = [c / q for c, q in zip(ct, qt) if q > 0]
        d["cache_hit_token_rate_p50"] = float(np.median(rates or [0.0]))
    return d

def null_floor(vals, reps=50, seed=0):
    """EMPIRICAL NULL. Split ONE condition's own samples into two random halves and run the
    same separation test. Whatever it reports is pure noise: two samples of the SAME
    distribution are never perfectly identical, so ks_stat is NOT 0 and KDE overlap is NOT 1
    even when nothing is going on (measured ~0.8 overlap at n=30, not ~1.0). Any leak claim
    must clear this floor, and a 'channel closed' claim must sit ON it."""
    v = np.asarray(vals, float)
    if len(v) < 8:
        return {"note": "too few samples", "n": int(len(v))}
    rng = np.random.default_rng(seed)
    h = len(v) // 2
    ks, auc, ov = [], [], []
    for _ in range(reps):
        idx = rng.permutation(len(v))
        s = separation(v[idx[:h]], v[idx[h:2 * h]], "a", "b")
        ks.append(s["ks_stat"]); auc.append(s["auc"]); ov.append(s["kde_overlap"])
    auc = np.asarray(auc)
    return {"n_per_half": int(h), "reps": reps,
            "ks_stat_mean": float(np.mean(ks)), "ks_stat_p95": float(np.percentile(ks, 95)),
            "auc_mean": float(np.mean(auc)),
            "auc_absdev_from_half_mean": float(np.mean(np.abs(auc - 0.5))),
            "auc_absdev_from_half_p95": float(np.percentile(np.abs(auc - 0.5), 95)),
            "kde_overlap_mean": float(np.mean(ov))}

def recover_fields(build, cands, V, measure, order_seed=900):
    """Block-by-block recovery of a secret from a noisy HIT/MISS timing oracle.

    build(known, upto) -> probe text for the first `upto` fields being `known`.
    cands[f]           -> candidate words for field f.
    measure(text)      -> TTFT in ms for one request.

    CRITICAL: each candidate is probed EXACTLY ONCE. After one probe the attacker has
    cached that candidate's own prefix, so a second probe returns a self-hit and carries
    no information about the victim. That one-shot property is exactly why the query cost
    in exp4 is V per block -- the model and the live attack agree by construction.

    Returns (recovered_words, per_field_detail)."""
    known = ["" for _ in cands]
    recovered, detail = [], []
    for f in range(len(cands)):
        order = list(np.random.default_rng(order_seed + f).permutation(V))
        times = {}
        for idx in order:
            k = list(known); k[f] = cands[f][idx]
            times[idx] = measure(build(k, f + 1))
        best = min(times, key=times.get)
        srt = sorted(times.items(), key=lambda kv: kv[1])
        known[f] = cands[f][best]
        recovered.append(cands[f][best])
        detail.append({"field": f, "guess": cands[f][best],
                       "best_ms": float(times[best]), "runner_up_ms": float(srt[1][1]),
                       "margin_ms": float(srt[1][1] - srt[0][1]),
                       "all_times_ms": {cands[f][i]: float(times[i]) for i in times}})
    return recovered, detail

def _rec_summary(runs, V):
    """Aggregate a recovery run: field accuracy, trials fully correct, and the margin
    between the winning candidate and the runner-up (the margin is what tells you the
    oracle was real rather than a coin flip)."""
    done = [r for r in runs if "salt_off" in r]

    def agg(key):
        fld = [d for r in done for d in r[key]["per_field"]]
        ok = [d for d in fld if d["correct"]]
        margins = [d["margin_ms"] for d in fld]
        ranks = [d["truth_rank"] for d in fld if d.get("truth_rank")]
        return {
            "fields_correct": len(ok), "fields_total": len(fld),
            "field_accuracy": float(len(ok) / max(len(fld), 1)),
            "trials_fully_correct": sum(1 for r in done if r[key]["correct"]),
            "trials_total": len(done),
            "median_margin_ms": float(np.median(margins)) if margins else None,
            "median_margin_ms_when_correct": (
                float(np.median([d["margin_ms"] for d in ok])) if ok else None),
            "median_truth_rank": float(np.median(ranks)) if ranks else None,
            "chance_field_accuracy": 1.0 / V, "chance_truth_rank": (V + 1) / 2.0,
        }

    return {"salt_off": agg("salt_off"), "salt_on": agg("salt_on"),
            "queries_per_trial": 3 * V, "naive_joint_space": V ** 3,
            "speedup_vs_joint": float(V ** 3 / (3 * V))}
'''

@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=60 * 60)
def text():
    exec(CLIENT_SRC, globals())
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    N = int(os.getenv("N", "30"))
    lengths = [int(x) for x in os.getenv("LENGTHS", "16,64,256,1024").split(",")]
    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, "warmup", None)          # discard: connection/tokenizer/graph warmup
        results = {}
        for L in lengths:
            rows = {k: [] for k in ("off_miss", "off_hit", "off_ctl", "on_victim_miss",
                                    "on_same_hit", "on_attacker", "on_attacker_ctl")}
            align = []
            for i in range(N):
                seed = 7919 * L + i
                v, nv = make_prefix(model, L, seed)              # victim's prompt (block-aligned)
                u, nu = make_prefix(model, L, seed + 500000)     # a prompt nobody has sent
                align.append((nv, nu))
                sv = "victim-%d-%d" % (L, i)                     # victim's own tenant salt
                sa = "attacker-%d-%d" % (L, i)                   # attacker's different salt

                # ---- defense OFF (no cache_salt): the classic open channel
                rows["off_miss"].append(probe(cli, model, v, None))        # 1st sighting  -> MISS
                rows["off_hit"].append(probe(cli, model, v, None))         # repeat       -> HIT
                rows["off_ctl"].append(probe(cli, model, u, None))         # unseen control -> MISS

                # ---- defense ON (per-tenant cache_salt)
                rows["on_victim_miss"].append(probe(cli, model, v, sv))    # same text, new key -> MISS
                rows["on_same_hit"].append(probe(cli, model, v, sv))       # victim repeat -> HIT
                # The attacker's two probes below share the attacker's salt. Prime that salt
                # FIRST with an unrelated prompt: otherwise whichever of the two probes runs
                # first is the first-ever use of that salt and gets 0 cached tokens, while the
                # second inherits block 0 (the chat-template block) -> a latency difference
                # that is pure ORDER, not a leak of the victim's content. (This is exactly the
                # artifact the first text run produced at L=16: ks=0.433, p=0.007, auc=0.748.)
                probe(cli, model, "attacker-prime-%d-%d" % (L, i), sa)
                rows["on_attacker"].append(probe(cli, model, v, sa))       # attacker probes victim text
                rows["on_attacker_ctl"].append(probe(cli, model, u, sa))   # attacker probes unseen text

            off_hit = [r["ttft_ms"] for r in rows["off_hit"]]
            off_miss = [r["ttft_ms"] for r in rows["off_miss"]]
            off_ctl = [r["ttft_ms"] for r in rows["off_ctl"]]
            on_att = [r["ttft_ms"] for r in rows["on_attacker"]]
            on_att_ctl = [r["ttft_ms"] for r in rows["on_attacker_ctl"]]
            on_same = [r["ttft_ms"] for r in rows["on_same_hit"]]

            results[str(L)] = {
                "prefix_tokens_requested": L,
                "prefix_tokens_achieved": sorted(set(a for pair in align for a in pair)),
                "off": {
                    "hit": gt(rows["off_hit"]),
                    "miss": gt(rows["off_miss"]),
                    "control_unseen": gt(rows["off_ctl"]),
                    "attacker_detector_hit_vs_unseen": separation(off_hit, off_ctl, "hit", "unseen"),
                    "attacker_detector_hit_vs_cold": separation(off_hit, off_miss, "hit", "cold"),
                },
                "on": {
                    "victim_same_salt_hit": gt(rows["on_same_hit"]),
                    "victim_cold_miss": gt(rows["on_victim_miss"]),
                    "attacker_probe_cached_content": gt(rows["on_attacker"]),
                    "attacker_probe_unseen": gt(rows["on_attacker_ctl"]),
                    "attacker_detector_cached_vs_unseen": separation(
                        on_att, on_att_ctl, "cached_content", "unseen"),
                },
                # what this detector looks like with NOTHING going on (same condition,
                # random halves). Channel-closed claims must land on this, not below it.
                "null_floor_same_condition": {
                    "off_hit_vs_itself": null_floor(off_hit),
                    "on_attacker_vs_itself": null_floor(on_att),
                },
                "salt_cost_ms_attacker_loses_reuse": float(np.median(on_att) - np.median(off_hit)),
                "salt_cost_ms_within_tenant": float(np.median(on_same) - np.median(off_hit)),
            }
        out = {"meta": {**vllm_meta(), "model": model, "n_trials_per_length": N,
                        "defense_off": "no cache_salt (shared keyspace, multi-tenant sharing ON)",
                        "defense_on": "distinct per-tenant cache_salt (victim != attacker)",
                        "ttft_definition": "chat.completions create -> first streamed chunk, max_tokens=1"},
               "by_prefix_tokens": results}
        json.dump(out, open("/out/exp3_text.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out, indent=2))
        return out
    finally:
        stop(p)

@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=90 * 60)
def mm():
    """Image channel: is a REPEATED image detectable (TTFA hit vs miss), and does
    cache_salt close it? Also measures whether the first-ever send of an image is a miss."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    N = int(os.getenv("N", "25"))

    def b64img(seed=None):
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        a = rng.integers(0, 256, (336, 336, 3), dtype="uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt="describe"):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        img = b64img(7)
        probe(cli, model, content(b64img(999)), None)                 # warmup, discard

        miss = [probe(cli, model, content(b64img()), None) for _ in range(N)]   # fresh image each -> MISS
        probe(cli, model, content(img), None)                                   # warm the repeated image
        hit = [probe(cli, model, content(img), None) for _ in range(N)]         # identical image -> HIT

        # ---- defense ON: per-tenant cache_salt.
        # The attacker must PRIME its salt and then probe with it, because the threat is
        # "can I tell that the VICTIM cached image X?". The first version of this function
        # sent the same image 25 times under one attacker salt and reported a 54 ms
        # "leak" -- but only probe #1 was a miss; probes #2-25 were the attacker hitting
        # ITS OWN cache entry. That is not a leak of the victim's content. (Same class of
        # bug as the L=16 text artifact.)
        on_same, on_att, on_att_ctl = [], [], []
        for i in range(N):
            victim_salt = "victim-img-%d" % i
            att_salt = "attacker-img-%d" % i
            vi = b64img(1000 + i)                                   # this trial's victim image
            probe(cli, model, content(vi), victim_salt)              # victim caches under its salt
            on_same.append(probe(cli, model, content(vi), victim_salt))     # same tenant -> HIT
            # prime the attacker's salt on an unrelated image, so the two probes below are
            # symmetric (both already have the shared template block cached under att_salt)
            probe(cli, model, content(b64img()), att_salt)
            on_att.append(probe(cli, model, content(vi), att_salt))  # victim's image -> must MISS
            on_att_ctl.append(probe(cli, model, content(b64img()), att_salt))  # unseen -> MISS

        out = {
            "meta": {**vllm_meta(), "model": model, "n_trials": N,
                     "image": "336x336x3 random uint8 PNG, sent as a base64 data: URL",
                     "hit_condition": "byte-identical PNG re-sent",
                     "miss_condition": "fresh random image every call"},
            "off": {
                "hit_repeated_image": gt(hit),
                "miss_fresh_image": gt(miss),
                "attacker_detector_hit_vs_miss": separation(
                    [r["ttft_ms"] for r in hit], [r["ttft_ms"] for r in miss], "hit", "miss"),
            },
            "on": {
                "victim_same_salt_hit": gt(on_same),
                "attacker_probe_cached_image": gt(on_att),
                "attacker_probe_unseen_image": gt(on_att_ctl),
                "attacker_detector_cached_vs_unseen": separation(
                    [r["ttft_ms"] for r in on_att], [r["ttft_ms"] for r in on_att_ctl],
                    "cached_content", "unseen"),
            },
            "null_floor_same_condition": {
                "off_hit_vs_itself": null_floor([r["ttft_ms"] for r in hit]),
                "on_attacker_vs_itself": null_floor([r["ttft_ms"] for r in on_att]),
            },
            "salt_cost_ms_attacker_loses_reuse": float(
                np.median([r["ttft_ms"] for r in on_att]) - np.median([r["ttft_ms"] for r in hit])),
        }
        json.dump(out, open("/out/exp3_mm.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out, indent=2)); return out
    finally:
        stop(p)

@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=90 * 60)
def audio():
    """Audio channel + the re-encode determinism question exp2 raised.
    (1) identical wav bytes -> does it hit?  (2) does a soundfile decode round-trip of the
    'same' clip still hit (exp2 says the key is over the DECODED array, so a bit-exact PCM
    round-trip should hit but a resample changes the array and should not)?"""
    exec(CLIENT_SRC, globals())
    import base64, io, soundfile as sf
    model = os.getenv("AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
    N = int(os.getenv("N", "20"))

    def wav_b64(arr, sr=16000):
        b = io.BytesIO(); sf.write(b, arr, sr, format="WAV", subtype="PCM_16")
        return "data:audio/wav;base64," + base64.b64encode(b.getvalue()).decode()

    def roundtrip(arr, sr=16000):
        """decode the written wav back to float32: bit-identical PCM path"""
        b = io.BytesIO(); sf.write(b, arr, sr, format="WAV", subtype="PCM_16"); b.seek(0)
        y, _ = sf.read(b, dtype="float32"); return y

    def resample(arr, sr=16000, target=22050):
        """a genuine re-encode: resample to another rate (changes the decoded array)"""
        import scipy.signal as ss
        n = int(len(arr) * target / sr)
        return ss.resample(arr, n).astype("float32")

    def content(url, txt="transcribe"):
        return [{"type": "text", "text": txt}, {"type": "audio_url", "audio_url": {"url": url}}]

    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        base = (np.random.default_rng(3).standard_normal(16000).astype("float32") * 0.1)
        a = wav_b64(base)
        probe(cli, model, content(wav_b64(np.zeros(16000, "float32"))), None)   # warmup

        miss = [probe(cli, model, content(wav_b64(
            np.random.default_rng(i).standard_normal(16000).astype("float32") * 0.1)), None)
            for i in range(N)]
        probe(cli, model, content(a), None)                                     # warm identical clip
        hit = [probe(cli, model, content(a), None) for _ in range(N)]           # identical bytes -> HIT

        # Re-encode question, answered unambiguously: what matters is whether the FIRST send
        # of the re-encoded clip matches the ORIGINAL clip's media key (i.e. hits the entry
        # `a` already created), NOT the median over repeats -- after the first send the clip
        # is cached under its own key and every later send hits itself regardless.
        rt_bytes = wav_b64(roundtrip(base))
        rs_bytes = wav_b64(resample(base))
        rt_first = probe(cli, model, content(rt_bytes), None)      # vs the ORIGINAL's key
        rt_again = probe(cli, model, content(rt_bytes), None)      # self-repeat (sanity)
        rs_first = probe(cli, model, content(rs_bytes), None)      # resampled -> different array
        rs_again = probe(cli, model, content(rs_bytes), None)      # self-repeat (sanity)

        # ---- defense ON: does cache_salt close the AUDIO channel?
        # Same design as the image test: the attacker's salt is primed once, then the
        # victim's clip and an unseen clip are each probed ONCE, so neither arm can be the
        # attacker hitting its own cache. Both arms are KV-prefix-cache misses.
        on_same, on_att, on_att_ctl = [], [], []
        for i in range(N):
            victim_salt = "victim-aud-%d" % i
            att_salt = "attacker-aud-%d" % i
            clip = wav_b64(np.random.default_rng(5000 + i)
                           .standard_normal(16000).astype("float32") * 0.1)
            probe(cli, model, content(clip), victim_salt)                  # victim warms its own
            on_same.append(probe(cli, model, content(clip), victim_salt))  # same tenant -> HIT
            probe(cli, model, content(wav_b64(np.random.default_rng(6000 + i)
                  .standard_normal(16000).astype("float32") * 0.1)), att_salt)   # prime attacker salt
            on_att.append(probe(cli, model, content(clip), att_salt))      # victim's clip -> MISS?
            on_att_ctl.append(probe(cli, model, content(wav_b64(np.random.default_rng(7000 + i)
                  .standard_normal(16000).astype("float32") * 0.1)), att_salt))  # unseen -> MISS

        out = {
            "meta": {**vllm_meta(), "model": model, "n_trials": N,
                     "clip": "1.0 s of 16 kHz float32 noise -> PCM_16 WAV data: URL"},
            "off": {
                "hit_identical_bytes": gt(hit),
                "miss_fresh_clip": gt(miss),
                "attacker_detector_hit_vs_miss": separation(
                    [r["ttft_ms"] for r in hit], [r["ttft_ms"] for r in miss], "hit", "miss"),
                "decode_roundtrip_first_send_vs_original_key": {
                    **gt([rt_first]), "cached_tokens": rt_first.get("cached_tokens")},
                "decode_roundtrip_self_repeat": {
                    **gt([rt_again]), "cached_tokens": rt_again.get("cached_tokens")},
                "resampled_first_send": {
                    **gt([rs_first]), "cached_tokens": rs_first.get("cached_tokens")},
                "resampled_self_repeat": {
                    **gt([rs_again]), "cached_tokens": rs_again.get("cached_tokens")},
            },
            "on": {
                "victim_same_salt_hit": gt(on_same),
                "attacker_probe_cached_clip": gt(on_att),
                "attacker_probe_unseen_clip": gt(on_att_ctl),
                "attacker_detector_cached_vs_unseen": separation(
                    [r["ttft_ms"] for r in on_att], [r["ttft_ms"] for r in on_att_ctl],
                    "cached_content", "unseen"),
            },
            "reencode_verdict": {
                # "hits" here means: does the FIRST send of this variant reuse the ORIGINAL
                # clip's cached entry (cached_tokens clearly above the template baseline),
                # i.e. did the different byte path decode to the same media array?
                "template_baseline_cached_tokens": int(miss[0].get("cached_tokens") or 0),
                "identical_bytes_first_send_cached_tokens": int(hit[0].get("cached_tokens") or 0),
                "decode_roundtrip_reuses_original_entry":
                    bool((rt_first.get("cached_tokens") or 0) > (miss[0].get("cached_tokens") or 0) + 8),
                "resample_reuses_original_entry":
                    bool((rs_first.get("cached_tokens") or 0) > (miss[0].get("cached_tokens") or 0) + 8),
                "roundtrip_cached_tokens": rt_first.get("cached_tokens"),
                "resample_cached_tokens": rs_first.get("cached_tokens"),
                "roundtrip_self_repeat_cached_tokens": rt_again.get("cached_tokens"),
                "resample_self_repeat_cached_tokens": rs_again.get("cached_tokens"),
            },
            "null_floor_same_condition": null_floor([r["ttft_ms"] for r in hit]),
        }
        json.dump(out, open("/out/exp3_audio.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out, indent=2)); return out
    finally:
        stop(p)

@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=90 * 60)
def mm_mechanism():
    """WHY does the image channel survive cache_salt? This is the headline finding, so pin
    the mechanism with two independent controls rather than infer it.

    An image touches three caches in vLLM v0.11.0:
      1. the KV prefix cache -- salted via block-0 extra_keys, correctly isolated;
      2. the MM *processor* cache -- keyed on the media hash, `mm_processor_cache_gb`
         (default 4 GiB, LRU), NO salt;
      3. the MM *encoder* cache (vllm/v1/core/encoder_cache_manager.py) -- keyed on
         `request.mm_features[i].identifier` = "mm_hash or uuid for caching encoder
         outputs", and `cache_salt` appears nowhere in that file.
    (2) and (3) are content-keyed, so a co-tenant that sends the SAME image gets the
    speedup regardless of salt.

    Controls:
      A. re-run the attacker comparison with `--mm-processor-cache-gb 0` (processor cache
         disabled). If the advantage SURVIVES, the culprit is the encoder cache.
      B. flood the encoder cache with distinct images and then re-probe the victim's image.
         If the advantage DISAPPEARS after the flood, that pins it on the encoder cache.

    Both arms of the comparison use a FRESH salt per probe, so both are KV-cache misses
    (0 cached tokens) and the KV cache cannot explain any difference."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    N = int(os.getenv("N", "12"))
    FLOOD = int(os.getenv("FLOOD", "400"))

    def b64img(seed=None):
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        a = rng.integers(0, 256, (336, 336, 3), dtype="uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt="describe"):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    def compare(cli, tag):
        """attacker probes the victim's image vs a never-sent image, fresh salt on each."""
        cached_arm, unseen_arm = [], []
        for i in range(N):
            victim_img = b64img(1000 + i)
            probe(cli, model, content(victim_img), "%s-v-%d" % (tag, i))     # victim caches it
            time.sleep(0.05)
            cached_arm.append(probe(cli, model, content(victim_img), "%s-c-%d" % (tag, i)))
            unseen_arm.append(probe(cli, model, content(b64img(2000 + i)), "%s-u-%d" % (tag, i)))
        return cached_arm, unseen_arm, separation(
            [r["ttft_ms"] for r in cached_arm], [r["ttft_ms"] for r in unseen_arm],
            "cached_content", "unseen")

    results = {"meta": {**vllm_meta(), "model": model, "n_per_arm": N, "flood_images": FLOOD,
                        "design": "fresh cache_salt on every probe -> both arms are KV-cache misses"}}
    p = serve(model, extra=("--max-num-batched-tokens", "2048"))
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, content(b64img(999)), None)                        # warmup

        # ---- A. processor cache ON (default)
        ca, ua, det = compare(cli, "procON")
        results["processor_cache_on"] = {
            "victim_image": gt(ca), "never_sent_image": gt(ua), "detector": det}
        results["processor_cache_on"]["cached_tokens_identical_both_arms"] = (
            ca[0].get("cached_tokens") == ua[0].get("cached_tokens"))

        # ---- B. encoder-cache eviction: flood, then re-probe the same victim image
        Z = b64img(31337)
        probe(cli, model, content(Z), "evict-v")                             # victim caches Z
        before = probe(cli, model, content(Z), "evict-before")               # encoder HIT expected
        for k in range(FLOOD):
            probe(cli, model, content(b64img(40000 + k)), "flood-%d" % k)    # evict Z
        after = probe(cli, model, content(Z), "evict-after")                 # encoder MISS?
        ctl = probe(cli, model, content(b64img(50000)), "evict-ctl")         # never-sent control
        results["encoder_cache_eviction"] = {
            "before_flood_victim_image": {"ttft_ms": before["ttft_ms"],
                                          "cached_tokens": before.get("cached_tokens")},
            "after_flood_victim_image": {"ttft_ms": after["ttft_ms"],
                                         "cached_tokens": after.get("cached_tokens")},
            "never_sent_control": {"ttft_ms": ctl["ttft_ms"],
                                   "cached_tokens": ctl.get("cached_tokens")},
            "advantage_before_ms": float(ctl["ttft_ms"] - before["ttft_ms"]),
            "advantage_after_ms": float(ctl["ttft_ms"] - after["ttft_ms"]),
            "eviction_removed_the_advantage": bool(
                abs(ctl["ttft_ms"] - after["ttft_ms"]) < abs(ctl["ttft_ms"] - before["ttft_ms"]) / 2)}
    finally:
        stop(p)

    # ---- C. processor cache explicitly DISABLED
    p = serve(model, extra=("--max-num-batched-tokens", "2048", "--mm-processor-cache-gb", "0"))
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, content(b64img(999)), None)
        ca, ua, det = compare(cli, "procOFF")
        results["processor_cache_off"] = {
            "victim_image": gt(ca), "never_sent_image": gt(ua), "detector": det}
        results["processor_cache_off"]["cached_tokens_identical_both_arms"] = (
            ca[0].get("cached_tokens") == ua[0].get("cached_tokens"))
    finally:
        stop(p)

    results["conclusion"] = {
        "advantage_survives_processor_cache_disabled":
            bool(results["processor_cache_off"]["detector"]["auc"] < 0.1),
        "advantage_removed_by_encoder_cache_eviction":
            results["encoder_cache_eviction"]["eviction_removed_the_advantage"],
        "reading": ("AUC<0.1 with --mm-processor-cache-gb 0 means the processor cache is NOT "
                    "the mechanism, leaving the content-keyed EncoderCacheManager; eviction "
                    "removing the advantage confirms it directly."),
    }
    json.dump(results, open("/out/exp3_mm_mechanism.json", "w"), indent=2); OUT.commit()
    print(json.dumps(results, indent=2)); return results

@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=60 * 60)
def text_load():
    """Repeat just the concurrent-load phase, now KEEPING the raw per-trial latencies so the
    detector can be characterised with ROC/PR curves under load the same way as when idle."""
    exec(CLIENT_SRC, globals())
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    N = int(os.getenv("NLOAD", "30"))
    levels = [int(x) for x in os.getenv("LOADLEVELS", "0,4,8,16,32").split(",")]
    out = {"meta": {**vllm_meta(), "model": model, "n_per_arm": N, "load_levels": levels,
                    "note": "timing-only requests; under load the per-request KV counter delta "
                            "is meaningless because other requests advance the same counters"}}
    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, "warmup", None)
        load_prompt = make_prefix(model, 256, 4242)[0]
        for k in levels:
            stop_flag = {"stop": False}
            ths = []
            if k:
                import threading
                def worker():
                    while not stop_flag["stop"]:
                        try: _plain(cli, model, load_prompt, None)
                        except Exception: pass
                for _ in range(k):
                    th = threading.Thread(target=worker, daemon=True); th.start(); ths.append(th)
                time.sleep(3)
            L = 1024
            hv, mv = [], []
            for i in range(N):
                v, _ = make_prefix(model, L, 7000 + k * 100 + i)
                if not k:
                    probe(cli, model, v, None)          # warm it (skip under load: counter delta unusable)
                else:
                    _plain(cli, model, v, None)
                hv.append(_plain(cli, model, v, None))
                mv.append(_plain(cli, model, make_prefix(model, L, 900000 + k * 100 + i)[0], None))
            stop_flag["stop"] = True
            for th in ths:
                th.join(timeout=5)
            time.sleep(1)
            out.setdefault("by_load", {})[str(k)] = {
                "concurrent_threads": k, "prefix_tokens": L, "n": len(hv),
                "hit_p50_ms": float(np.median(hv)), "miss_p50_ms": float(np.median(mv)),
                "detector": separation(hv, mv, "hit", "miss"),
                "null_floor_hit": null_floor(hv),
                "raw_hit_ttft_ms": [float(x) for x in hv],
                "raw_miss_ttft_ms": [float(x) for x in mv],
            }
            json.dump(out, open("/out/exp3_text_load.json", "w"), indent=2); OUT.commit()
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "detector"}
                          for k, v in out["by_load"].items()}, indent=2))
        return out
    finally:
        stop(p)


@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=60 * 60)
def smoke():
    """Cheap end-to-end validation before spending GPU time on the real runs:
    image builds, server starts, /tokenize + /detokenize exist, exact-token prefix
    construction converges, cache_salt is ACCEPTED by the API, the engine counters
    move, and a repeat request really hits the cache. Prints a PASS/FAIL checklist."""
    exec(CLIENT_SRC, globals())
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    checks = {}
    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        checks["meta"] = vllm_meta()
        probe(cli, model, "warmup", None)

        # 1. exact-token prefix construction (block alignment)
        v, nv = make_prefix(model, 64, 11)
        u, nu = make_prefix(model, 64, 999)
        checks["prefix_exact_tokens"] = {"requested": 64, "achieved": nv, "ok": nv == 64}
        checks["prefix_control_exact"] = {"requested": 64, "achieved": nu, "ok": nu == 64}
        checks["prefixes_differ"] = bool(v != u)

        # 2. counters move at all (use a throwaway prompt so it cannot pollute step 3)
        c0 = counters(); probe(cli, model, "counter probe", None); c1 = counters()
        checks["counters_present"] = bool(c0 is not None and c1 is not None and c1 != (0.0, 0.0))

        # 3. a FIRST-EVER sighting of a prefix must be a real MISS and a repeat a real HIT.
        #    (v3 is fresh here: the first draft of this checklist sent `v` once for the
        #    counter check and then again as the "miss", so both samples were hits.)
        v3, n3 = make_prefix(model, 64, 4242)
        first = probe(cli, model, v3, None)
        second = probe(cli, model, v3, None)
        checks["miss_looks_like_miss"] = {"cached_tokens": first.get("cached_tokens"),
                                         "queried_tokens": first.get("queried_tokens")}
        checks["hit_looks_like_hit"] = {"cached_tokens": second.get("cached_tokens"),
                                        "queried_tokens": second.get("queried_tokens")}
        checks["hit_faster_than_miss"] = bool(second["ttft_ms"] < first["ttft_ms"])
        checks["hit_miss_ttft_ms"] = {"miss": first["ttft_ms"], "hit": second["ttft_ms"]}

        # 4. cache_salt accepted, and it really re-keys the chain: a FRESH prefix sent
        #    under salt-A, then under salt-B (must MISS despite identical tokens), then
        #    under salt-A again (must HIT).
        v4, _ = make_prefix(model, 64, 777)
        s1 = probe(cli, model, v4, "salt-A")
        s2 = probe(cli, model, v4, "salt-B")
        s3 = probe(cli, model, v4, "salt-A")
        checks["cache_salt_accepted"] = True
        checks["salt_isolates"] = {"first_saltA_cached_tokens": s1.get("cached_tokens"),
                                   "other_saltB_cached_tokens": s2.get("cached_tokens"),
                                   "same_saltA_cached_tokens": s3.get("cached_tokens")}

        # NOTE on expected numbers: every chat request carries the model's chat template,
        # whose first 16-token block is IDENTICAL for all users, so after the server's very
        # first request that block is always cached. A "miss" on a fresh prefix therefore
        # caches 16 tokens (the template block), not 0 -- measured miss=16 / hit=80 on a
        # 93-token prompt. Under a fresh cache_salt the template block is re-keyed too, so
        # that first salted request really is 0. The assertion is on CONTENT blocks, not on
        # a 0/16 absolute.
        first_ct = int(first.get("cached_tokens") or 0)
        second_ct = int(second.get("cached_tokens") or 0)
        checks["chat_template_baseline_tokens"] = first_ct
        checks["content_blocks_reused_on_repeat"] = second_ct - first_ct
        checks["PASS"] = bool(
            checks["prefix_exact_tokens"]["ok"]
            and checks["prefix_control_exact"]["ok"]
            and checks["counters_present"]
            and second_ct > first_ct                       # a repeat really reuses more
            and (second_ct - first_ct) % 16 == 0           # ... in whole 16-token blocks
            and (s3.get("cached_tokens") or 0) > (s2.get("cached_tokens") or 0)  # salt re-keys
            and (s2.get("cached_tokens") or 0) == 0)
        json.dump(checks, open("/out/exp3_smoke.json", "w"), indent=2); OUT.commit()
        print("SMOKE " + json.dumps(checks, indent=2))
        return checks
    finally:
        stop(p)

# ---------------------------------------------------------------------------------
# SECOND PASS. The first pass left three things unsatisfying:
#   * "cache_salt closes the text channel" rested on a p-value that failed to reach
#     significance -> absence of evidence. Fixed by persisting RAW per-trial latencies
#     (see gt()) and bootstrapping them in analyze_rigor.py.
#   * exp4's query-complexity model was a simulation over an IDEAL oracle and had never
#     been connected to the real, noisy server. text2() closes that loop by actually
#     recovering a templated secret from a live vLLM using nothing but TTFT.
#   * the multimodal experiments only ever re-sent the identical item with the identical
#     surrounding prompt, so they could not say how much the attacker needs to know.
#     mm_extra() shows the encoder cache is keyed on media content ALONE.
# ---------------------------------------------------------------------------------

def _plain(cli, model, content, salt=None, max_tokens=1):
    """Timing-only request (no /metrics scrape) -- safe to call from load threads,
    where per-request counter deltas would be meaningless anyway."""
    extra = {} if salt is None else {"cache_salt": salt}
    t0 = time.perf_counter()
    st = cli.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens, temperature=0, stream=True, extra_body=extra)
    try:
        for _ in st:
            return (time.perf_counter() - t0) * 1000.0
    finally:
        try: st.close()
        except Exception: pass
    raise RuntimeError("no chunk")


def pick_candidates(model, head, n, seed0):
    """Find n words w such that `head + ' ' + w` tokenises to exactly tokenize(head)+1
    tokens. Verifying through the server's tokenizer is what makes the candidates
    structurally identical, so the ONLY difference between two probes is the word."""
    base_n = _tokenize(model, head)
    out, seen, s = [], set(), seed0
    while len(out) < n and s < seed0 + 3000:
        try:
            w, _ = make_prefix(model, 1, s)
        except Exception:
            s += 1; continue
        s += 1
        w = w.strip()
        if not w or len(w) > 10 or not w.isalpha() or w in seen:
            continue
        if _tokenize(model, head + " " + w) == base_n + 1:
            seen.add(w); out.append(w)
    return out


@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=90 * 60)
def text2():
    """Text channel, second pass:
       A. salt OFF vs ON at 16..4096 tokens, keeping RAW per-trial latencies;
       B. END-TO-END recovery of a templated 3-field secret from the live server using
          only TTFT (the missing link to exp4), plus a salt-ON negative control;
       C. the same detector under CONCURRENT LOAD (real servers are not idle).
    Results are committed to the volume after each phase so a later crash cannot lose
    earlier work."""
    exec(CLIENT_SRC, globals())
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    N = int(os.getenv("N2", "60"))
    V = int(os.getenv("V", "12"))          # candidates per secret field
    lengths = [int(x) for x in os.getenv("LENGTHS2", "16,64,256,1024,2048,4096").split(",")]
    results = {"meta": {**vllm_meta(), "model": model, "n_trials_per_length": N,
                        "n_candidates_per_field": V, "lengths": lengths,
                        "note": "raw per-trial latencies are stored; bootstrap them with analyze_rigor.py"}}
    json.dump(results, open("/out/exp3_text2.json", "w"), indent=2); OUT.commit()

    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, "warmup", None)

        # ---------------- A. salt OFF vs ON with raw samples, extended lengths
        by_len = {}
        for L in lengths:
            rows = {k: [] for k in ("off_miss", "off_hit", "off_ctl", "on_same_hit",
                                    "on_attacker", "on_attacker_ctl")}
            for i in range(N):
                seed = 7919 * L + i
                v, _ = make_prefix(model, L, seed)
                u, _ = make_prefix(model, L, seed + 500000)
                sv, sa = "victim-%d-%d" % (L, i), "attacker-%d-%d" % (L, i)
                rows["off_miss"].append(probe(cli, model, v, None))
                rows["off_hit"].append(probe(cli, model, v, None))
                rows["off_ctl"].append(probe(cli, model, u, None))
                rows["on_same_hit"].append(probe(cli, model, v, sv))
                probe(cli, model, "attacker-prime-%d-%d" % (L, i), sa)   # prime (see FINDINGS)
                rows["on_attacker"].append(probe(cli, model, v, sa))
                rows["on_attacker_ctl"].append(probe(cli, model, u, sa))
            off_hit = [r["ttft_ms"] for r in rows["off_hit"]]
            off_ctl = [r["ttft_ms"] for r in rows["off_ctl"]]
            on_att = [r["ttft_ms"] for r in rows["on_attacker"]]
            on_ctl = [r["ttft_ms"] for r in rows["on_attacker_ctl"]]
            by_len[str(L)] = {
                "off_hit": gt(rows["off_hit"]), "off_miss": gt(rows["off_miss"]),
                "off_control_unseen": gt(rows["off_ctl"]),
                "on_same_hit": gt(rows["on_same_hit"]),
                "on_attacker": gt(rows["on_attacker"]),
                "on_attacker_control": gt(rows["on_attacker_ctl"]),
                "off_detector": separation(off_hit, off_ctl, "hit", "unseen"),
                "on_detector": separation(on_att, on_ctl, "cached_content", "unseen"),
                "null_floor_on_attacker": null_floor(on_att),
                "salt_cost_ms": float(np.median(on_att) - np.median(off_hit)),
                "salt_cost_within_tenant_ms": float(
                    np.median([r["ttft_ms"] for r in rows["on_same_hit"]]) - np.median(off_hit)),
            }
            results["by_prefix_tokens"] = by_len
            json.dump(results, open("/out/exp3_text2.json", "w"), indent=2); OUT.commit()

        # ---------------- B. end-to-end live secret recovery
        # Victim prompt:  HEAD [A] LONG1 [B] LONG2 [C] LONG3
        # The attacker knows HEAD/LONG1/LONG2/LONG3 and that each field comes from a
        # small candidate set; it knows nothing else. Each field is followed by a long
        # KNOWN run, so a correct guess extends the matched cached prefix by ~that whole
        # run -> a large TTFT difference; a wrong guess diverges at the field.
        HEAD = make_prefix(model, 32, 61)[0]
        run = {n: make_prefix(model, n, 100 + n)[0] for n in (1024, 512, 256)}
        # Candidates for field f are validated against the KNOWN text that immediately
        # precedes that field (HEAD / LONG1 / LONG2), which does not depend on the secret:
        # each accepted word must add exactly one token in that context.
        pre_ctx = {0: HEAD, 1: run[1024], 2: run[512]}
        cands = {f: pick_candidates(model, pre_ctx[f], V, 4000 * (f + 1)) for f in range(3)}
        want = [len(c) for c in cands.values()]
        results["recovery_setup"] = {
            "candidates_per_field": want, "v_requested": V,
            "candidate_words": {str(f): cands[f] for f in range(3)},
            "field_amplifier_tokens": {"field0_after": 1024, "field1_after": 512, "field2_after": 256},
        }
        if min(want) < V:
            results["recovery"] = {"error": "only found %s candidates (need %d)" % (want, V)}
            json.dump(results, open("/out/exp3_text2.json", "w"), indent=2); OUT.commit()
        else:
            rec = []
            for trial in range(3):
                secret = [cands[f][int(np.random.default_rng(trial * 31 + f).integers(0, V))]
                          for f in range(3)]

                def build(known, upto):
                    parts = [HEAD, known[0], run[1024]]
                    if upto >= 2:
                        parts += [known[1], run[512]]
                    if upto >= 3:
                        parts += [known[2], run[256]]
                    return " ".join(parts)

                # --- salt OFF: the attack should work
                victim_full = build(secret, 3)
                probe(cli, model, victim_full, None)                 # victim sends it
                recovered, detail = recover_fields(
                    build, cands, V, lambda txt: _plain(cli, model, txt, None))
                for f, d in enumerate(detail):
                    d["truth"] = secret[f]; d["correct"] = d["guess"] == secret[f]
                queries = 3 * V
                rec.append({"trial": trial, "defense": "OFF", "secret": secret,
                            "recovered": recovered, "correct": recovered == secret,
                            "oracle_queries": queries, "per_field": detail})

                # --- salt ON: the same attack must fail (negative control).
                # Prime the attacker's salt once so every candidate probe starts from the
                # same state (otherwise the first candidate probed pays the template-block
                # miss and the later ones inherit it -> a pure ordering signal). Then
                # exactly ONE probe per candidate, exactly as in the salt-OFF arm.
                sv, sa = "rec-victim-%d" % trial, "rec-attacker-%d" % trial
                probe(cli, model, victim_full, sv)                  # victim, own salt
                probe(cli, model, "attacker-recovery-prime-%d" % trial, sa)
                recovered_on, detail_on = recover_fields(
                    build, cands, V, lambda txt: _plain(cli, model, txt, sa))
                for f, d in enumerate(detail_on):
                    d["truth"] = secret[f]; d["correct"] = d["guess"] == secret[f]
                rec.append({"trial": trial, "defense": "ON", "secret": secret,
                            "recovered": recovered_on, "correct": recovered_on == secret,
                            "oracle_queries": 3 * V, "per_field": detail_on})
            off_fields = [d for r in rec if r["defense"] == "OFF" for d in r["per_field"]]
            on_fields = [d for r in rec if r["defense"] == "ON" for d in r["per_field"]]
            results["recovery"] = {
                "design": ("victim prompt = HEAD [A] LONG1 [B] LONG2 [C] LONG3; the attacker "
                           "knows the fixed parts and only that each field is one of V candidates; "
                           "each candidate is probed EXACTLY ONCE (a second probe would hit the "
                           "attacker's own cache entry) and argmin(TTFT) wins the field"),
                "v_candidates_per_field": V, "fields": 3,
                "queries_per_trial": 3 * V, "naive_joint_space": V ** 3,
                "runs": rec,
                "salt_off": {"fields_recovered": sum(1 for d in off_fields if d["correct"]),
                             "fields_total": len(off_fields),
                             "trials_fully_correct": sum(1 for r in rec if r["defense"] == "OFF"
                                                         and r["correct"]),
                             "median_margin_ms": float(np.median([d["margin_ms"] for d in off_fields]))},
                "salt_on": {"fields_recovered": sum(1 for d in on_fields if d["correct"]),
                            "fields_total": len(on_fields),
                            "trials_fully_correct": sum(1 for r in rec if r["defense"] == "ON"
                                                        and r["correct"]),
                            "median_margin_ms": float(np.median([d["margin_ms"] for d in on_fields])),
                            "chance_level_fields": len(on_fields) / float(V)},
            }
            json.dump(results, open("/out/exp3_text2.json", "w"), indent=2); OUT.commit()

        # ---------------- C. detector robustness under concurrent load
        load_prompt = make_prefix(model, 256, 4242)[0]
        load_res = {}
        for k in (0, 8, 32):
            stop_flag = {"stop": False}
            ths = []
            if k:
                import threading
                def worker():
                    while not stop_flag["stop"]:
                        try: _plain(cli, model, load_prompt, None)
                        except Exception: pass
                for _ in range(k):
                    th = threading.Thread(target=worker, daemon=True); th.start(); ths.append(th)
                time.sleep(3)
            L = 1024
            hv, mv = [], []
            for i in range(20):
                v, _ = make_prefix(model, L, 7000 + k * 100 + i)
                probe(cli, model, v, None)
                hv.append(_plain(cli, model, v, None))
                mv.append(_plain(cli, model, make_prefix(model, L, 900000 + k * 100 + i)[0], None))
            stop_flag["stop"] = True
            for th in ths: th.join(timeout=5)
            time.sleep(1)
            load_res["load_%d" % k] = {
                "concurrent_threads": k, "prefix_tokens": L, "n": len(hv),
                "hit_p50_ms": float(np.median(hv)), "miss_p50_ms": float(np.median(mv)),
                "detector": separation(hv, mv, "hit", "miss"),
                "null_floor_hit": null_floor(hv),
                "note": "timing-only requests under load; per-request KV counter deltas are "
                        "not usable while other requests are in flight"}
        results["concurrent_load"] = load_res
        json.dump(results, open("/out/exp3_text2.json", "w"), indent=2); OUT.commit()
        print(json.dumps(results, indent=2))
        return results
    finally:
        stop(p)


@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=90 * 60)
def mm_extra():
    """Second pass on the image channel -- what does the attacker actually need to know?

      A. SAME IMAGE, DIFFERENT PROMPT TEXT: the encoder cache is keyed on the media hash
         alone, so the attacker should get the speed-up WITHOUT knowing the victim's
         prompt. If so the multimodal channel is strictly stronger than the text one.
      B. ONE PIXEL DIFFERENT: must NOT get the speed-up (bounds the channel to exact
         content -- exp2 showed this in the hasher, never end-to-end on TTFA).
      C. DEEPER EVICTION: drive the 7.2 ms residual from the first mechanism run to ~0.

    Every probe uses a FRESH salt, so both arms of every comparison are KV-cache misses
    and the KV prefix cache cannot explain any difference."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    N = int(os.getenv("N", "15"))
    FLOOD = int(os.getenv("FLOOD", "1200"))

    def arr(seed):
        return np.random.default_rng(seed).integers(0, 256, (336, 336, 3), dtype="uint8")

    def enc(a):
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    p = serve(model, extra=("--max-num-batched-tokens", "2048"))
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, content(enc(arr(999)), "warmup"), None)
        out = {"meta": {**vllm_meta(), "model": model, "n_per_arm": N, "flood_images": FLOOD}}

        # ---- A. same image, different surrounding prompt text
        same_img, unseen_img = [], []
        for i in range(N):
            X = enc(arr(1000 + i))
            Y = enc(arr(2000 + i))
            victim_txt = "summarise this figure in detail for the report number %d" % i
            attacker_txt = "what objects appear here? question id %d" % i
            probe(cli, model, content(X, victim_txt), "a-v-%d" % i)      # victim sends (X, victim_txt)
            same_img.append(probe(cli, model, content(X, attacker_txt), "a-s-%d" % i))
            unseen_img.append(probe(cli, model, content(Y, attacker_txt), "a-u-%d" % i))
        out["same_image_different_prompt"] = {
            "victim_prompt_example": "summarise this figure in detail for the report number i",
            "attacker_prompt_example": "what objects appear here? question id i",
            "victim_image_attacker_prompt": gt(same_img),
            "never_sent_image_attacker_prompt": gt(unseen_img),
            "detector": separation([r["ttft_ms"] for r in same_img],
                                   [r["ttft_ms"] for r in unseen_img], "cached_content", "unseen"),
            "kv_cached_tokens_identical": (same_img[0].get("cached_tokens")
                                           == unseen_img[0].get("cached_tokens")),
        }
        json.dump(out, open("/out/exp3_mm_extra.json", "w"), indent=2); OUT.commit()

        # ---- B. one pixel different
        onepx, ctl = [], []
        for i in range(N):
            X = arr(3000 + i)
            Xp = X.copy(); Xp[10 + i, 20, 1] = (int(Xp[10 + i, 20, 1]) + 1) % 256   # single channel, single pixel
            probe(cli, model, content(enc(X), "describe"), "b-v-%d" % i)
            onepx.append(probe(cli, model, content(enc(Xp), "describe"), "b-p-%d" % i))
            ctl.append(probe(cli, model, content(enc(arr(4000 + i)), "describe"), "b-c-%d" % i))
        out["one_pixel_different"] = {
            "victim_image_resubmitted": gt(onepx),
            "never_sent_image": gt(ctl),
            "detector": separation([r["ttft_ms"] for r in onepx],
                                   [r["ttft_ms"] for r in ctl], "one_pixel_off", "unseen"),
            "identical_resend_for_contrast": None,
            "reading": ("a 1-pixel change must look like an unseen image (AUC ~0.5, delta ~0); "
                        "compared against the byte-identical resend measured in exp3_mm.json"),
        }
        json.dump(out, open("/out/exp3_mm_extra.json", "w"), indent=2); OUT.commit()

        # ---- C. deeper eviction
        Z = enc(arr(31337))
        probe(cli, model, content(Z, "describe"), "ev-v")
        before = probe(cli, model, content(Z, "describe"), "ev-before")
        for k in range(FLOOD):
            probe(cli, model, content(enc(arr(50000 + k)), "describe"), "flood-%d" % k)
        after = probe(cli, model, content(Z, "describe"), "ev-after")
        near_ctl = probe(cli, model, content(enc(arr(60000)), "describe"), "ev-ctl")
        # and confirm the victim's own re-send is still fast (cache still works at all)
        re_warm = probe(cli, model, content(Z, "describe"), "ev-v2")
        re_hit = probe(cli, model, content(Z, "describe"), "ev-v2")
        out["deep_eviction"] = {
            "before_flood_victim_image": {"ttft_ms": before["ttft_ms"],
                                          "cached_tokens": before.get("cached_tokens")},
            "after_flood_victim_image": {"ttft_ms": after["ttft_ms"],
                                         "cached_tokens": after.get("cached_tokens")},
            "never_sent_control": {"ttft_ms": near_ctl["ttft_ms"],
                                   "cached_tokens": near_ctl.get("cached_tokens")},
            "advantage_before_ms": float(near_ctl["ttft_ms"] - before["ttft_ms"]),
            "advantage_after_ms": float(near_ctl["ttft_ms"] - after["ttft_ms"]),
            "residual_fraction": float((near_ctl["ttft_ms"] - after["ttft_ms"]) /
                                       max(near_ctl["ttft_ms"] - before["ttft_ms"], 1e-9)),
            "sanity_same_tenant_reuse_still_works": {
                "warm_ttft_ms": re_warm["ttft_ms"], "hit_ttft_ms": re_hit["ttft_ms"],
                "hit_cached_tokens": re_hit.get("cached_tokens")},
        }
        json.dump(out, open("/out/exp3_mm_extra.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out, indent=2)); return out
    finally:
        stop(p)


@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=90 * 60)
def audio_extra():
    """Second pass on the audio channel: how much re-encoding does it survive?

    exp2 says the media key is a hash of the DECODED array, so the prediction is:
      * a different CONTAINER carrying identical PCM (WAV -> FLAC, lossless) should still
        reuse the original entry -> the channel is robust to container changes;
      * a LOSSY re-encode (OGG/Vorbis, MP3) should NOT -> the channel needs a bit-exact
        decode path.
    That is the practical bound on the audio channel: it confirms 'someone sent exactly
    this file', and dies at the first lossy step of a real pipeline.

    Each variant is judged by its FIRST send (does it reuse the original clip's entry);
    self-repeats are recorded only as a sanity check."""
    exec(CLIENT_SRC, globals())
    import base64, io, soundfile as sf
    model = os.getenv("AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")

    def content(url, txt="transcribe"):
        return [{"type": "text", "text": txt}, {"type": "audio_url", "audio_url": {"url": url}}]

    def to_b64(arr, sr, fmt, subtype=None):
        b = io.BytesIO()
        kw = {"format": fmt}
        if subtype:
            kw["subtype"] = subtype
        sf.write(b, arr, sr, **kw)
        return "data:audio/%s;base64,%s" % (fmt.lower(), base64.b64encode(b.getvalue()).decode())

    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        sr = 16000
        base = np.random.default_rng(11).standard_normal(sr).astype("float32") * 0.1
        probe(cli, model, content(to_b64(np.zeros(sr, "float32"), sr, "WAV", "PCM_16")), None)

        out = {"meta": {**vllm_meta(), "model": model, "clip_seconds": 1.0, "sample_rate": sr}}

        # the ORIGINAL the victim sends, and an unseen control, to establish the baseline
        wav = to_b64(base, sr, "WAV", "PCM_16")
        unseen = to_b64(np.random.default_rng(12).standard_normal(sr).astype("float32") * 0.1,
                        sr, "WAV", "PCM_16")
        probe(cli, model, content(wav), None)                     # victim sends the original
        out["baseline_identical_wav_resend"] = {
            **gt([probe(cli, model, content(wav), None)]), "meaning": "reuses the original entry"}
        out["baseline_never_sent_wav"] = {
            **gt([probe(cli, model, content(unseen), None)]), "meaning": "template block only"}

        variants = {}
        for label, fmt, sub, arr in (
                ("flac_lossless_same_pcm", "FLAC", None, base),
                ("ogg_vorbis_lossy", "OGG", "VORBIS", base),
                ("mp3_lossy", "MP3", None, base),
        ):
            try:
                data = to_b64(arr, sr, fmt, sub)
            except Exception as e:
                variants[label] = {"error": "%s encoder unavailable in this image: %s" % (fmt, e)}
                continue
            first = probe(cli, model, content(data), None)          # vs the ORIGINAL's key
            again = probe(cli, model, content(data), None)          # self-repeat (sanity)
            tpl = out["baseline_never_sent_wav"]["cached_tokens_p50"]
            variants[label] = {
                "first_send_cached_tokens": first.get("cached_tokens"),
                "first_send_ttft_ms": first["ttft_ms"],
                "self_repeat_cached_tokens": again.get("cached_tokens"),
                "reuses_original_entry": bool((first.get("cached_tokens") or 0) > tpl + 8),
                "template_baseline_cached_tokens": tpl,
            }
        out["reencode_variants"] = variants
        json.dump(out, open("/out/exp3_audio_extra.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out, indent=2)); return out
    finally:
        stop(p)


@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=60 * 60)
def text_recovery():
    """End-to-end recovery of a templated 3-field secret from a live vLLM, using ONLY TTFT.

    This is the link between exp4 (a query-complexity model over an IDEAL oracle) and exp3
    (proof that the noisy oracle is reliable). Nothing here uses /metrics or any privileged
    view: the attacker sees one number per request.

    Victim prompt:   HEAD [A] LONG1 [B] LONG2 [C] LONG3
    The attacker knows the fixed parts and only that each field is one of V candidates.
    Each field is followed by a long KNOWN run, so guessing a field correctly extends the
    matched cached prefix by that whole run -> a large TTFT difference.

    THE POINT THAT MAKES OR BREAKS THIS EXPERIMENT: every trial must use FRESH template
    material. An earlier version reused one HEAD/LONG* across trials, so after trial 0 the
    attacker's own probes were already in the cache and every candidate came back as a
    self-hit (all 12 candidates within 3 ms of each other, 0/6 fields recovered). With
    fresh material each trial, trial 0 recovered 3/3 fields with margins of 88.5/49.8/14.1 ms
    -- i.e. the attack works, and the earlier failure was entirely self-inflicted.

    Each candidate is still probed exactly ONCE within a trial: a second probe would hit the
    attacker's own entry, which is exactly why the query cost is V per field (exp4's model).
    """
    exec(CLIENT_SRC, globals())
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    V = int(os.getenv("V", "12"))
    TRIALS = int(os.getenv("TRIALS", "6"))
    out = {"meta": {**vllm_meta(), "model": model, "v_candidates_per_field": V,
                    "fields": 3, "trials": TRIALS,
                    "design": "HEAD [A] LONG1 [B] LONG2 [C] LONG3; fresh template per trial"}}
    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, "warmup", None)
        runs = []
        for trial in range(TRIALS):
            # FRESH template material every trial (see the docstring)
            HEAD = make_prefix(model, 32, 61 + 1000 * trial)[0]
            rn = {n: make_prefix(model, n, 100 + n + 7000 * trial)[0] for n in (1024, 512, 256)}
            pre_ctx = {0: HEAD, 1: rn[1024], 2: rn[512]}
            cands = {f: pick_candidates(model, pre_ctx[f], V, 4000 * (f + 1) + 50000 * trial)
                     for f in range(3)}
            if min(len(c) for c in cands.values()) < V:
                runs.append({"trial": trial, "skipped": "not enough 1-token candidates"})
                continue

            def build(known, upto):
                parts = [HEAD, known[0], rn[1024]]
                if upto >= 2:
                    parts += [known[1], rn[512]]
                if upto >= 3:
                    parts += [known[2], rn[256]]
                return " ".join(parts)

            rng = np.random.default_rng(5000 + trial)
            secret = [cands[f][int(rng.integers(0, V))] for f in range(3)]
            victim_full = build(secret, 3)

            # --- salt OFF
            probe(cli, model, victim_full, None)
            rec, det = recover_fields(build, cands, V,
                                      lambda t: _plain(cli, model, t, None))
            for f, d in enumerate(det):
                d["truth"] = secret[f]; d["correct"] = d["guess"] == secret[f]
                d["truth_ttft_ms"] = d["all_times_ms"].get(secret[f])
                d["truth_rank"] = (sorted(d["all_times_ms"], key=d["all_times_ms"].get)
                                   .index(secret[f]) + 1)

            # --- salt ON negative control (same texts, different key space)
            sv, sa = "rec2-victim-%d" % trial, "rec2-attacker-%d" % trial
            probe(cli, model, victim_full, sv)
            probe(cli, model, "attacker-recovery-prime-%d" % trial, sa)
            rec_on, det_on = recover_fields(build, cands, V,
                                            lambda t: _plain(cli, model, t, sa))
            for f, d in enumerate(det_on):
                d["truth"] = secret[f]; d["correct"] = d["guess"] == secret[f]
                d["truth_ttft_ms"] = d["all_times_ms"].get(secret[f])
                d["truth_rank"] = (sorted(d["all_times_ms"], key=d["all_times_ms"].get)
                                   .index(secret[f]) + 1)

            runs.append({"trial": trial, "secret": secret,
                         "salt_off": {"recovered": rec, "correct": rec == secret,
                                      "per_field": det},
                         "salt_on": {"recovered": rec_on, "correct": rec_on == secret,
                                     "per_field": det_on}})
            out["runs"] = runs
            out["summary"] = _rec_summary(runs, V)
            json.dump(out, open("/out/exp3_text_recovery.json", "w"), indent=2); OUT.commit()
        out["runs"] = runs
        out["summary"] = _rec_summary(runs, V)
        json.dump(out, open("/out/exp3_text_recovery.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out["summary"], indent=2))
        return out
    finally:
        stop(p)


@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=90 * 60)
def mm_evict():
    """Last loose end: the deep-eviction run left a ~6 ms residual (22% of the effect) that
    did NOT shrink when the flood grew from 400 to 1200 images. That is the signature of a
    SECOND content-keyed cache the flood cannot reach.

    There are two:
      * the MM ENCODER cache -- token-budgeted (a few thousand encoder tokens), so a few
        hundred distinct images evict it;
      * the MM PROCESSOR cache -- `mm_processor_cache_gb`, default **4 GiB**, keyed on the
        media hash. 1200 images of 336x336x3 PNG are only ~0.4 GB, nowhere near evicting it.

    Test: disable the processor cache, flood only the encoder cache, and see whether the
    attacker's advantage goes to ~0. That separates the two caches' contributions."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    FLOOD = int(os.getenv("FLOOD", "800"))

    def b64img(seed=None):
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        a = rng.integers(0, 256, (336, 336, 3), dtype="uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt="describe"):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    out = {"meta": {**vllm_meta(), "model": model, "flood_images": FLOOD,
                    "server_flags": "--max-num-batched-tokens 2048 --mm-processor-cache-gb 0"}}
    p = serve(model, extra=("--max-num-batched-tokens", "2048", "--mm-processor-cache-gb", "0"))
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, content(b64img(999)), None)
        Z = b64img(31337)
        probe(cli, model, content(Z), "ev-v")                       # victim caches Z
        before = probe(cli, model, content(Z), "ev-before")         # encoder HIT expected
        ctl_before = probe(cli, model, content(b64img(40001)), "ev-ctl1")
        for k in range(FLOOD):
            probe(cli, model, content(b64img(70000 + k)), "flood-%d" % k)
        after = probe(cli, model, content(Z), "ev-after")
        ctl_after = probe(cli, model, content(b64img(80001)), "ev-ctl2")
        out["result"] = {
            "processor_cache_disabled": True,
            "before_flood": {"victim_image_ms": before["ttft_ms"],
                             "never_sent_ms": ctl_before["ttft_ms"],
                             "advantage_ms": float(ctl_before["ttft_ms"] - before["ttft_ms"])},
            "after_flood": {"victim_image_ms": after["ttft_ms"],
                            "never_sent_ms": ctl_after["ttft_ms"],
                            "advantage_ms": float(ctl_after["ttft_ms"] - after["ttft_ms"])},
            "residual_fraction": float((ctl_after["ttft_ms"] - after["ttft_ms"]) /
                                       max(ctl_before["ttft_ms"] - before["ttft_ms"], 1e-9)),
            "reading": ("advantage ~0 after the flood means the whole effect was the encoder "
                        "cache; a remaining advantage means something else still leaks"),
        }
        json.dump(out, open("/out/exp3_mm_evict.json", "w"), indent=2); OUT.commit()
        print(json.dumps(out, indent=2)); return out
    finally:
        stop(p)


@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=120 * 60)
def reproduce_gu():
    """FAITHFUL implementation of the experimental protocol of Gu et al., "Auditing Prompt
    Caching in Language Model APIs" (ICML 2025, arXiv:2502.07776), on our own vLLM.

    OWNERSHIP: reproducing that paper is ANOTHER TEAMMATE'S deliverable. This function is
    kept here as (a) an offerable reference implementation they can reuse or diff against,
    and (b) an optional internal baseline. It is NOT part of our extension's claims, and it
    has not been run to completion -- do not cite it as a reproduction.

    What is reproduced, matching the paper:
      * observable: time-to-first-token with max_tokens=1 (their Sec. 2.2);
      * CACHE MISS procedure: the attacker sends a fresh random prompt (their Sec. 3.1);
      * CACHE HIT procedure: the victim sends x, then the attacker sends x-tilde sharing
        exactly PrefixFraction x PromptLength tokens with x (their Sec. 3.2);
      * prompt distribution: PromptLength space-separated single English letters,
        uniform upper+lowercase -- their exact construction;
      * sample order randomised across the run (their Sec. 3.2);
      * detector: ONE-SIDED two-sample Kolmogorov-Smirnov, alternative="greater",
        H0: D_hit = D_miss (their Sec. 3.2), at their alpha = 1e-8;
      * metric: precision-recall curve + AVERAGE PRECISION, cache hits as the positive
        class (their Fig. 4/6 and Table 1).
      * their prefix-fraction ablation (Fig. 5): AP should fall to chance as the shared
        prefix shrinks. Their levels 2-4 use PrefixFraction = 0.95; level 1 uses 1.0.

    What deliberately differs (and why the numbers are NOT directly comparable to Table 1):
      * target: OUR OWN local vLLM v0.11.0 instead of 17 commercial black-box APIs;
      * NumSamples per condition: 100 here vs 250 there (kept down for cost);
      * no server-side timing header, no per-org level, no architecture inference.
    Not attempted: auditing third-party providers. That is the paper's own contribution
    and it is out of scope for a defensive study on our own instance."""
    exec(CLIENT_SRC, globals())
    from scipy.stats import ks_2samp as _ks
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    N = int(os.getenv("NGU", "100"))                    # paper uses 250
    L = int(os.getenv("PROMPT_LEN", "5000"))            # paper's PromptLength
    fracs = [float(x) for x in os.getenv("FRACS", "1.0,0.95,0.75,0.5,0.25,0.1").split(",")]
    ALPHA = 1e-8                                        # paper's significance level
    out = {"meta": {**vllm_meta(), "model": model, "paper": "arXiv:2502.07776 (ICML 2025)",
                    "prompt_length_tokens": L, "n_samples_per_condition": N,
                    "prefix_fractions": fracs, "alpha": ALPHA,
                    "detector": "one-sided two-sample KS, alternative='greater'",
                    "metric": "average precision from the precision-recall curve",
                    "paper_reference_values": {
                        "AP range on real APIs (their Table 1)": "0.71 - 0.90",
                        "AP figures (their Fig. 4)": "Perplexity 0.90, OpenAI GPT-4o mini 0.86",
                        "their NumSamples": 250, "their level-2+ PrefixFraction": 0.95},
                    "differences_from_paper": [
                        "target is our own local vLLM, not 17 commercial APIs",
                        "%d samples per condition vs their 250" % N,
                        "no server-side timing header; client-side timing only",
                        "NumVictimRequests = 1 (single-engine, no request routing, so one "
                        "victim send is enough; the paper needed up to 25 for multi-server APIs)"]}}
    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        probe(cli, model, "warmup", None)
        by_frac = {}
        for pf in fracs:
            k = int(round(pf * L))
            hits, misses, verify = [], [], {"prefix_verified": 0, "prefix_failed": 0,
                                            "hit_cached_tokens": [], "miss_cached_tokens": []}
            # randomise the order of the two procedures, as the paper does
            order = ["hit", "miss"] * N
            np.random.default_rng(1234).shuffle(order)
            victim_cache = {}
            for j, kind in enumerate(order):
                if kind == "hit":
                    i = j // 2
                    if i not in victim_cache:
                        vtext, _ = letter_prompt(model, L, 10000 + i)
                        probe(cli, model, vtext, None)              # victim caches x
                        vid = _tokenize_ids(model, vtext)
                        victim_cache[i] = (vtext, vid)
                    vtext, vid = victim_cache[i]
                    atext, _ = letter_prompt(model, L, 20000 + i, prefix_letters=None)
                    # rebuild the attacker prompt from the victim's own leading letters
                    letters = vtext.split(" ")[:k]
                    atext, n_a = letter_prompt(model, L, 20000 + i, prefix_letters=letters)
                    aid = _tokenize_ids(model, atext)
                    ok = (len(aid) == len(vid)) and (aid[:k] == vid[:k]) and (aid[k] != vid[k])
                    verify["prefix_verified" if ok else "prefix_failed"] += 1
                    r = probe(cli, model, atext, None)
                    hits.append(r["ttft_ms"])
                    verify["hit_cached_tokens"].append(r.get("cached_tokens"))
                else:
                    mtext, _ = letter_prompt(model, L, 30000 + j)
                    r = probe(cli, model, mtext, None)
                    misses.append(r["ttft_ms"])
                    verify["miss_cached_tokens"].append(r.get("cached_tokens"))
            hits = np.asarray(hits, float); misses = np.asarray(misses, float)
            ks = _ks(hits, misses, alternative="greater")       # the paper's one-sided test
            # average precision, cache hits = positive class, score = -latency
            sc = np.r_[-hits, -misses]; y = np.r_[np.ones(len(hits)), np.zeros(len(misses))]
            o = np.argsort(-sc); y = y[o]
            tp = np.cumsum(y); fp = np.cumsum(1 - y)
            prec = tp / np.maximum(tp + fp, 1e-12); rec = tp / len(hits)
            ap = float(np.sum(np.diff(np.r_[0.0, rec]) * np.r_[1.0, prec][1:]))
            # ROC AUC for reference (paper does not use ROC)
            tpr = rec; fpr = fp / len(misses)
            roc_auc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
            by_frac[str(pf)] = {
                "prefix_fraction": pf, "shared_prefix_tokens": k,
                "n_hit": len(hits), "n_miss": len(misses),
                "hit_p50_ms": float(np.median(hits)), "miss_p50_ms": float(np.median(misses)),
                "delta_p50_ms": float(np.median(misses) - np.median(hits)),
                "ks_statistic": float(ks.statistic), "ks_p_one_sided": float(ks.pvalue),
                "significant_at_alpha_1e_8": bool(ks.pvalue < ALPHA),
                "average_precision": ap, "roc_auc_reference": roc_auc,
                "median_cached_tokens_hit": float(np.median(verify["hit_cached_tokens"]))
                if verify["hit_cached_tokens"] else None,
                "median_cached_tokens_miss": float(np.median(verify["miss_cached_tokens"]))
                if verify["miss_cached_tokens"] else None,
                "prefix_verification": {"verified": verify["prefix_verified"],
                                        "failed": verify["prefix_failed"]},
                "raw_hit_ttft_ms": [float(x) for x in hits],
                "raw_miss_ttft_ms": [float(x) for x in misses],
            }
            out["by_prefix_fraction"] = by_frac
            json.dump(out, open("/out/exp3_reproduce_gu.json", "w"), indent=2); OUT.commit()
            # compact progress line
            print("pf=%.2f k=%5d  AP=%.3f  KS=%.3f p=%.2e  delta=%+.1f ms" %
                  (pf, k, ap, ks.statistic, ks.pvalue, by_frac[str(pf)]["delta_p50_ms"]))
        print(json.dumps({k: {kk: vv for kk, vv in v.items()
                              if kk not in ("raw_hit_ttft_ms", "raw_miss_ttft_ms")}
                          for k, v in out["by_prefix_fraction"].items()}, indent=2))
        return out
    finally:
        stop(p)


# =====================================================================================
# REPEATABILITY STUDY
#
# THREATS_TO_VALIDITY.md item 2: every exp3 conclusion was measured ONCE. The load test
# then produced AUC 0.000 in one run and 0.212 in another at the same 8 threads, so
# single-run reliability is a real problem, not a hypothetical one.
#
# Design: K independent repetitions, and each repetition RESTARTS the vLLM server (fresh
# engine, fresh cache, fresh CUDA graphs, fresh routing state). That captures
# session-to-session variability, not just timing jitter within one server. Prompts are
# generated from FIXED seeds across reps, so any spread is attributable to the system
# rather than to the prompts -- prompt-level variance is deliberately controlled out.
#
# For each metric we report the per-rep values plus mean/std, and compare the across-rep
# spread with the within-rep bootstrap CI. If across-rep >> within-rep, single-run
# conclusions are not trustworthy at that sample size.
# =====================================================================================

def _spread(vals):
    v = [float(x) for x in vals]
    return {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(np.min(v)), "max": float(np.max(v))}


@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=120 * 60)
def repeat_text():
    """K restarts; each rep measures (a) the open channel at 1024 tokens, (b) the same
    thing under 8 concurrent load threads, (c) the salt-ON attack at 1024 tokens."""
    exec(CLIENT_SRC, globals())
    model = os.getenv("TEXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    K = int(os.getenv("K", "3"))
    N = int(os.getenv("NREP", "30"))
    L = 1024
    out = {"meta": {**vllm_meta(), "model": model, "repetitions": K, "n_per_arm": N,
                    "prefix_tokens": L,
                    "design": "each repetition restarts the vLLM server; prompts fixed across reps"}}
    reps = []
    for rep in range(K):
        p = serve(model)
        try:
            cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
            probe(cli, model, "warmup", None)
            r = {"rep": rep}
            # --- (a) open channel, idle
            hv, mv = [], []
            for i in range(N):
                v, _ = make_prefix(model, L, 7919 * L + i)
                u, _ = make_prefix(model, L, 500000 + 7919 * L + i)
                probe(cli, model, v, None)
                hv.append(probe(cli, model, v, None)["ttft_ms"])
                mv.append(probe(cli, model, u, None)["ttft_ms"])
            r["idle_open_channel"] = separation(hv, mv, "hit", "miss")
            r["idle_open_channel"]["raw_hit"] = hv; r["idle_open_channel"]["raw_miss"] = mv
            # --- (b) open channel under 8 concurrent load threads (where runs disagreed)
            stop_flag = {"stop": False}
            import threading
            load_prompt = make_prefix(model, 256, 4242)[0]
            def worker():
                while not stop_flag["stop"]:
                    try: _plain(cli, model, load_prompt, None)
                    except Exception: pass
            ths = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
            for th in ths: th.start()
            time.sleep(5)
            h8, m8 = [], []
            for i in range(N):
                v, _ = make_prefix(model, L, 31000 + i)
                u, _ = make_prefix(model, L, 910000 + i)
                _plain(cli, model, v, None)          # warm (do not use probe(): under load the
                h8.append(_plain(cli, model, v, None))     # per-request counter delta is invalid)
                m8.append(_plain(cli, model, u, None))
            stop_flag["stop"] = True
            for th in ths: th.join(timeout=5)
            time.sleep(2)
            r["load8_open_channel"] = separation(h8, m8, "hit", "miss")
            r["load8_open_channel"]["raw_hit"] = h8; r["load8_open_channel"]["raw_miss"] = m8
            # --- (c) salt-ON attack at the same length
            oh, om = [], []
            for i in range(N):
                v, _ = make_prefix(model, L, 44000 + i)
                u, _ = make_prefix(model, L, 66000 + i)
                sv, sa = "rv-%d-%d" % (rep, i), "ra-%d-%d" % (rep, i)
                probe(cli, model, v, sv)                       # victim caches under its salt
                probe(cli, model, "prime-%d-%d" % (rep, i), sa)
                oh.append(probe(cli, model, v, sa)["ttft_ms"])   # attacker probes victim content
                om.append(probe(cli, model, u, sa)["ttft_ms"])   # attacker probes unseen content
            r["salt_on_attack"] = separation(oh, om, "cached_content", "unseen")
            r["salt_on_attack"]["raw_hit"] = oh; r["salt_on_attack"]["raw_miss"] = om
            reps.append(r)
        finally:
            stop(p)
        out["reps"] = reps
        json.dump(out, open("/out/exp3_repeat_text.json", "w"), indent=2); OUT.commit()
        print("rep %d: idle AUC %.3f AP %.3f | load8 AUC %.3f AP %.3f | saltON AUC %.3f AP %.3f" %
              (rep, reps[-1]["idle_open_channel"]["auc"], reps[-1]["idle_open_channel"]["average_precision"],
               reps[-1]["load8_open_channel"]["auc"], reps[-1]["load8_open_channel"]["average_precision"],
               reps[-1]["salt_on_attack"]["auc"], reps[-1]["salt_on_attack"]["average_precision"]))
    out["summary"] = {
        k: {"auc": _spread([r[k]["auc"] for r in reps]),
            "average_precision": _spread([r[k]["average_precision"] for r in reps]),
            "delta_p50_ms": _spread([r[k]["delta_p50_ms"] for r in reps]),
            "ks_stat": _spread([r[k]["ks_stat"] for r in reps])}
        for k in ("idle_open_channel", "load8_open_channel", "salt_on_attack")}
    json.dump(out, open("/out/exp3_repeat_text.json", "w"), indent=2); OUT.commit()
    print(json.dumps(out["summary"], indent=2))
    return out


@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=150 * 60)
def repeat_mm():
    """K restarts; each rep measures the headline image result twice over:
    (a) attacker probes the victim's image vs a never-sent image (fresh salts, primed),
    (b) attacker probes the victim's image under a DIFFERENT prompt text."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    K = int(os.getenv("K", "3"))
    N = int(os.getenv("NREP", "15"))

    def b64img(seed=None):
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        a = rng.integers(0, 256, (336, 336, 3), dtype="uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt="describe"):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    out = {"meta": {**vllm_meta(), "model": model, "repetitions": K, "n_per_arm": N,
                    "design": "each repetition restarts the vLLM server; image seeds fixed across reps"}}
    reps = []
    for rep in range(K):
        p = serve(model)
        try:
            cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
            probe(cli, model, content(b64img(999)), None)
            r = {"rep": rep}
            # (a) salt-ON attack
            oh, om = [], []
            for i in range(N):
                X = b64img(1000 + i); Y = b64img(2000 + i)
                probe(cli, model, content(X), "rv-%d-%d" % (rep, i))
                probe(cli, model, content(b64img(3000 + i)), "ra-%d-%d" % (rep, i))   # prime
                oh.append(probe(cli, model, content(X), "ra-%d-%d" % (rep, i))["ttft_ms"])
                om.append(probe(cli, model, content(Y), "ra-%d-%d" % (rep, i))["ttft_ms"])
            r["salt_on_attack"] = separation(oh, om, "cached_content", "unseen")
            r["salt_on_attack"]["raw_hit"] = oh; r["salt_on_attack"]["raw_miss"] = om
            # (b) same image, different prompt text
            sh, sm = [], []
            for i in range(N):
                X = b64img(5000 + i); Y = b64img(6000 + i)
                probe(cli, model, content(X, "summarise figure %d" % i), "pv-%d-%d" % (rep, i))
                sh.append(probe(cli, model, content(X, "what objects? q%d" % i),
                                "ps-%d-%d" % (rep, i))["ttft_ms"])
                sm.append(probe(cli, model, content(Y, "what objects? q%d" % i),
                                "pu-%d-%d" % (rep, i))["ttft_ms"])
            r["same_image_different_prompt"] = separation(sh, sm, "cached_content", "unseen")
            r["same_image_different_prompt"]["raw_hit"] = sh
            r["same_image_different_prompt"]["raw_miss"] = sm
            reps.append(r)
        finally:
            stop(p)
        out["reps"] = reps
        json.dump(out, open("/out/exp3_repeat_mm.json", "w"), indent=2); OUT.commit()
        print("rep %d: saltON AUC %.3f AP %.3f delta %.1f | diffPrompt AUC %.3f AP %.3f delta %.1f" %
              (rep, reps[-1]["salt_on_attack"]["auc"], reps[-1]["salt_on_attack"]["average_precision"],
               reps[-1]["salt_on_attack"]["delta_p50_ms"],
               reps[-1]["same_image_different_prompt"]["auc"],
               reps[-1]["same_image_different_prompt"]["average_precision"],
               reps[-1]["same_image_different_prompt"]["delta_p50_ms"]))
    out["summary"] = {
        k: {"auc": _spread([r[k]["auc"] for r in reps]),
            "average_precision": _spread([r[k]["average_precision"] for r in reps]),
            "delta_p50_ms": _spread([r[k]["delta_p50_ms"] for r in reps]),
            "ks_stat": _spread([r[k]["ks_stat"] for r in reps])}
        for k in ("salt_on_attack", "same_image_different_prompt")}
    json.dump(out, open("/out/exp3_repeat_mm.json", "w"), indent=2); OUT.commit()
    print(json.dumps(out["summary"], indent=2))
    return out


@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=150 * 60)
def repeat_audio():
    """K restarts; each rep measures the audio salt-ON attack and re-checks the FLAC
    (lossless container) verdict -- i.e. the two audio claims we care about."""
    exec(CLIENT_SRC, globals())
    import base64, io, soundfile as sf
    model = os.getenv("AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
    K = int(os.getenv("K", "3"))
    N = int(os.getenv("NREP", "15"))

    def wav_b64(arr, sr=16000, fmt="WAV", sub="PCM_16"):
        b = io.BytesIO(); kw = {"format": fmt}
        if sub: kw["subtype"] = sub
        sf.write(b, arr, sr, **kw)
        return "data:audio/%s;base64,%s" % (fmt.lower(), base64.b64encode(b.getvalue()).decode())

    def content(url, txt="transcribe"):
        return [{"type": "text", "text": txt}, {"type": "audio_url", "audio_url": {"url": url}}]

    out = {"meta": {**vllm_meta(), "model": model, "repetitions": K, "n_per_arm": N}}
    reps = []
    for rep in range(K):
        p = serve(model)
        try:
            cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
            sr = 16000
            probe(cli, model, content(wav_b64(np.zeros(sr, "float32"))), None)
            r = {"rep": rep}
            oh, om = [], []
            for i in range(N):
                clip = (np.random.default_rng(7000 + i).standard_normal(sr).astype("float32") * 0.1)
                unseen = (np.random.default_rng(8000 + i).standard_normal(sr).astype("float32") * 0.1)
                a, b = wav_b64(clip), wav_b64(unseen)
                probe(cli, model, content(a), "rv-%d-%d" % (rep, i))
                probe(cli, model, content(wav_b64(np.zeros(sr, "float32"))), "ra-%d-%d" % (rep, i))
                oh.append(probe(cli, model, content(a), "ra-%d-%d" % (rep, i))["ttft_ms"])
                om.append(probe(cli, model, content(b), "ra-%d-%d" % (rep, i))["ttft_ms"])
            r["salt_on_attack"] = separation(oh, om, "cached_content", "unseen")
            r["salt_on_attack"]["raw_hit"] = oh; r["salt_on_attack"]["raw_miss"] = om
            # FLAC verdict: does the first send reuse the ORIGINAL's entry?
            base = (np.random.default_rng(9000 + rep).standard_normal(sr).astype("float32") * 0.1)
            wav = wav_b64(base)
            probe(cli, model, content(wav), None)                     # victim sends the original
            wav_first = probe(cli, model, content(wav), None).get("cached_tokens")
            tpl = probe(cli, model, content(wav_b64(
                np.random.default_rng(9500 + rep).standard_normal(sr).astype("float32") * 0.1)),
                None).get("cached_tokens")
            try:
                flac_first = probe(cli, model, content(wav_b64(base, fmt="FLAC", sub=None)),
                                   None).get("cached_tokens")
                flac_err = None
            except Exception as e:
                flac_first, flac_err = None, str(e)
            r["flac"] = {"template_baseline_cached_tokens": tpl,
                         "identical_wav_first_send_cached_tokens": wav_first,
                         "flac_first_send_cached_tokens": flac_first, "error": flac_err,
                         "flac_reuses_original": bool(flac_first is not None and tpl is not None
                                                      and flac_first > tpl + 8)}
            reps.append(r)
        finally:
            stop(p)
        out["reps"] = reps
        json.dump(out, open("/out/exp3_repeat_audio.json", "w"), indent=2); OUT.commit()
        print("rep %d: saltON AUC %.3f AP %.3f delta %.1f | FLAC reuses original: %s" %
              (rep, reps[-1]["salt_on_attack"]["auc"], reps[-1]["salt_on_attack"]["average_precision"],
               reps[-1]["salt_on_attack"]["delta_p50_ms"], reps[-1]["flac"]["flac_reuses_original"]))
    out["summary"] = {
        "salt_on_attack": {"auc": _spread([r["salt_on_attack"]["auc"] for r in reps]),
                           "average_precision": _spread([r["salt_on_attack"]["average_precision"] for r in reps]),
                           "delta_p50_ms": _spread([r["salt_on_attack"]["delta_p50_ms"] for r in reps]),
                           "ks_stat": _spread([r["salt_on_attack"]["ks_stat"] for r in reps])},
        "flac_reuses_original_all_reps": [r["flac"]["flac_reuses_original"] for r in reps]}
    json.dump(out, open("/out/exp3_repeat_audio.json", "w"), indent=2); OUT.commit()
    print(json.dumps(out["summary"], indent=2))
    return out


@app.function(image=IMG, gpu="A100", volumes={"/out": OUT}, timeout=120 * 60)
def mm_direct():
    """THREATS_TO_VALIDITY.md item 1: turn the INFERRED mechanism into a DIRECTLY OBSERVED
    one.

    Upstream vLLM 0.11.0 exposes no metric for the multimodal processor/encoder caches
    (`vllm/v1/metrics/loggers.py` has none), so the earlier evidence was elimination plus
    perturbation. Here we instrument two stock files in place
    (`vllm/v1/core/encoder_cache_manager.py` and `vllm/multimodal/cache.py`) to emit
    per-request hit/miss counters, and read them around every request.

    Prediction to falsify: when the attacker probes an image the VICTIM already sent, the
    encoder cache reports a HIT even though the KV prefix cache is a miss (different salt);
    when it probes a never-sent image, the encoder cache reports a MISS. If that holds, the
    mechanism is observed rather than inferred.

    Three arms:
      A. default: does the counter follow the victim's image?
      B. `--mm-processor-cache-gb 0`: processor counters must go silent while the ENCODER
         counter still tracks the victim's image -> the encoder is the mechanism.
      C. after flooding the encoder cache, the counter must flip to MISS and the timing
         advantage must vanish -> counter and effect move together."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    N = int(os.getenv("N", "10"))
    FLOOD = int(os.getenv("FLOOD", "400"))

    def b64img(seed=None):
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        a = rng.integers(0, 256, (336, 336, 3), dtype="uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt="describe"):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    out = {"meta": {**vllm_meta(), "model": model, "n_per_arm": N, "flood_images": FLOOD,
                    "design": "per-request DIRECT encoder/processor cache counters, read around "
                              "each request (patched vLLM; two files, counters only)"}}

    def arm(tag, extra=(), flood_after=False):
        p = serve(model, extra=extra)
        try:
            cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
            probe(cli, model, content(b64img(999)), None)
            cached_arm, unseen_arm = [], []
            for i in range(N):
                X = b64img(1000 + i); Y = b64img(2000 + i)
                probe(cli, model, content(X), "%s-v-%d" % (tag, i))          # victim caches X
                probe(cli, model, content(b64img(3000 + i)), "%s-p-%d" % (tag, i))  # prime attacker salt
                cached_arm.append(probe_direct(cli, model, content(X), "%s-a-%d" % (tag, i)))
                unseen_arm.append(probe_direct(cli, model, content(Y), "%s-a-%d" % (tag, i)))
            res = {
                "victim_image_probe": {
                    "ttft_p50_ms": float(np.median([r["ttft_ms"] for r in cached_arm])),
                    "encoder_cache_hit_total": int(sum(r["encoder_cache_hit"] for r in cached_arm)),
                    "encoder_cache_miss_total": int(sum(r["encoder_cache_miss"] for r in cached_arm)),
                    "processor_cache_hit_total": int(sum(r["processor_cache_hit"] for r in cached_arm)),
                    "processor_cache_miss_total": int(sum(r["processor_cache_miss"] for r in cached_arm)),
                    "n": len(cached_arm)},
                "never_sent_image_probe": {
                    "ttft_p50_ms": float(np.median([r["ttft_ms"] for r in unseen_arm])),
                    "encoder_cache_hit_total": int(sum(r["encoder_cache_hit"] for r in unseen_arm)),
                    "encoder_cache_miss_total": int(sum(r["encoder_cache_miss"] for r in unseen_arm)),
                    "processor_cache_hit_total": int(sum(r["processor_cache_hit"] for r in unseen_arm)),
                    "processor_cache_miss_total": int(sum(r["processor_cache_miss"] for r in unseen_arm)),
                    "n": len(unseen_arm)},
                "ttft_advantage_ms": float(np.median([r["ttft_ms"] for r in unseen_arm])
                                           - np.median([r["ttft_ms"] for r in cached_arm])),
                "detector": separation([r["ttft_ms"] for r in cached_arm],
                                       [r["ttft_ms"] for r in unseen_arm], "cached", "unseen"),
            }
            if flood_after:
                Z = b64img(31337)
                probe(cli, model, content(Z), "ev-v")
                before = probe_direct(cli, model, content(Z), "ev-b")
                for k in range(FLOOD):
                    probe(cli, model, content(b64img(40000 + k)), "fl-%d" % k)
                after = probe_direct(cli, model, content(Z), "ev-a")
                ctl = probe_direct(cli, model, content(b64img(50000)), "ev-c")
                res["after_encoder_flood"] = {
                    "before": {k: before[k] for k in ("ttft_ms", "encoder_cache_hit", "encoder_cache_miss")},
                    "after": {k: after[k] for k in ("ttft_ms", "encoder_cache_hit", "encoder_cache_miss")},
                    "control": {k: ctl[k] for k in ("ttft_ms", "encoder_cache_hit", "encoder_cache_miss")},
                }
            return res
        finally:
            stop(p)

    out["patch_report"] = apply_vllm_patch()
    clear_pcache_stats()
    out["A_default"] = arm("A", flood_after=True)
    json.dump(out, open("/out/exp3_mm_direct.json", "w"), indent=2); OUT.commit()
    out["B_processor_cache_off"] = arm("B", extra=("--max-num-batched-tokens", "2048",
                                                   "--mm-processor-cache-gb", "0"))
    json.dump(out, open("/out/exp3_mm_direct.json", "w"), indent=2); OUT.commit()

    a, b = out["A_default"], out["B_processor_cache_off"]
    out["verdict"] = {
        "encoder_hits_on_victim_image_default": a["victim_image_probe"]["encoder_cache_hit_total"],
        "encoder_hits_on_never_sent_default": a["never_sent_image_probe"]["encoder_cache_hit_total"],
        "encoder_hits_on_victim_image_proc_cache_off": b["victim_image_probe"]["encoder_cache_hit_total"],
        "encoder_hits_on_never_sent_proc_cache_off": b["never_sent_image_probe"]["encoder_cache_hit_total"],
        "processor_counters_silent_when_disabled": (
            b["victim_image_probe"]["processor_cache_hit_total"] == 0
            and b["victim_image_probe"]["processor_cache_miss_total"] == 0),
        "mechanism_observed": bool(
            a["victim_image_probe"]["encoder_cache_hit_total"] > 0
            and a["never_sent_image_probe"]["encoder_cache_hit_total"] == 0
            and b["victim_image_probe"]["encoder_cache_hit_total"] > 0
            and b["never_sent_image_probe"]["encoder_cache_hit_total"] == 0),
    }
    json.dump(out, open("/out/exp3_mm_direct.json", "w"), indent=2); OUT.commit()
    print(json.dumps(out, indent=2))
    return out


@app.function(image=IMG, gpu="L4", volumes={"/out": OUT}, timeout=45 * 60)
def patch_check():
    """Cheap validation that the instrumented vLLM still runs and that the new counters
    actually move, BEFORE spending A100 time on mm_direct."""
    exec(CLIENT_SRC, globals())
    import base64, io
    from PIL import Image
    model = os.getenv("MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    checks = {"patch_report": apply_vllm_patch()}

    def b64img(seed):
        a = np.random.default_rng(seed).integers(0, 256, (336, 336, 3), dtype="uint8")
        b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG")
        return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

    def content(url, txt="describe"):
        return [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": url}}]

    clear_pcache_stats()
    p = serve(model)
    try:
        cli = OpenAI(base_url=BASE_URL + "/v1", api_key="x")
        first = probe_direct(cli, model, content(b64img(1)), None)     # cold: encoder MISS
        second = probe_direct(cli, model, content(b64img(1)), None)    # warm: encoder HIT
        third = probe_direct(cli, model, content(b64img(2)), None)     # new image: MISS
        checks["cold_request"] = first
        checks["repeat_request"] = second
        checks["new_image_request"] = third
        checks["counters_move"] = bool(second["encoder_cache_hit"] > 0
                                       and third["encoder_cache_hit"] == 0)
        checks["PASS"] = bool(all(checks["patch_report"].values()) and checks["counters_move"])
        print("PATCH_CHECK " + json.dumps(checks, indent=2))
        return checks
    finally:
        stop(p)


@app.local_entrypoint()
def main():
    print(json.dumps(text.remote(), indent=2))
