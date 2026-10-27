# vLLM prefix-cache timing side channel — findings (source-verified)

Extends Gu et al., *"Auditing Prompt Caching in Language Model APIs"* (ICML 2025,
arXiv:2502.07776) to **self-hosted vLLM**, and characterizes the parts the paper did not:
vLLM's own block-hash mechanism, the **image and audio** channels, and vLLM's built-in
defense (`cache_salt`).

> **Scope of this document (read first).** The paper's own experiment — a black-box audit of
> 17 commercial API providers — is **reproduced separately by another teammate**, and nothing
> here is that reproduction. This document is the **extension half**: it reuses the paper's
> observable (TTFT at `max_tokens=1`) and its detector family (two-sample KS; we report the
> paper's one-sided variant alongside our two-sided one), then goes where the paper did not.
>
> **The bridge between the two halves.** Gu et al.'s §6 proposes four mitigations and
> evaluates **none** of them. The first is *"only per-user caching should be allowed."*
> vLLM's `cache_salt` is the production implementation of exactly that mitigation — its own
> source says it salts the prefix cache *"to prevent an attacker to guess prompts in
> multi-user environments."* This document measures whether that mitigation works and what it
> costs. It works for text; it does **not** work for image or audio.

All hashing claims below are **verified byte-for-byte against the real vLLM v0.11.0 source**
(`verify_against_source.py` imports the real `MultiModalHasher` and diffs digests; block
hash reproduced from `vllm/utils/__init__.py::sha256`) — re-run on a v0.11.0 checkout, result
"ALL BYTE-EQUAL". Exp1/2/4 were re-run on CPU with real output (`RESULTS_*.txt`).

**The GPU timing numbers are now real**: exp3 ran on Modal (vLLM v0.11.0, block_size=16,
Qwen2.5-3B-Instruct on L4, Qwen2.5-VL-3B on A100, Qwen2-Audio-7B on A100). Every latency
below comes from `RESULTS_exp3_*.txt`, and every HIT/MISS is corroborated by the server's own
`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` counters read around each request.
No number in this document is estimated, extrapolated, or fabricated.

---

## ⚠️ Correction that changes the pitch — read first

The earlier draft (and the "vLLM's hash can be hacked" line from your teammate) rested on
vLLM defaulting to Python's **builtin non-crypto `hash()`**. **That is no longer true.** In
v0.11.0:

```
vllm/config/cache.py:  prefix_caching_hash_algo: Literal["sha256","sha256_cbor"] = "sha256"
vllm/utils/__init__.py: get_hash_fn_by_name() accepts ONLY "sha256"/"sha256_cbor"
```

So the block hash is **SHA-256 by default** and the builtin hash is not even selectable.
**Forged-hit / offline-collision attacks on the hash are infeasible on current vLLM.** The
"hack the weak hash" angle was correct for *old* vLLM only — do **not** build the proposal
on it, or a reviewer/teammate who checks v0.11.0 will sink it.

**Where the real contribution lives instead:** the timing side channel does **not depend on
hash strength at all** — a HIT reveals shared content whether the hash is SHA-256 or not.
The novelty sits on three things the literature/vLLM has not nailed down:

1. **Image & audio membership oracle** (multimodal timing channel) — least-covered, and the
   media-hash behavior is subtle (below).
2. **Reconstruction query-complexity** made precise: `V·n` linear, not `V^n` (exp4).
3. **cache_salt / NONE_HASH / PYTHONHASHSEED** defense trade-offs, measured (exp1 + exp3).

Prior art to cite and beat on the *text* channel (so you don't re-claim it): PrefixWall
(arXiv:2603.10726) and "Governing the KV Cache" (arXiv:2608.09225) already show text-prefix
timing on vLLM. Your delta is **multimodal + the defense pricing + the complexity model**.

---

## Exp1 — block-hash mechanism (CPU, source-verified) — `RESULTS_exp1.txt`

- **Hit condition.** A block-`k` HIT requires identical parent-chain **and** identical 16
  tokens **and** identical `extra_keys` (LoRA id / mm-hash / `cache_salt`). Resolution is
  the block: **16 tokens**; a partial trailing block is never cached.
- **Hash chain.** Flip one token in block 1 of a 64-token prompt → blocks 0 identical,
  blocks 1–3 all differ. The shared cached prefix is exactly the unchanged **leading**
  blocks. This left-to-right structure is what makes reconstruction cheap (exp4).
- **Determinism.** With SHA-256 the per-block function is fully deterministic; the **only**
  per-process randomness is `NONE_HASH` (the block-0 parent), which is `os.urandom(32)`
  unless `PYTHONHASHSEED` is set.
  - Within one running process: hashing is self-consistent → the timing channel **works out
    of the box**, no reproducibility needed.
  - Offline precomputation of a victim's hashes needs `NONE_HASH` → only possible if
    `PYTHONHASHSEED` is pinned.
  - **Defense inversion worth flagging:** pinning `PYTHONHASHSEED` (commonly done "for
    reproducibility") *removes the only entropy* and makes block-0 hashes predictable across
    restarts. A nice, non-obvious point for the report.
- **Cost (real microbench).** `sha256` (default) 1.57 M blocks/s (638 ns/block), `sha256_cbor`
  0.36 M/s (2.75 µs), builtin (historical) 13.3 M/s (75 ns). *Absolute* rates are
  machine/CPython dependent — the cloud session measured 0.77 / 0.17 / 5.5 M/s on its CPU;
  the ratios and the conclusion are what matter. Per-block hashing is negligible vs prefill
  FLOPs → "SHA-256 by default" is essentially free; the defense cost is *not* in hashing but
  in lost cache reuse (that's the TTFT number exp3 measures).
- Truncated-width collision demo (24/32-bit) matches the birthday bound (4.7k vs 4.1k;
  55k vs 65k) — shown only to illustrate the mechanism; **not** applicable to the 256-bit
  default.

## Exp2 — multimodal identifier (CPU, byte-equal to real `MultiModalHasher`) — `RESULTS_exp2.txt`

Real algorithm (corrected):
```
identifier = blake3( model_id, <media>, **hf_processor_mm_kwargs, **tokenization_kwargs )
```
For an image, `<media>` is the **decoded PIL image** (`mode` + `np.asarray(img)` pixels +
palette) — the raw decoded pixels, not the post-processor feature tensor.

- Identical decoded pixels + same model + same kwargs → identical id (**HIT**). Any 1-pixel
  bit flip → different id (avalanche 59/64 hex chars). ⇒ **exact-content membership oracle.**
- **Cross-model isolation is automatic:** different `model_id` → different id. The oracle
  only confirms media submitted to the *same* deployment.
- **Processor-kwargs sensitivity:** any change in processor/tokenization kwargs changes the
  id. This is also the lever for the audio determinism question.
- Two real override paths worth a sentence: an **EXIF ImageID UUID** on the image is used
  *instead of* pixels; callers can pass explicit `mm_uuids` to skip content hashing.
- **Audio:** same hasher over the audio array. Sub-perceptual change (±1e-6 one sample) or a
  dtype change (float32→float64) already misses. ⇒ two recordings of "the same" clip collide
  **only if the full decode path is byte-identical** (container→PCM, resample rate, channel
  mix, trim, dtype). Bit-exact re-send hits; a re-encode generally won't. `modal_exp3_ttft.py::audio`
  measures whether a real re-encode still hits — that bounds the audio channel's practicality.
- **Capability statement (put this in the paper):** the multimodal channel is a
  **membership/confirmation oracle on a *known* image/audio** ("did someone submit exactly
  this?"), **not** a reconstruction channel for unknown media.

## Exp4 — reconstruction query complexity (CPU, exact oracle) — `RESULTS_exp4.txt`

Separates the *information* question from the *noisy timing* question. Given a reliable
HIT/MISS oracle (what exp3 validates), how expensive is recovery?

- **Low-entropy secret in a shared prefix** (one unknown token/block from V candidates):
  mean **45.6 / 401 / 4331** queries for V = 10 / 100 / 1000 over 8 blocks. **Linear in V
  and in #blocks.**
- **Why linear, not exponential:** the hash chain lets the oracle confirm each block
  independently once the prefix before it is fixed, collapsing `V^n → V·n`. At V=50, 6
  blocks: naive space 15.6 B vs **199** oracle queries — a ~**78,000,000×** reduction.
- **High-entropy secrets are safe:** ≥2 unknown high-entropy tokens/block → ~10⁹+
  queries/block → infeasible. **The channel threatens templated / low-entropy secrets**
  (system-prompt fields, PINs, ids, membership of a known document), not arbitrary content.
- **Defense mapping:** `cache_salt` per tenant/session → every probe misses → oracle
  destroyed (cost → ∞). Rate-limiting only scales the `V·n` linearly (slows, doesn't stop).

## Exp3 — GPU timing, REAL NUMBERS — `modal_exp3_ttft.py`, `RESULTS_exp3_text.txt`

Ran on Modal. vLLM 0.11.0, `block_size=16`, `--enable-prefix-caching`.
Detector = two-sample **KS test** + **AUC** + **KDE-overlap**, with **per-request white-box
ground truth** (delta of `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` around every
single request, so each trial's cache outcome is *measured*, not assumed).
TTFT = `chat.completions` create → first streamed chunk, `max_tokens=1`.

### Text channel — Qwen2.5-3B-Instruct on L4, 30 trials per condition per length

Block-aligned prefixes (verified through the server's own `/tokenize`: **exactly** 16/64/256/1024
tokens), 60 distinct prefixes per length.

| prefix | hit p50 | miss p50 | saved | speedup | KS | AUC | KDE ovl | KS p |
|---|---|---|---|---|---|---|---|---|
| 16 | 49.3 ms | 51.1 ms | 1.8 ms | 1.04× | 0.57 | 0.220 | 0.47 | 8.7e-5 |
| 64 | 50.6 ms | 54.0 ms | 3.4 ms | 1.07× | 0.90 | 0.047 | 0.09 | 5.8e-13 |
| 256 | 52.2 ms | 63.5 ms | 11.2 ms | 1.21× | **1.00** | **0.000** | **0.00** | 1.7e-17 |
| 1024 | 56.2 ms | 151.7 ms | **95.5 ms** | **2.70×** | **1.00** | **0.000** | **0.00** | 1.7e-17 |

From ~64 tokens the distributions are **completely separable**; at 256+ tokens the KS
statistic is exactly 1.0 and the KDE overlap exactly 0.0 (zero false-positive/false-negative
region). At 16 tokens the channel is present but weak (AUC 0.22).

**`cache_salt` closes it.** With a distinct per-tenant salt, the attacker's probe of content
the victim cached is **statistically indistinguishable** from its probe of content nobody
sent — and every observed separation sits *at or below an empirical null floor* measured by
splitting a single condition's own samples in half:

| prefix | OFF KS | ON delta | ON KS | ON KS p | ON AUC | ON KDE ovl | null-floor KS | null-floor ovl |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.57 | −0.9 ms | 0.20 | 0.59 | 0.563 | 0.74 | 0.28 | 0.75 |
| 64 | 0.90 | 0.2 ms | 0.13 | 0.96 | 0.477 | 0.80 | 0.29 | 0.67 |
| 256 | 1.00 | 0.3 ms | 0.20 | 0.59 | 0.449 | 0.86 | 0.28 | 0.73 |
| 1024 | 1.00 | 0.1 ms | 0.20 | 0.59 | 0.466 | 0.68 | 0.30 | 0.66 |

The ground truth says *why*: with salts on, the attacker's probe of the victim's cached
content returns the **same** cached-token count as its probe of never-sent content
(16 tokens = the shared chat-template block) at **every** length — 1024-token case:
`ON same-tenant hit = 1040 cached`, `ON attacker probe = 16`, `ON attacker control = 16`.
There is nothing to detect, so nothing is detected.

### The price of the defense

| prefix | attacker / other tenant loses | *same tenant* |
|---|---|---|
| 16 | 2.4 ms | 0.70 ms |
| 64 | 3.6 ms | 0.41 ms |
| 256 | 11.3 ms | −0.09 ms |
| 1024 | **95.4 ms** | **0.16 ms** |

**Within a tenant the defense is free** (≤0.7 ms, i.e. noise): legitimate prefix reuse is
fully preserved. The entire cost is the *lost cross-tenant / cross-session reuse*, which is
exactly the sharing the attacker was exploiting. This is the cleanest possible defense story:
you pay only for what you were leaking.

### Three methodology traps we hit and fixed (worth a slide — they are the kind of thing that
silently produces a fake positive)

1. **The chat template is a permanently shared block.** Its first 16 tokens are identical for
   every user, so block 0 is always cached after the server's first request. A "miss" on a
   fresh prefix therefore caches **16** tokens, not 0 (measured: miss 16 / hit 80 on a
   93-token prompt; the trailing 13 tokens are a partial block and are never cached). Any
   harness that asserts `cached_tokens == 0` for a miss is wrong. It also means every
   deployment has a small always-on cross-user shared entry.
2. **Ordering artifacts look like leaks.** If the attacker's two probes share a salt and the
   salt is fresh, whichever probe runs *first* pays the template-block miss and the *second*
   inherits it. That produced a spurious `KS=0.433, p=0.007, AUC=0.748` "leak" at 16 tokens
   that vanished once the salt was primed first.
3. **An attacker hitting its own cache looks exactly like a leak.** Repeating the same probe
   under one salt means probe #1 is a miss and #2..#N are self-hits. The first version of the
   image experiment reported a 54 ms "leak" of precisely this kind. The salt must be primed
   and each measured probe taken once.

---

## Exp3 — IMAGE channel, and the headline finding — `RESULTS_exp3_mm.txt`

Qwen2.5-VL-3B-Instruct on A100, 25 trials per condition. Image = a 336×336×3 random uint8 PNG
sent as a base64 `data:` URL.

| defense OFF | TTFA p50 | cached tokens | frac trials with a hit |
|---|---|---|---|
| byte-identical image re-sent | **36.6 ms** | 160 | 1.00 |
| fresh random image each call | **85.0 ms** | 16 | 1.00 |

Detector: Δ = **48.4 ms**, KS = **1.000**, AUC = **0.000**, KDE overlap = **0.000**.

### `cache_salt` does NOT close the image channel

Victim and attacker use different salts; the attacker's salt is primed first and each image is
probed **once**, so neither arm can be the attacker hitting its own cache.

| | TTFA p50 | cached tokens |
|---|---|---|
| victim, same salt (legitimate reuse) | 38.4 ms | 160 |
| **attacker probes the victim's image** | **39.1 ms** | **16** |
| attacker probes a never-sent image | 85.9 ms | **16** |
| detector | **Δ 46.7 ms, KS 1.000, AUC 0.000, overlap 0.000** | |

Both attacker arms are **KV-prefix-cache misses with identical cached-token counts (16)**, i.e.
the KV cache is correctly isolated and contributes the same to both — yet the victim's image is
**46.7 ms faster**. The signal comes from a cache that `cache_salt` does not touch.

### Why — vLLM keeps three caches for one image and only one is salted

| cache | key | salted? |
|---|---|---|
| KV **prefix** cache | block hash with `extra_keys` ⊃ `cache_salt` | **yes** |
| MM **processor** cache (`mm_processor_cache_gb`, default 4 GiB LRU, `vllm/config/multimodal.py:42`) | media hash | **no** |
| MM **encoder** cache (`vllm/v1/core/encoder_cache_manager.py`) | `mm_features[i].identifier` — *"mm_hash or uuid for caching encoder outputs"* (`vllm/multimodal/inputs.py:229`) | **no** |

`grep cache_salt` over `encoder_cache_manager.py` returns **nothing**; the cache maps `mm_hash →
request ids` and is shared across requests by construction. The encoder cache is enabled for any
multimodal model (`encoder_cache_size = max(max_num_batched_tokens, max_tokens_per_mm_item)`,
`scheduler.py:202`), and a hit skips the vision-encoder forward pass entirely.

**This is confirmed by two independent controls** (`modal_exp3_ttft.py::mm_mechanism`,
`RESULTS_exp3_mm_mechanism.txt`). Every arm uses a **fresh salt per probe**, so the engine
reports **0 cached KV tokens in every arm** — the KV prefix cache contributes nothing and cannot
explain any difference:

| control | victim's image | never-sent image | Δ | KS | AUC |
|---|---|---|---|---|---|
| A. default (processor cache ON, 4 GiB) | 40.1 ms | 82.6 ms | **42.5 ms** | 1.000 | 0.000 |
| B. `--mm-processor-cache-gb 0` (processor cache **OFF**) | 50.2 ms | 91.2 ms | **41.1 ms** | 1.000 | 0.000 |

The advantage **survives disabling the processor cache** ⇒ the processor cache is not the
mechanism. It survives with the *same magnitude* (41.1 vs 42.5 ms); both arms simply get ~8 ms
slower across the board, exactly as expected when a preprocessing cache is removed for everyone.

| C. encoder-cache eviction (flood 400 distinct images, then re-probe the same image with a fresh salt) | TTFA |
|---|---|
| before the flood | 38.3 ms |
| after the flood | **78.4 ms** |
| never-sent control | 85.6 ms |
| ⇒ attacker's advantage | **47.3 ms → 7.2 ms** |

Flooding the encoder cache **removes the advantage**, pinning it directly on the content-keyed
`EncoderCacheManager`. (The 7.2 ms residual is ~8% of the effect, consistent with a partially
retained entry rather than a second mechanism.)

## Exp3 — AUDIO channel — `RESULTS_exp3_audio.txt`

Qwen2-Audio-7B-Instruct on A100. Clip = 1.0 s of 16 kHz float32 noise → PCM_16 WAV `data:` URL.

| defense OFF | TTFA p50 | cached tokens |
|---|---|---|
| identical clip bytes re-sent | **38.5 ms** | 48 |
| fresh clip each call | **87.5 ms** | 16 |

Detector: Δ = **49.0 ms**, KS = **1.000**, AUC = **0.000**, KDE overlap = **0.000**.

**The audio channel also survives `cache_salt`** — same design as the image test:

| defense ON | TTFA p50 | cached tokens |
|---|---|---|
| victim, same salt | 38.8 ms | 48 |
| **attacker probes the victim's clip** | **40.4 ms** | **16** |
| attacker probes a never-sent clip | 88.8 ms | **16** |
| detector | **Δ 48.4 ms, KS 1.000, AUC 0.000** | |

Same mechanism: the media is content-keyed in the processor/encoder caches, which carry no salt.
So **both** multimodal channels defeat vLLM's built-in defence.

### Decode determinism (exp2's open question, answered end-to-end)

What matters is whether the **first** send of a re-encoded clip reuses the **original** clip's
cached entry — repeats always hit themselves, so a median over repeats is meaningless.

| variant (first send) | cached tokens | reuses the original entry? |
|---|---|---|
| unseen clip (template baseline) | 16 | — |
| byte-identical WAV | 48 | **yes** |
| soundfile decode round-trip (same PCM path) | 48 | **yes** |
| resampled 16 kHz → 22.05 kHz | 16 | **no** |

⇒ A bit-exact re-send or a lossless decode round-trip collides; **any resample/re-encode breaks
it**. This *bounds the practicality* of the audio channel: it confirms "someone sent exactly this
file", and it will not survive the re-encoding real pipelines perform.

---

## SECOND PASS — rigour, the missing attack, and the real bounds

The first pass left four things unsatisfying. All four are now closed. Everything below is
new measured data (`RESULTS_exp3_text2.txt`, `RESULTS_exp3_text_recovery.txt`,
`RESULTS_exp3_mm_extra.txt`, `RESULTS_exp3_audio_extra.txt`, `RESULTS_exp3_mm_evict.txt`),
and the "channel closed" claims are now **bounds with confidence intervals**, not failed
significance tests (`analyze_rigor.py` → `rigor_summary.json`; raw per-trial latencies are in
the JSONs).

### 1. Text: the channel scales super-linearly, and the defence is bounded

Extended to 2048 and 4096 tokens (60 trials/condition, raw samples kept):

| prefix | hit p50 | miss p50 | saved | miss/hit | KS | AUC |
|---|---|---|---|---|---|---|
| 16 | 49.5 ms | 51.4 ms | 1.7 ms | 1.04× | 0.63 | 0.137 |
| 64 | 49.6 ms | 53.9 ms | 4.0 ms | 1.09× | 1.000 | 0.000 |
| 256 | 51.8 ms | 63.3 ms | 11.3 ms | 1.22× | 1.000 | 0.000 |
| 1024 | 57.9 ms | 152.0 ms | 94.2 ms | 2.63× | 1.000 | 0.000 |
| 2048 | 64.4 ms | 295.8 ms | 232.6 ms | 4.59× | 1.000 | 0.000 |
| **4096** | 74.5 ms | **567.6 ms** | **492.9 ms** | **7.62×** | 1.000 | 0.000 |

The advantage is **super-linear in prefix length** (1.7 → 492.9 ms), and at 4096 tokens a
co-tenant with a clock gets a **7.6× TTFT gap** with zero overlap between the distributions.

**And the defence is now a measured bound, not a null result.** Bootstrap (12k resamples) on
the salt-ON arms, with the ground truth confirming **identical cached KV tokens (16 vs 16)**
in both arms at every length:

| prefix | open effect | salt-ON residual, 95% CI | bound as % of open effect | AUC 95% CI |
|---|---|---|---|---|
| 16 | 1.7 ms | [−0.39, +0.42] ms | 24% (of a 1.7 ms effect) | [0.397, 0.606] |
| 64 | 4.0 ms | [−0.32, +0.38] ms | 9.7% | [0.395, 0.605] |
| 256 | 11.3 ms | [−0.46, +0.85] ms | 7.5% | [0.355, 0.561] |
| 1024 | 94.2 ms | [−0.45, +0.97] ms | **1.0%** | [0.351, 0.559] |
| 2048 | 232.6 ms | [−0.33, +1.83] ms | **0.8%** | [0.342, 0.551] |
| 4096 | 492.9 ms | [−1.03, +1.81] ms | **0.4%** | [0.357, 0.559] |

So the honest statement is: **at 1024–4096 tokens, `cache_salt` bounds any residual text leak to
under 2 ms — at most 1% of the 94–493 ms effect it is defending against** — and at short
prefixes the *absolute* bound is even tighter (0.4 ms) even though the relative bound is weaker
because there is barely any effect to hide.

### 1b. Detector characterisation in the base paper's own style (precision-recall + AP)

Gu et al. report **precision-recall curves and average precision** (their Fig. 4/6 and
Table 1) — cache hits positive, misses negative — and never use ROC/AUC. We now report the
same quantity, so our numbers sit directly beside the reproduction's:

| condition | ROC AUC | **average precision** | best threshold | TPR / FPR at it |
|---|---|---|---|---|
| text ≥64 tokens, defence OFF | 1.000 | **1.000** | e.g. 51.7 ms @64 | 1.00 / 0.00 |
| text 16 tokens, defence OFF | 0.863 | 0.860 | 50.1 ms | 0.77 / 0.13 |
| text, defence ON (all lengths) | 0.498–0.554 | 0.518–0.590 | — | no useful point |
| image, same image / different prompt | 1.000 | **1.000** | 33.0 ms | 1.00 / 0.00 |
| image, one pixel different | 0.618 | 0.655 | — | no reliable signal |

**On the base paper's own metric, a self-hosted vLLM is a far cleaner oracle than any
commercial API they audited: AP = 1.000 vs their 0.71–0.90** (`average precisions mostly lie
around a moderately high value of 0.8`, their §4.2/Table 1). That is the headline comparison
between our half and the reproduction's.

### 2. Text under load — the channel survives real serving conditions

Idle measurements can flatter a side channel. Repeat at 1024 tokens while other clients hammer
the same server:

**Third revision — and this one has K=3 restarts behind it.** Because a single run had already
flipped this conclusion, we re-ran the key claims **K = 3 times, restarting the vLLM server
between repetitions** (fresh engine, fresh cache, fresh CUDA graphs). Prompts are fixed across
reps, so the spread is the *system's*, not the prompts':

| measurement (1024-token prefix) | AUC per rep | AP per rep | Δ p50 per rep |
|---|---|---|---|
| text, idle | 0.000, 0.000, 0.000 | 1.000 ×3 | 91.5, 92.1, 93.4 ms |
| text, **8 concurrent load threads** | 0.014, 0.044, 0.043 | 0.988, 0.965, 0.959 | 53.4, 53.0, 58.8 ms |
| text, salt ON | 0.570, 0.491, 0.514 | 0.511, 0.543, 0.541 | −0.4, 0.1, −0.0 ms |

**Summary (mean ± sd over the 3 reps).** idle: Δ 92.4 ± 1.0 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000. 8 load threads: Δ 55.1 ± 3.3 ms, AUC 0.034 ± 0.017, AP 0.971 ± 0.015. salt ON: Δ -0.1 ± 0.3 ms, AUC 0.525 ± 0.041, AP 0.532 ± 0.018.

**Corrected reading.** Under 8 concurrent load threads the effect size drops by ~40 %
(**92.4 ± 1.0 ms → 55.1 ± 3.3 ms**) but the detector stays strongly effective
(**AP 0.971 ± 0.015, AUC 0.034 ± 0.017**). Pooling every 8-thread measurement in this project —
0.000, 0.212, 0.014, 0.044, 0.043 — **four of five give AUC ≤ 0.044**; the 0.212 came from the
earlier run that swept five load levels sequentially inside one server session, i.e. a less
controlled protocol. So the defensible claim is: **the channel persists under concurrent load
with a roughly 40 % smaller effect size and, occasionally, a destabilised detector — it does not
disappear.** (An intermediate draft of this section claimed the opposite; the K = 3 data settled
it.)

### 2c. Repeatability of the OTHER key claims (same K = 3, server restarted each rep)

| claim | AUC per rep | AP per rep | Δ p50 per rep |
|---|---|---|---|
| **image**, attacker probes the victim's image (salt ON) | 0.000, 0.000, 0.000 | 1.000 ×3 | 65.3, 65.9, 66.6 ms |
| **image**, same image under a *different prompt* | 0.000, 0.000, 0.000 | 1.000 ×3 | 64.9, 66.8, 66.7 ms |
| **audio**, attacker probes the victim's clip (salt ON) | 0.000, 0.062, 0.000 | 1.000, 0.968, 1.000 | 55.6, 56.2, 59.9 ms |
| **audio**, does FLAC reuse the original entry? | — | — | **False in 3/3 reps** |

**Summary (mean ± sd over the 3 reps).** image salt-ON attack: Δ 65.9 ± 0.7 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000. image, same image under a different prompt: Δ 66.1 ± 1.1 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000. audio salt-ON attack: Δ 57.2 ± 2.3 ms, AUC 0.021 ± 0.036, AP 0.989 ± 0.019. FLAC reuses the original entry: **False in 3/3 reps**.

Two things this buys us:

1. **Every headline claim reproduced, and the separability is essentially exact.** The image and
   audio attacks gave AUC ≤ 0.062 and AP ≥ 0.968 in *every* repetition. This is the strongest
   evidence in the document that the findings are not single-run artifacts.
2. **Separability is reproducible; the absolute effect size is only reproducible within a
   protocol.** Within these three reps the image Δ was 65.9 ± 0.7 ms (CV ≈ 1 %), but the *original*
   image run measured 46.7 ms — same design, same model, same GPU type, **41 % apart**. So effect
   sizes must be quoted as ranges (e.g. "≈47–67 ms"), never as a single figure. The AUC/AP
   conclusions are unaffected because they saturate.

### 3. The missing piece: end-to-end recovery from a live server using only TTFT

exp4 was a query-complexity model over an *ideal* oracle. This is the real thing: a live vLLM,
the attacker reads one number per request, no `/metrics`, no privileged view.

Victim prompt: `HEAD [A] LONG1 [B] LONG2 [C] LONG3`, each field one of **V = 12** candidates.
Each field is followed by a long *known* run, so a correct guess extends the matched cached
prefix by that whole run.

| | fields recovered | trials fully correct | median margin | median rank of the true candidate |
|---|---|---|---|---|
| **salt OFF** | **18 / 18 (100%)** | **6 / 6** | **39.3 ms** | **1.0** |
| salt ON (control) | 1 / 18 (5.6%) | 0 / 6 | 0.44 ms | 8.0 (chance 6.5) |

**36 queries per trial against a naive joint search space of 12³ = 1728 (48× fewer)**, and the
secret is recovered exactly, every time. `cache_salt` reduces it to chance.

*The operational catch, which is also a nice consistency check on exp4:* each candidate can be
probed **exactly once**. A second probe of the same candidate hits the attacker's *own* cache
entry, so it carries no information — which is precisely why the query cost is V per block.
Our first version of this experiment reused one template across trials; after trial 0 every
candidate probe was the attacker's own cache hit (all 12 within 3 ms) and it recovered
**0/6**. With fresh template material per trial it recovers 6/6. The failure mode and the
success are the same mechanism.

### 4. Image: the attacker does not even need the victim's prompt

The KV channel requires guessing the *whole prefix*. The encoder cache is keyed on the media
hash alone, so it should not. Victim sends image X with one prompt; the attacker sends the
**same image X with completely different text**, fresh salt, one probe:

| | TTFA p50 | cached KV tokens |
|---|---|---|
| victim's image, attacker's own prompt | **31.2 ms** | **0** |
| never-sent image, attacker's prompt | 61.4 ms | **0** |
| detector | **Δ 30.2 ms, KS 1.000, AUC 0.000** | |

Both arms cache **zero** KV tokens — the text is different, so the KV prefix cache contributes
nothing to either — yet the victim's image is 30 ms faster. **The multimodal channel is
strictly stronger than the text one: the attacker needs only the image, not the prompt.**

### 5. Image is bounded to *exact* content

Change a single pixel and the advantage disappears:

| | TTFA p50 | ROC AUC vs unseen (95% CI) |
|---|---|---|
| victim's image, 1 pixel changed | 60.6 ms | 0.618 [0.404, 0.813] |
| never-sent image | 60.8 ms | — |

Δ = **0.14 ms, 95% CI [−0.63, +1.67]**, and the AUC CI **contains 0.5**: at n = 15 there is no
reliable signal, and any effect is bounded at ~1.7 ms. Contrast this with the byte-identical
resend (48.4 ms, ROC AUC 1.000) and with the same image under a *different prompt* (30.2 ms,
ROC AUC 1.000). So the channel needs **exact** pixels: it confirms *this image*, it does not
match a near-duplicate. (exp2's avalanche claim, now measured end-to-end in TTFA.)

### 6. Both multimodal caches leak — the two are now separated

Flooding the encoder cache removes most of the effect but plateaus: 400 images → residual 22%,
1200 images → residual still 22% (6.4 ms). A residual that does not shrink with a bigger flood
is the signature of a second cache the flood cannot reach: the **processor cache is 4 GiB**
(`mm_processor_cache_gb`), and 1200 images (~0.4 GB) never evict it.

The decisive run (`exp3_mm_evict`, `RESULTS_exp3_mm_evict.txt`) disables the processor cache
(`--mm-processor-cache-gb 0`) and floods only the encoder cache:

| | victim's image | never-sent image | attacker's advantage |
|---|---|---|---|
| before the flood | 49.1 ms | 86.4 ms | **37.4 ms** |
| after the flood (800 distinct images) | 80.8 ms | 81.1 ms | **0.37 ms (1.0% residual)** |

With the processor cache out of the picture, flooding the encoder cache removes **99% of the
advantage**. So the two contributions separate cleanly:

- **encoder cache** (skips the vision-encoder forward pass): the bulk, ~30-37 ms;
- **processor cache** (skips decode / resize / normalise): a smaller, *persistent* ~6 ms that a
  token-budgeted flood cannot evict.

Both are content-keyed and neither is salted. This is the complete mechanism.

### 7. Audio: even a *lossless* container change breaks the key

exp2 predicted the audio key is over the decoded array, so a different container carrying the
same PCM should still collide. **It does not.** None of FLAC, OGG/Vorbis or MP3 reuses the
original clip's entry:

| variant (first send) | cached KV tokens | reuses the original entry? |
|---|---|---|
| never-sent clip (baseline) | 16 | — |
| byte-identical WAV | 48 | **yes** |
| **FLAC — lossless, same PCM** | 16 | **no** |
| OGG/Vorbis (lossy) | 16 | no |
| MP3 (lossy) | 16 | no |

Verified locally with libsndfile 1.2.2: the WAV and FLAC paths decode to arrays that differ by
**3.05e-05** (different PCM quantisation), which is exactly the magnitude exp2 showed is enough
to change the media key. So the honest bound is stronger than predicted: **the audio channel
requires a bit-identical decode path — the attacker must hold the identical file.** It is the
least practical of the three channels.

---

## The corrected headline

The handoff expected "`cache_salt` closes the channel; measure its cost." Measured, that is true
for **text** — the cost is ~0 inside a tenant, any residual leak is bounded under 2 ms (≤1% of
a 94–493 ms effect), and the channel survives concurrent load. But for **image and audio the
defence does not close the channel at all**: vLLM's `cache_salt` salts only the KV prefix cache,
while the expensive multimodal processor and encoder caches are keyed on content alone and stay
shared across tenants. Two controls separate the two caches, and the encoder-cache channel is
strictly stronger than the text one because **the attacker needs only the image, not the
victim's prompt**. That is a concrete, actionable defect with a clean fix (salt those caches, or
make them per-tenant), and it moves our half of the project past "a known text channel"
to "the defence that the base paper proposed but never evaluated is incomplete for
multimodal inputs".

The **end-to-end live recovery** (100% of fields, 36 queries vs 1728) is the piece that turns
exp4's complexity model and exp3's oracle-reliability measurements into a demonstrated attack
rather than two disconnected results.

---

## Proposal skeleton this supports

- **Threat model:** self-hosted / multi-tenant vLLM with cross-request prefix + multimodal
  encoder-cache sharing; attacker = co-tenant with API access + a clock.
- **Contribution:** (1) first characterization of the **image & audio** membership oracle on
  vLLM incl. the exact-content / model-scoped / decode-determinism properties **and the fact
  that `cache_salt` does not close them**; (2) a
  **query-complexity model** (`V·n`) that says precisely which secrets are at risk; (3)
  **measured defense pricing** for `cache_salt` (TTFT/TTFA cost vs channel closure) plus the
  `PYTHONHASHSEED` determinism-inversion caveat.
- **Explicitly NOT claimed:** breaking SHA-256; the already-published text-prefix channel.
- **Feasibility:** CPU analysis done + source-verified; **all three GPU channels measured**, plus the mechanism controls. Everything reproducible via `modal run modal_exp3_ttft.py[::mm|::audio|::mm_mechanism]`; harness regressions covered by `test_harness_offline.py`.
