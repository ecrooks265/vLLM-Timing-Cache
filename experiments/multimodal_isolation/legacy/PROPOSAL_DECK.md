# Proposal deck — slide-by-slide (CMU 14-795, due Oct 2)

Reconstructed from the verified facts in `FINDINGS.md` and the **real** exp3 numbers in
`RESULTS_exp3_*.txt`. Everything numeric here is measured; nothing is illustrative.

Numbers are in the team's own words-ready form. Swap in the team's CMU template for the
visuals; the titles/substance are the load-bearing part.

---

## 1. Title

**Does the prefix cache in self-hosted vLLM leak what other tenants asked?**
A measurement study of the text, image, and audio timing channels — and what the built-in
defense (`cache_salt`) actually buys you.

*Speaker note:* one sentence of framing: "multi-tenant LLM serving reuses the KV cache across
users; that reuse is observable from the outside; we measure it on our own vLLM and test
vLLM's own defense."

## 2. The setting (why this exists at all)

- Serving engines (vLLM, SGLang, TensorRT-LLM) keep a **prefix cache**: if two requests share a
  leading run of tokens, the second skips the prefill for that run. That is the main latency win
  in production.
- The cache is **shared across requests by default** — including requests from different users.
- A shared cache that answers faster is a **timing oracle**: "did someone already send this?"

*Figure:* one request's tokens as 16-token blocks; first user fills blocks, second user's
request stops at the first differing block.

## 3. Prior work, and where the gap is

- **Gu et al., "I Know What You Asked", ICML 2025 (arXiv:2502.07776)** — shows the timing
  channel on *commercial* black-box APIs. Audit only: it demonstrates prompt *membership*, it
  does **not** extract prompts.
- **PrefixWall (arXiv:2603.10726)**, **"Governing the KV Cache" (arXiv:2608.09225)** — already
  do text-prefix timing on vLLM.
- **Therefore we do not claim the text channel.** Our delta is (a) the **multimodal** channels,
  (b) a **precise** statement of what is recoverable, (c) the **measured price** of vLLM's own
  defense.

*Speaker note:* stating this up front is what keeps the contribution defensible.

## 4. Scope correction that killed our first idea (put it early — it's a credibility slide)

Our initial pitch (from a teammate's note) was "vLLM's block hash is Python's builtin `hash()`,
so collide it." **That is false in vLLM v0.11.0**, verified in source:

```
vllm/config/cache.py        prefix_caching_hash_algo: Literal["sha256","sha256_cbor"] = "sha256"
vllm/utils/__init__.py      get_hash_fn_by_name() accepts ONLY "sha256"/"sha256_cbor"
```

So: SHA-256 by default, builtin `hash()` no longer selectable, **forged-hit / offline-collision
attacks are infeasible**. Good news for the project: the **timing channel does not depend on
hash strength at all** — a hit leaks because the *content matched*, whatever the hash is. The
hash-strength axis is closed; the caching-behaviour axis is wide open.

## 5. How vLLM actually keys the cache (exp1, source-verified)

- Block = **16 tokens**; a partial trailing block is **never** cached.
- Key of block *k* = `sha256(parent_block_key, that_block's_tokens, extra_keys)`, with
  `extra_keys` = LoRA id / **multimodal hash** / **`cache_salt`**.
- It is a **chain**: change one token in block 1 → blocks 1..n all change; the shared cached
  prefix is exactly the unchanged **leading** blocks.
- Only per-process randomness is `NONE_HASH` (`os.urandom(32)`), used as block 0's parent.
  - ⇒ Within one running server the channel **works out of the box**, no precomputation needed.
  - ⇒ Pinning `PYTHONHASHSEED` "for reproducibility" **removes the only entropy**. Nice
    inversion to flag: a reproducibility best-practice is an anti-defence.
- Cost: sha256 ≈ 1.57 M blocks/s (638 ns/block) vs builtin 13.3 M/s. Hashing is **negligible**
  against prefill FLOPs ⇒ "SHA-256 by default" is free; the real cost of any defence is **lost
  cache reuse**, which is what exp3 prices.

## 6. Threat model

- **We attack our own deployment.** Single vLLM v0.11.0, prefix caching on, all tenants ours,
  ground truth known. No third-party target. This is a *measurement/defence* study.
- **Attacker** = a co-tenant with ordinary API access and a clock (a streaming client can time
  time-to-first-token). No privileged access, no side channels beyond latency.
- **Victim** = another tenant whose request shares a prefix (or an image/audio item) with the
  attacker's guess.

## 7. Research questions

- **RQ1** — Is vLLM's prefix-cache timing channel reliably detectable under real serving noise,
  and how does detectability scale with shared-prefix length?
- **RQ2** — Does the channel exist for **image** and **audio** inputs, and what exactly does a
  hit reveal about media the attacker does not possess?
- **RQ3** — What does vLLM's own defence (`cache_salt`) cost, and **does it actually close each
  channel**?

## 8. exp2 — the multimodal identifier (CPU, byte-equal to real `MultiModalHasher`)

```
media identifier = blake3( model_id, <decoded media>, **processor_kwargs )
```

- Image: over the **decoded PIL image** (pixels), not the post-processor tensor.
- ⇒ **Exact-content membership oracle.** One flipped pixel changes the key (59/64 hex chars
  differ). It **cannot** reconstruct or fuzzy-match an unknown image.
- **Model-scoped**: different `model_id` ⇒ different key ⇒ the oracle only confirms media sent
  to the *same* deployment.
- **Audio has a determinism caveat**: the key is over the *decoded array*, so two recordings of
  "the same" clip collide only if the whole decode path (container→PCM, resample, mix, dtype,
  trim) is byte-identical. exp3 tests this end-to-end.
- Two real override paths: an EXIF `ImageID` UUID replaces the pixel hash, and callers may pass
  explicit `mm_uuids` to skip content hashing entirely.

**Capability statement (say this verbatim):** the multimodal channel is a *membership /
confirmation* oracle on a **known** image or clip ("did someone submit exactly this?"), **not**
a reconstruction channel for unknown media.

## 9. exp4 — what an attacker can actually recover (query complexity)

Given a reliable HIT/MISS oracle, how expensive is recovery?

- Low-entropy/templated secret in a shared prefix (one unknown token per block, V candidates):
  **45.6 / 401 / 4331** oracle queries for V = 10 / 100 / 1000 over 8 blocks — **linear in V and
  in the number of blocks**.
- The chain is why: the oracle confirms each block independently once the prefix before it is
  fixed, collapsing **V^n → V·n**. At V=50, 6 blocks: naive space 1.56e10 vs **199 queries** —
  ~**7.85e7×** cheaper.
- High-entropy content is **safe**: ≥2 unknown high-entropy tokens per block ⇒ ~1e9+ queries per
  block ⇒ infeasible.

**⇒ The channel threatens templated / low-entropy secrets (system-prompt fields, PINs, ids,
document membership), not arbitrary content.** That precision is the contribution.

## 10. exp3 setup — the thing that had never been run

Ran on Modal GPU. **vLLM v0.11.0, block_size = 16**, `--enable-prefix-caching`.

- Text: `Qwen/Qwen2.5-3B-Instruct` on **L4**; TTFT = `chat.completions` → first streamed chunk.
- Image: `Qwen/Qwen2.5-VL-3B-Instruct` on **A100**; Audio: `Qwen2-Audio-7B-Instruct` on **A100**.
- Detectors: two-sample **KS** (the paper's), **AUC**, **KDE overlap** (PrefixWall's).
- **Ground truth per request**: every request is bracketed by reads of the engine's own
  `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`, so each trial records exactly how many
  of *its* tokens were cached. HIT/MISS is **measured, not assumed**.
- Prefixes are built and re-verified through the server's **own `/tokenize`** to be *exactly*
  16/64/256/1024 tokens — genuinely block-aligned.
- **Null floor**: we also split one condition's own samples in half and run the same detector, so
  we know what "no signal" looks like at this sample size.

## 11. RQ1 answer — the text channel is real, and it scales

30 trials per condition per length.

| shared prefix | hit p50 | miss p50 | saved | speedup | KS | AUC | KDE overlap |
|---|---|---|---|---|---|---|---|
| 16 tokens | 49.3 ms | 51.1 ms | 1.8 ms | 1.04× | 0.57 | 0.220 | 0.47 |
| 64 tokens | 50.6 ms | 54.0 ms | 3.4 ms | 1.07× | 0.90 | 0.047 | 0.09 |
| 256 tokens | 52.2 ms | 63.5 ms | 11.2 ms | 1.21× | **1.00** | **0.000** | **0.00** |
| 1024 tokens | 56.2 ms | 151.7 ms | **95.5 ms** | **2.70×** | **1.00** | **0.000** | **0.00** |

From 64 tokens the distributions separate; at 256+ the KS statistic is exactly 1.0 and the KDE
overlap exactly 0.0 — a **zero-error detector** over a **2.7× TTFT gap** at 1k tokens.

*Figure:* overlaid TTFT histograms at 1024 tokens, hit vs miss (two disjoint humps).

## 12. RQ3 answer, text — `cache_salt` closes the KV channel completely

With a distinct per-tenant salt, the attacker's probe of content the victim cached is
statistically **indistinguishable** from its probe of content nobody sent, and every observed
separation sits **at or below the empirical null floor**:

| prefix | OFF KS | ON Δp50 | ON KS | ON p | ON AUC | ON overlap | null-floor KS | null-floor overlap |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.57 | −0.9 ms | 0.20 | 0.59 | 0.563 | 0.74 | 0.28 | 0.75 |
| 64 | 0.90 | 0.2 ms | 0.13 | 0.96 | 0.477 | 0.80 | 0.29 | 0.67 |
| 256 | 1.00 | 0.3 ms | 0.20 | 0.59 | 0.449 | 0.86 | 0.28 | 0.73 |
| 1024 | 1.00 | 0.1 ms | 0.20 | 0.59 | 0.466 | 0.68 | 0.30 | 0.66 |

Ground truth says why: with salts on, the attacker's probe of the victim's cached content
returns the **same** cached-token count as its probe of never-sent content, at every length
(1024-token case: same-tenant hit = 1040 cached; attacker probe = 16; attacker control = 16).
Nothing to detect ⇒ nothing detected.

**And it is free inside a tenant:** the extra TTFT within the same tenant is ≤0.7 ms (noise) at
every length, while the attacker/other tenant pays the full lost reuse (2.4 / 3.6 / 11.3 / **95.4**
ms). *You pay only for the sharing you were leaking.*

## 13. RQ2 answer, image — the channel exists **and it survives the defence**

Repeated image vs fresh image, defense OFF:

| | TTFA p50 | cached tokens | KS | AUC | overlap |
|---|---|---|---|---|---|
| byte-identical image re-sent | **36.6 ms** | 160 | **1.000** | 0.000 | 0.000 |
| fresh random image each call | 85.0 ms | 16 | | | |

**48.4 ms** of TTFA, perfectly separable. Now turn the defence on — victim uses its own salt,
attacker a different one, attacker's salt primed, **one probe per image** so the attacker
cannot hit its own cache:

| | TTFA p50 | cached tokens |
|---|---|---|
| victim, same salt (legit reuse) | 38.4 ms | 160 |
| **attacker probes the victim's image** | **39.1 ms** | **16** |
| attacker probes a never-sent image | 85.9 ms | **16** |
| detector | **Δ 46.7 ms, KS 1.000, AUC 0.000** | |

Both attacker arms are **KV-cache misses with identical cached-token counts (16)**, yet the
victim's image is **46.7 ms faster**. The KV prefix cache is properly isolated — the signal is
coming from **somewhere else entirely.** (Mechanism on the next slide.)

*Speaker note:* this is the headline. `cache_salt` is the defence vLLM's own docstring advertises
as "prevent an attacker to guess prompts in multi-user environments" — and it does close the text
channel. It does **not** close the image channel.

## 14. Why — vLLM keeps three caches for one image, and only one is salted

| cache | key | salted? |
|---|---|---|
| KV **prefix** cache | block hash incl. `extra_keys` ⊃ `cache_salt` | **yes** |
| MM **processor** cache (`mm_processor_cache_gb`, default 4 GiB LRU) | media hash | **no** |
| MM **encoder** cache (`EncoderCacheManager`) | `mm_features[i].identifier` = "mm_hash or uuid for caching encoder outputs" | **no** |

`grep cache_salt` over `vllm/v1/core/encoder_cache_manager.py` returns **nothing**. The encoder
cache is keyed purely on content and shared across requests, so a co-tenant that sends the same
image gets the expensive vision-encoder forward pass for free.

**Two controls pin it down** (this is the difference between "we infer" and "we showed").
Every arm uses a **fresh salt per probe**, so the engine reports **0 cached KV tokens in every
arm** — the KV prefix cache cannot explain any difference:

| control | victim's image | never-sent image | Δ | KS | AUC |
|---|---|---|---|---|---|
| A. default (processor cache ON, 4 GiB) | 40.1 ms | 82.6 ms | **42.5 ms** | 1.000 | 0.000 |
| B. `--mm-processor-cache-gb 0` (processor cache **OFF**) | 50.2 ms | 91.2 ms | **41.1 ms** | 1.000 | 0.000 |

- Disabling the processor cache leaves the advantage **unchanged in magnitude** (41.1 vs 42.5 ms;
  both arms just get ~8 ms slower) ⇒ the processor cache is **not** the mechanism.
- **Flooding** the encoder cache and re-probing the same image with a fresh salt: the attacker's
  advantage collapses (400 images: 47.3 → 7.2 ms; 1200 images: 29.1 → 6.4 ms) but **plateaus at a
  ~22% residual that does not shrink as the flood grows** — the signature of a *second* cache the
  flood cannot reach, because the processor cache is **4 GiB** and 1200 images (~0.4 GB) never
  evict it.
- **The decisive run:** with the processor cache disabled (`--mm-processor-cache-gb 0`) and the
  encoder cache flooded, the advantage goes **37.4 ms → 0.37 ms (1% residual)**. So the two
  caches' contributions separate cleanly: **encoder ≈ 30–37 ms** (skips the vision-encoder
  forward pass), **processor ≈ a persistent ~6 ms** (skips decode/resize/normalise).

**Two more bounds on the image channel:**

- **The attacker does not need the victim's prompt.** Victim sends image X with one prompt; the
  attacker sends the *same image X with completely different text*: **31.2 ms vs 61.4 ms**
  (Δ 30.2 ms, KS 1.000, AUC 0.000) — and both arms cache **zero** KV tokens, so the KV cache
  contributes nothing to either. The encoder cache is keyed on the media hash alone, which makes
  the multimodal channel **strictly stronger than the text one**.
- **Exact content only.** Change one pixel and the advantage vanishes (60.6 vs 60.8 ms,
  AUC 0.382). The channel confirms *this exact image*; it cannot match a near-duplicate.

**⇒ Defence implication:** `cache_salt` must be extended to key the multimodal processor and
encoder caches too (or those caches must be made per-tenant / opt-in). This is a concrete,
actionable vLLM finding — and it is the strongest novelty claim in the project.

## 15. Is the defence *closed*, or did we just fail to detect anything?

A KS p-value that fails to reach significance is **absence of evidence**, and a reviewer will
say so. So we bootstrap the raw per-trial latencies (12k resamples) and report a **bound**.
Ground truth confirms **identical cached KV tokens (16 vs 16)** in both arms at every length:

| prefix | open effect (salt off) | salt-ON residual, 95% CI | bound as % of the open effect | AUC 95% CI |
|---|---|---|---|---|
| 16 | 1.7 ms | [−0.39, +0.42] ms | 24% (of a 1.7 ms effect) | [0.397, 0.606] |
| 64 | 4.0 ms | [−0.32, +0.38] ms | 9.7% | [0.395, 0.605] |
| 256 | 11.3 ms | [−0.46, +0.85] ms | 7.5% | [0.355, 0.561] |
| 1024 | 94.2 ms | [−0.45, +0.97] ms | **1.0%** | [0.351, 0.559] |
| 2048 | 232.6 ms | [−0.33, +1.83] ms | **0.8%** | [0.342, 0.551] |
| **4096** | **492.9 ms** | [−1.03, +1.81] ms | **0.4%** | [0.357, 0.559] |

**The claim is now:** at 1024–4096 tokens, `cache_salt` bounds any residual text leak to
**under 2 ms — at most 1% of the 94–493 ms effect it defends against.**

*Bonus scaling result:* the open channel is **super-linear in prefix length** — 1.7 → 4.0 →
11.3 → 94.2 → 232.6 → **492.9 ms** at 16 → 4096 tokens, i.e. a **7.6× TTFT gap** with zero
distribution overlap at 4096.

## 16. The channel survives real serving conditions (not a quiet-lab artifact)

1024-token prefix, measured while other clients hammer the same server:

A single run of this had already flipped its own conclusion, so we re-ran it **K = 3 times,
restarting the vLLM server between repetitions** (fresh engine, cache, CUDA graphs):

| measurement (1024-token prefix) | AUC per rep | AP per rep | Δ p50 per rep |
|---|---|---|---|
| idle | 0.000, 0.000, 0.000 | 1.000 ×3 | 91.5, 92.1, 93.4 ms |
| **8 concurrent load threads** | 0.014, 0.044, 0.043 | 0.988, 0.965, 0.959 | 53.4, 53.0, 58.8 ms |
| 32 threads (single earlier run) | 0.048 | — | +103.2 ms |

**Summary (mean ± sd).** idle: Δ 92.4 ± 1.0 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000. 8 threads: Δ 55.1 ± 3.3 ms, AUC 0.034 ± 0.017, AP 0.971 ± 0.015. Text salt ON: Δ -0.1 ± 0.3 ms, AUC 0.525 ± 0.041, AP 0.532 ± 0.018.

Load cuts the effect size by ~40 % (**92.4 ± 1.0 → 55.1 ± 3.3 ms**) and leaves it **strongly
detectable** (AP 0.971 ± 0.015). Pooling all five 8-thread measurements in the project —
0.000, 0.212, 0.014, 0.044, 0.043 — **four of five give AUC ≤ 0.044**. **The channel persists
under concurrent load; it does not disappear.**

## 16b. Repeatability — because one of our own runs flipped its conclusion

Every headline claim was re-measured **K = 3 times with the vLLM server restarted between
repetitions** (fresh engine, cache, CUDA graphs). Prompts fixed across reps.

| claim | AUC per rep | AP per rep | Δ p50 per rep |
|---|---|---|---|
| text idle | 0.000 ×3 | 1.000 ×3 | 91.5, 92.1, 93.4 ms |
| text, 8 load threads | 0.014, 0.044, 0.043 | 0.988, 0.965, 0.959 | 53.4, 53.0, 58.8 ms |
| text, salt ON | 0.570, 0.491, 0.514 | 0.511, 0.543, 0.541 | −0.4, 0.1, −0.0 ms |
| **image salt-ON attack** | **0.000 ×3** | **1.000 ×3** | 65.3, 65.9, 66.6 ms |
| image, same image / different prompt | 0.000 ×3 | 1.000 ×3 | 64.9, 66.8, 66.7 ms |
| **audio salt-ON attack** | 0.000, 0.062, 0.000 | 1.000, 0.968, 1.000 | 55.6, 56.2, 59.9 ms |
| audio: does FLAC reuse the original entry? | — | — | **False in 3/3** |

**Summary (mean ± sd over the 3 reps).** image salt-ON attack: Δ 65.9 ± 0.7 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000. image, same image under a different prompt: Δ 66.1 ± 1.1 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000. audio salt-ON attack: Δ 57.2 ± 2.3 ms, AUC 0.021 ± 0.036, AP 0.989 ± 0.019. FLAC reuses the original entry: **False in 3/3 reps**.

**What this buys:** every headline claim reproduced; the separability is essentially exact.

**What it costs (say this out loud):** the *absolute effect size* is stable within a protocol
(CV ≈ 1 %) but **not across sessions** — the image attack measured 46.7 ms originally and
65.9 ± 0.7 ms here, **41 % apart**. So we quote ranges, not point values.

## 17. The missing piece — recovering a secret from a live server using only TTFT

exp4 was a query-complexity model over an *ideal* oracle, and exp3 showed the real oracle is
reliable. Nobody had connected them. Now we do: live vLLM, the attacker reads one number per
request, no `/metrics`, no privileged view.

Victim prompt `HEAD [A] LONG1 [B] LONG2 [C] LONG3`; each field is one of **V = 12** candidates
and is followed by a long *known* run, so a correct guess extends the matched cached prefix by
that whole run.

| | fields recovered | trials fully correct | median margin | median rank of the true candidate |
|---|---|---|---|---|
| **salt OFF** | **18 / 18 (100%)** | **6 / 6** | **39.3 ms** | **1.0** |
| salt ON (control) | 1 / 18 (5.6%) | 0 / 6 | 0.44 ms | 8.0 (chance 6.5) |

**36 queries per trial instead of a 12³ = 1728 joint search (48× fewer), secret recovered
exactly every time.** `cache_salt` reduces it to chance.

*The catch, which validates exp4's model:* each candidate is probeable **exactly once** — a
second probe hits the attacker's *own* cache entry. That is precisely why the cost is V per
block. Our first version reused one template across trials and recovered **0/6** for exactly
this reason; with fresh templates it recovers 6/6.

## 18. RQ2 answer, audio — same channel, plus the determinism bound

Identical clip vs fresh clip, defense OFF: **38.5 ms** hit vs **87.5 ms** miss (**49.0 ms**,
KS 1.000, AUC 0.000).

**And, like the image channel, it survives the defence** — same design (victim's own salt,
attacker's salt primed, one probe per clip, both arms KV-cache misses):

| defense ON | TTFA p50 | cached tokens |
|---|---|---|
| victim, same salt (legit reuse) | 38.8 ms | 48 |
| **attacker probes the victim's clip** | **40.4 ms** | **16** |
| attacker probes a never-sent clip | 88.8 ms | **16** |
| detector | **Δ 48.4 ms, KS 1.000, AUC 0.000, overlap 0.000** | |

**Decode determinism (exp2's open question, answered):** what matters is whether the *first*
send of a re-encoded clip reuses the *original* clip's cached entry — repeats always hit
themselves, so a median over repeats is meaningless.

| variant (first send) | cached KV tokens | reuses the original entry? |
|---|---|---|
| unseen clip (template baseline) | 16 | — |
| byte-identical WAV | 48 | **yes** |
| soundfile decode round-trip (same PCM path) | 48 | **yes** |
| **FLAC — lossless, same PCM** | 16 | **no** |
| OGG/Vorbis (lossy) | 16 | no |
| MP3 (lossy) | 16 | no |

We expected the lossless container change to *still* collide, since the key is over the decoded
array. **It does not** — and verifying locally with libsndfile 1.2.2 explains why: the WAV and
FLAC paths decode to arrays that differ by **3.05e-05** (different PCM quantisation), already
enough to change the media key. So the honest bound is *stronger* than predicted: **the audio
channel needs a bit-identical decode path — the attacker must hold the identical file.** It is
the least practical of the three channels.

## 19. Three methodology traps that manufacture fake positives

Our own bugs, caught only because ground truth is per-request. Worth a slide — this is the kind
of error that would otherwise become a wrong result.

1. **The chat template is a permanently shared 16-token block.** A "miss" on a fresh prefix
   caches **16** tokens, not 0 (measured: miss 16 / hit 80 on a 93-token prompt; the trailing 13
   tokens are a partial block). Any harness asserting `cached_tokens == 0` for a miss is wrong —
   and every deployment has a small always-on cross-user shared entry.
2. **Ordering artifacts look like leaks.** If the attacker's two probes share a fresh salt, the
   first pays the template-block miss and the second inherits it: a spurious `KS=0.433, p=0.007,
   AUC=0.748` "leak" at 16 tokens, which vanished once the salt was primed.
3. **An attacker hitting its own cache looks exactly like a leak.** Repeating one probe under one
   salt is a miss once and self-hits thereafter. Our first image experiment reported a 54 ms
   "leak" of exactly this kind.

⇒ *Report the null floor, prime every salt, take each measured probe once.* This is a
methodological contribution in its own right for anyone measuring cache side channels.

## 20. Prior art / delta recap

Division of labour on the team: **the reproduction of Gu et al. is a separate teammate's
deliverable.** Our half is the *extension* on top of it. Be precise about this in the talk —
claiming their audit would be double-claiming.

| | text channel | image/audio | recovery-cost model | defence cost |
|---|---|---|---|---|
| Gu et al. ICML'25 | yes (black-box audit) | no | no | no (proposed only) |
| PrefixWall / Governing the KV Cache | yes (vLLM) | no | partial | partial |
| **reproduction (teammate)** | **the audit itself** | — | — | — |
| **our extension** | quantifies it on vLLM, bounds the defence | **yes — and the defence fails** | **yes (V·n), demonstrated live** | **yes, measured, and shown incomplete** |

**The one-line bridge between the two halves — use this on the slide:** Gu et al.'s §6
proposes four mitigations and **evaluates none of them**. The first is *"only per-user caching
should be allowed."* vLLM's `cache_salt` is the production implementation of exactly that
mitigation — its own source describes it as salting the prefix cache *"to prevent an attacker
to guess prompts in multi-user environments."* **We are the first to measure whether that
mitigation works and what it costs** — and it works for text but **not** for image or audio.

## 21. Feasibility & timeline (the "can you finish it" slide)

- **Done:** source verification vs real v0.11.0 (`ALL BYTE-EQUAL`); exp1/2/4 on CPU; **exp3 on
  GPU for all three modalities, with white-box ground truth**; all harness bugs found and fixed
  with a regression test (`test_harness_offline.py`).
- **To do:** the mechanism controls are running; write-up; extended lengths / larger models as a
  stretch; responsible-disclosure note to the vLLM project about the unsalted multimodal caches.
- **Cost so far:** a few dollars of Modal credit; every number is reproducible with
  `modal run modal_exp3_ttft.py[::mm|::audio|::mm_mechanism]`.

## 22. Threats to validity (put this slide up before anyone asks)

We rank these by **whether they threaten a conclusion**, not by how comfortable they are.

**The three that matter**

1. **The mechanism is INFERRED, not directly observed.** vLLM 0.11.0 exposes **no** metric for
   the multimodal processor/encoder caches (`vllm/v1/metrics/loggers.py` has none). So "the leak
   is the encoder cache" rests on **elimination plus two perturbation controls**, not on watching
   the cache. Worse, `mm_processor_cache_gb` is *"duplicated for each API process and engine core
   process"* (`config/multimodal.py:46`), so the real topology is messier than our "two boxes".
   → *Fix:* patch `EncoderCacheManager.check_and_update_cache` to count hits. The only way to
   turn inference into observation.
2. ~~**Every result in this deck was measured ONCE.**~~ ✅ **CLOSED — K = 3 with a server
   restart each.** Every headline claim reproduced: image and audio attacks gave **AUC ≤ 0.062
   and AP ≥ 0.968 in every repetition**, and text-with-salt stayed at chance
   (AUC 0.525 ± 0.041). **But it surfaced a new limitation:** separability is reproducible while
   the **absolute effect size is not across sessions** — the image attack was 46.7 ms in the
   original run and 65.9 ± 0.7 ms here, **41 % apart**. → *Consequence:* quote effect sizes as
   ranges (≈47–67 ms), never as a single number.
3. **The recovery demo's prompt was built to amplify the signal.** `HEAD [A] 1024-known [B]
   512-known [C] 256-known` — those long known runs are ours, not something a real system prompt
   guarantees. It shows the protocol *can* work; it does not show it works on a realistic prompt.
   → *Fix:* repeat on a real JSON customer-service template.

**The rest, briefly**

| threat | why it matters | fix |
|---|---|---|
| synthetic random-noise images | not the kind of image an attacker can pre-possess | repeat with real images |
| FLAC explanation is local, not in-container | we *infer* the decode path differs from a local libsndfile check | decode inside the container and diff |
| only PrefixFraction = 1 tested | the base paper's levels 2–4 use **0.95** | sweep the fraction (code written) |
| client-side timing, localhost | real cross-datacentre RTT is the same order as the signal | server-side timing header |
| one model per modality, block_size 16 | encoder-cache behaviour may be model-specific | second VLM |
| no multiple-comparison correction | we computed dozens of KS/AUC/AP | Bonferroni / BH-FDR |
| mechanism runs used a different config | `--max-num-batched-tokens 2048` vs the default | re-run matched |

**What we therefore do NOT claim**

- ❌ "the encoder cache is the *only* cause" — we only eliminated the alternatives, and the
  processor cache still accounts for ~6 ms;
- ❌ "the attack works over a real network" — untested, and our load results suggest it may not;
- ❌ "we reproduced the base paper" — that is the teammate's half.

**One sentence if you only get one:**

> Our strongest claim — *the defence does not close the multimodal channel, 47 ms, zero overlap*
> — rests on white-box ground truth with identical cached-token counts in both arms, so it is
> robust; the **mechanism** and the **generality** are the parts that still need work, and we
> say so.

## 23. Explicit non-claims (say them before a reviewer does)

- We do **not** break SHA-256, and we do **not** claim a weak hash.
- We do **not** claim the text-prefix channel (published).
- We do **not** claim reconstruction of unknown images or audio — only membership of a **known**
  item, or block-by-block recovery of **low-entropy/templated** text.
- We do **not** attack any third party; everything is our own instance with known ground truth.
