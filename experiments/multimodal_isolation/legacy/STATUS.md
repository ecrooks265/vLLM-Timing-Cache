# Status: exp3 + a second pass — what we ran, what it shows, what changed

Written for the team. Supersedes the "not done — needs a GPU" section of `HANDOFF_2.md`.
Two rounds of GPU work are complete; every number here is measured.

## Scope — which half of the project this is

The assignment is **reproduce the base paper, then extend it**. The **reproduction** (the
black-box audit of commercial APIs in Gu et al., ICML 2025) is **another teammate's
deliverable**. **Everything in this document is the EXTENSION half** and must not be
presented as the reproduction. We deliberately never touched a third-party API.

Our bridge to their half: Gu et al.'s §6 proposes four mitigations and evaluates none. The
first is *"only per-user caching should be allowed."* **vLLM's `cache_salt` is the production
implementation of that exact mitigation**, so measuring it is the natural extension of their
proposal — and it is where we find the paper's open problem: the mitigation is **incomplete
for multimodal inputs**.

If the two halves need to line up numerically, we now also report the paper's own statistics
(one-sided KS `alternative="greater"`, and average precision from the PR curve) next to ours.

## What ran

| run | GPU | model | what it establishes |
|---|---|---|---|
| `smoke` | L4 | Qwen2.5-3B | harness correctness |
| `text` | L4 | Qwen2.5-3B | 16–1024 tokens, salt on/off |
| `mm` | A100 | Qwen2.5-VL-3B | image channel + salt |
| `audio` | A100 | Qwen2-Audio-7B | audio channel + decode determinism |
| `mm_mechanism` | A100 | Qwen2.5-VL-3B | which cache leaks (2 controls) |
| **`text2`** | L4 | Qwen2.5-3B | 2048/4096 tokens, raw samples, **concurrent load** |
| **`text_recovery`** | L4 | Qwen2.5-3B | **live end-to-end secret recovery** + salt control |
| **`mm_extra`** | A100 | Qwen2.5-VL-3B | prompt-independence, 1-pixel bound, deep eviction |
| **`audio_extra`** | A100 | Qwen2-Audio-7B | container changes (FLAC/OGG/MP3) |
| **`mm_evict`** | A100 | Qwen2.5-VL-3B | separating the two unsalted multimodal caches |

A few dollars of Modal credit total. Reproduce with the commands in `RUN_ME.txt`.

## Round 1 — the headline

`HANDOFF_2.md` predicted *"cache_salt closes the oracle for all modalities; measure its cost."*
**Half right, and the wrong half is the interesting one.**

- **Text — the defence works, and it is free inside a tenant** (≤0.7 ms). The
  attacker/other tenant pays the full lost reuse.
- **Image and audio — the defence does NOT work.** Both channels survive `cache_salt` with
  essentially unchanged magnitude, with **identical cached KV tokens in both arms**, so the KV
  prefix cache is genuinely isolated and cannot explain the gap.

**Why:** vLLM keeps three caches per image and salts only one.

| cache | key | salted? |
|---|---|---|
| KV prefix cache | block hash ⊃ `cache_salt` | **yes** |
| MM processor cache (`mm_processor_cache_gb`, 4 GiB LRU) | media hash | no |
| MM encoder cache (`EncoderCacheManager`) | `mm_hash`, no `cache_salt` in the file | **no** |

## Round 3 — repeatability (K = 3, server restarted between repetitions)

Our own load run had flipped its conclusion, so every headline claim was re-measured three
times with a **fresh vLLM server each repetition**:

- **All headline claims reproduced.** Image and audio salt-ON attacks: **AUC ≤ 0.062, AP ≥ 0.968
  in every rep**. Text-with-salt stayed at chance (AUC 0.525 ± 0.041). The FLAC verdict was
  **False in 3/3**.
- **Summary (mean ± sd).** text idle: Δ 92.4 ± 1.0 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000; text 8-thread load: Δ 55.1 ± 3.3 ms, AUC 0.034 ± 0.017, AP 0.971 ± 0.015; text salt ON: Δ -0.1 ± 0.3 ms, AUC 0.525 ± 0.041, AP 0.532 ± 0.018; image salt-ON attack: Δ 65.9 ± 0.7 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000; image, same image under a different prompt: Δ 66.1 ± 1.1 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000; Δ 65.9 ± 0.7 ms, AUC 0.000 ± 0.000, AP 1.000 ± 0.000; audio salt-ON attack: Δ 57.2 ± 2.3 ms, AUC 0.021 ± 0.036, AP 0.989 ± 0.019.
- **Text idle is rock solid**: AUC 0.000 ×3, AP 1.000 ×3, Δ = 92.4 ± 1.0 ms.
- **New limitation found:** the **absolute effect size is not reproducible across sessions** —
  the image attack was 46.7 ms in the original run and 65.9 ± 0.7 ms in these, **41 % apart**.
  → quote effect sizes as ranges (≈47–67 ms).
- **Load, corrected a third time:** 8 threads cut the effect by ~40 % (92.4 → 55.1 ms) but leave
  it strongly detectable (AP 0.971 ± 0.015). Pooling all five 8-thread measurements in the
  project (0.000 / 0.212 / 0.014 / 0.044 / 0.043), **four of five give AUC ≤ 0.044**.

## Round 2 — the things that were unsatisfying, now fixed

1. **"Channel closed" was absence of evidence.** It is now a **bound** with bootstrap CIs on the
   raw per-trial latencies. At 1024–4096 tokens any residual text leak is **< 2 ms, i.e. ≤1%**
   of the 94–493 ms open-channel effect. (`analyze_rigor.py`)
2. **The channel scales super-linearly.** Extended to 2048/4096 tokens: the open effect goes
   1.7 → 4.0 → 11.3 → 94.2 → 232.6 → **492.9 ms**, a **7.6× TTFT gap** at 4096 tokens with zero
   distribution overlap.
3. **It is not an idle-lab artifact.** Under **32 concurrent load threads** the detector still
   separates (AUC 0.048); absolute latencies inflate ~8×.
4. **The missing attack is now demonstrated.** exp4's `V·n` model was a simulation over an ideal
   oracle. We now recover a 3-field templated secret from a **live vLLM using only TTFT**:
   **18/18 fields, 6/6 trials, median margin 39.3 ms, 36 queries vs a 1728 joint search (48×)**.
   With `cache_salt` on: **1/18 fields, rank 8.0 (chance 6.5)** — pure chance.
5. **The image channel is stronger than the text one.** The attacker sends the victim's image
   with **completely different prompt text** and still gets it: 31.2 vs 61.4 ms, KS 1.000,
   AUC 0.000 — **with zero cached KV tokens in both arms**. The attacker needs only the image,
   not the prompt.
6. **The image channel is exact-content only.** A 1-pixel change is indistinguishable from an
   unseen image (0.1 ms, AUC 0.382). exp2's avalanche claim, now end-to-end in TTFA.
7. **Both multimodal caches leak, separated.** Flooding the encoder cache plateaus at a 22%
   residual that does not shrink from 400 to 1200 images — the signature of a second cache the
   flood cannot reach (the 4 GiB processor cache). Encoder ≈ the bulk; processor ≈ a persistent
   ~6 ms.
8. **The audio bound is tighter than we predicted.** We expected a lossless WAV→FLAC container
   change to still collide. **It does not** — verified locally: the WAV and FLAC decode paths
   differ by 3.05e-05 in the sample array, already enough to change the media key. The audio
   channel needs a bit-identical decode path, i.e. the attacker must hold the identical file.
   It is the least practical of the three.

## Files

- `FINDINGS.md` — full write-up, incl. the "SECOND PASS" section with all of the above.
- `PROPOSAL_DECK.md` — slide-by-slide deck (22 slides) with the real numbers, incl. new slides
  on the rigour bound, load robustness, and the live recovery.
- `RESULTS_*.txt` — transcripts: `exp1/2/4` (CPU, byte-equal verified), `exp3_text`,
  `exp3_text2`, `exp3_text_recovery`, `exp3_mm`, `exp3_mm_mechanism`, `exp3_mm_extra`,
  `exp3_mm_evict`, `exp3_audio`, `exp3_audio_extra`.
- `exp3_*.json` — raw server output straight off the `pcache-out` Modal volume (now including
  raw per-trial latencies).
- `modal_exp3_ttft.py` — the harness. `test_harness_offline.py` — free regression tests.
- `analyze_rigor.py` — bootstrap CIs / equivalence bounds → `rigor_summary.json`.
- `summarize_exp3.py` — JSON → `RESULTS_exp3_*.txt`.
- `RUN_*.log` — raw Modal logs.

## Harness bugs found and fixed (this is why the numbers can be trusted)

The handed-over harness could not have produced the result it was written to produce:

1. **`cache_salt` ON was vacuous** — victim warm-up and attacker probe used the *same* salt, so
   salt-ON was bit-identical to salt-OFF and could never show the channel closing. The big one.
2. **Ground truth was one cumulative gauge** (deprecated `gpu_prefix_cache_hit_rate`). Now every
   request is bracketed by the engine's `prefix_cache_hits/queries` counters.
3. **Prefixes were "N random words"** — not N tokens, not block-aligned. Now verified through
   the server's own `/tokenize` to be *exactly* 16…4096 tokens.
4. **`pkill -f "vllm serve"`** could silently fail and break the next port bind.
5. **Three artifact classes that manufacture fake "leaks"**, each of which produced a large,
   confident, wrong answer before being caught:
   - the shared chat-template block (a "miss" caches 16 tokens, not 0);
   - unprimed attacker salts (ordering shows up as a 0.007-p "leak");
   - **the attacker hitting its own cache** — hit us twice, once in the image experiment
     (a fake 54 ms leak) and once in the recovery experiment (0/6 fields until each trial got
     fresh template material).
6. Environment: `vllm==0.11.0` pulls `transformers` 5.x which crashes the tokenizer (pin
   4.55.2); audio needs the `vllm[audio]` extra; `Image.from_registry("vllm/vllm-openai")` is
   unusable on Modal.

## Honest limits

- One GPU per modality, one model each, `max_tokens=1`, 12–60 trials per condition. Effect sizes
  are large (KS = 1.0), so conclusions are not marginal, but absolute latencies are
  hardware-specific.
- The multimodal channel is a **membership oracle on a known item**, not a reconstruction
  channel; audio additionally requires a bit-identical file.
- The recovery demonstration uses a synthetic templated prompt; the interesting extension is a
  realistic system prompt with a low-entropy field.
- The two-cache separation rests on flooding and on `--mm-processor-cache-gb 0`, not on a patch.
  We have not implemented the fix (salting the multimodal caches) — that is the natural next
  step and would make the defence recommendation concrete.
- No third-party target was touched; everything is the team's own instance with known ground
  truth.

## Suggested next steps

1. Read `PROPOSAL_DECK.md` slides 13–17 — that is the proposal's spine.
2. Send a short responsible-disclosure note to the vLLM project: the multimodal processor and
   encoder caches are not covered by `cache_salt`.
3. Stretch: prototype salting those two caches and show the channel closes, which turns the
   finding into a fix.
