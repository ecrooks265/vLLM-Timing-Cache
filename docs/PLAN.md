# Cache-Timing Lab — Go-Forward Plan

_As of 2026-10-20. Living version (with the interactive timeline) is the Claude Doc "Cache-Timing Lab — Go-Forward Plan" in the AI Infosec Final Proj project._

## Where we are now

Detection is fully replicated and slightly exceeds the paper; extraction — the paper's stated open problem and our core contribution — is scaffolded but not yet producing results.

| Component | Status | Evidence / notes |
| --- | --- | --- |
| Detection side channel (Exp 01) | Done | Per-user and global both AP = 1.00, KS p ≈ 10⁻⁷²; median hit 16.0 ms vs miss 22.5 ms (Δ 6.5 ms), n = 250 |
| Cross-user / global sharing | Confirmed | Stock vLLM prefix cache is process-global, not partitioned by the OpenAI `user` field — cross-user hits out of the box |
| Ground-truth labeling | Working | Reads vLLM `/metrics` for true hit/miss; separates cache-hit rate from detection rate |
| Salting defense (Exp 04 / 06) | Coded, not yet run | Uses vLLM `cache_salt`; three zones = no-salt (global), shared salt (per-org), distinct salt (per-user). No saved results found yet |
| Prefix extraction (Exp 02 / 09) | Open gap | `recovery: {}` empty; full-vocab = `NotImplementedError`; token-granular needs `block_size=1` (ROCm/WSL2 stuck at 16) |
| Model fidelity | Stand-in | Qwen2.5-0.5B-Instruct, not the paper's Llama-3.1-8B (HF access pending); only `model.name` changes |
| Engineering hygiene | Partial | Not under git; server-side TTFT not wired (client only); first-token text not parsed |

## Did we recreate the paper?

Yes for the detection half of Gu et al. (ICML 2025), with a white-box improvement they could not make; not yet for the extraction half — exactly the gap they flagged.

We match the method point for point: two-sample KS test, average precision as headline metric, NUMSAMPLES = 250 with warmup discarded, prefix caching on, hit = primed prefix vs miss = unique prefix. Where they could only infer the label from timing, we read vLLM's real cache counters, so every AP is measured against ground truth.

Two boxes remain open: (1) model fidelity (Qwen stand-in → Llama-3.1-8B is a one-line swap), and (2) the paper's own limitation — "We were unable to execute practical prompt extraction attacks." Turning "something leaked" into "this is what leaked" is the first real deliverable below.

## The expansion thesis

The field moved in three directions our white-box testbed is well placed to study. The contribution is the ground-truth measurement every black-box paper is missing.

- **Actual extraction** — EarlyBird and InputSnatch recover *what* was cached, never calibrated against real cache state.
- **New channels** — speculative decoding leaks via acceptance rate / inter-token latency; RAG leaks via the non-prefix KV cache. Both reproducible on vLLM, both simulatable.
- **A defense arms race** — selective/threshold KV sharing, PrefixWall, KVGov, provenance schemes — almost none measured against ground-truth leakage.

The wedge: because we serve vLLM ourselves, we can put a number on both sides of every defense — leakage removed (AP/KS vs true labels) and cost paid (hit-rate, TTFT, throughput). Pricing defenses on one calibrated testbed is the novel result.

## Granularity: the block_size=1 problem

**Recommendation:** a calibrated simulator for token-granular recovery, anchored by real block-granular recovery on the 7900 XTX. Skip the GCP / pinned-vLLM path under this deadline.

1. **Real, on-hardware anchor** — run block-granular recovery (Exp 09) at `block_size=16`. Recover one 16-token block at a time; honest measured baseline.
2. **Calibrated simulator** — seed from the TTFT distributions already measured (hit ≈ 16.0 ms, miss ≈ 22.5 ms, with real variance/tail from `requests.jsonl`); run the EarlyBird search against them at `block_size=1` for token-granular recovery, multi-tenant contention, and per-org isolation sweeps the single GPU can't stage.
3. **Validate the simulator** — run it at `block_size=16` and show its recovery curve matches the real Exp 09 run within noise; then the token-granular predictions are credible.

This is what "we can simulate many things" buys: scale and granularity the hardware forbids, with one foot on measured ground.

## The two-week plan

Phases overlap by design: replication + priced three-zone defense finish by day 4, so a complete project exists before extraction and the new channel add upside.

| Phase | Days | Dates | Deliverable |
| --- | --- | --- | --- |
| Phase 0 · Setup | 1 | Oct 20–21 | git init, pin vLLM/harness versions, clean salting run saved |
| Phase 1 · Replicate + price defense | 1–4 | Oct 20–24 | Exp 01 locked; full three-zone salting + padding sweep with real numbers |
| Phase 2 · Extraction (critical path) | 4–9 | Oct 24–29 | real block-granular recovery (Exp 09) + calibrated token-granular simulator, scored vs ground truth |
| Phase 3 · New channel | 8–12 | Oct 28–Nov 1 | speculative-decoding timing channel (ITL vs acceptance) on vLLM |
| Writeup + repro | 13–14 | Nov 1–3 | simulator-vs-hardware validation, figures, reproducibility pass |

## New channel: speculative-decoding timing

Avoids `block_size=1` entirely and is natively supported in vLLM. Acceptance rate (hence inter-token latency) depends on text predictability, leaking info about the prompt and generated tokens.

1. Enable speculative decoding in vLLM (`speculative_config` with a small draft model, or n-gram speculation).
2. Measure per-token inter-token latency (ITL) and record vLLM's reported acceptance rate as the ground-truth label.
3. Show ITL distinguishes high- vs low-acceptance continuations — a timing oracle on *content*, not just cache membership.
4. Simulate the attacker's inference over candidates using the measured acceptance-vs-ITL relationship, sharing one harness with the cache simulator.

**Stretch alternative:** RAG / non-prefix KV-cache attack (heavier setup: retriever + corpus) — backup, not default.

## Defense pricing matrix

One table pricing each defense on both axes, all measured on the same testbed.

| Defense | Leakage (AP / KS) | Cache hit-rate | Median TTFT | Throughput | Build effort |
| --- | --- | --- | --- | --- | --- |
| No defense (global cache) | AP ≈ 1.00 (measured) | highest | 16.0 ms hit | baseline | none |
| Per-org / shared salt | to measure | to measure | to measure | to measure | done (Exp 06) |
| Per-user / per-session salt | to measure (expect AP → chance) | to measure | to measure | to measure | done (Exp 06) |
| Constant-time / TTFT padding | to measure | unchanged | padded, higher | to measure | small |
| Selective / threshold KV sharing | to measure | to measure | to measure | to measure | medium |
| Full per-tenant isolation | expect AP → chance | lowest | highest miss | lowest | config |

Final figure: a scatter of leakage (y) vs performance cost (x), one point per defense — the security-vs-performance frontier no black-box paper can draw.

## Risks, cut-lines, and definition of done

Cut from the bottom up if time runs short:

1. Provider re-audit — already out of scope; do not re-add.
2. Speculative-decoding channel (Phase 3) — first to cut.
3. Selective / threshold KV-sharing implementation — downgrade to "reproduce reported numbers, position on our curve."
4. Token-granular simulator — if the real block-granular run is solid, present the simulator as a validated extension rather than a headline.

**Definition of done (minimal path):** detection replicated with ground truth; three-zone salting priced with real numbers; block-granular extraction demonstrated on hardware; a security-vs-performance frontier figure. Token-granular extraction and the speculative-decoding channel are the upside.

## References

- Gu et al. — Auditing Prompt Caching in Language Model APIs (ICML 2025, arXiv:2502.07776)
- Song et al. — The Early Bird Catches the Leak (arXiv:2409.20002)
- InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks (arXiv:2411.18191)
- When Speculation Spills Secrets: Side Channels via Speculative Decoding in LLMs (arXiv:2411.01076)
- Selective KV-Cache Sharing to Mitigate Timing Side-Channels (arXiv:2508.08438)
- PrefixWall: Mitigating Prefix Caching Side Channels (arXiv:2603.10726)
- Governing the KV Cache / KVGov (arXiv:2608.09225)
- Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG (arXiv:2606.21842)
- Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving (NDSS 2025)
