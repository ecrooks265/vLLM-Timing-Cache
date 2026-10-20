# Research context

## The paper
Gu, Li, Kuditipudi, Liang, Hashimoto. *Auditing Prompt Caching in Language
Model APIs.* ICML 2025. arXiv:2502.07776. Code (Apache-2.0):
github.com/chenchenygu/auditing-prompt-caching

**Core idea.** LLM APIs cache prompt *prefixes* to cut time-to-first-token
(TTFT). When that cache is shared across users, TTFT becomes a side channel:
send a prefix P, measure TTFT; a fast response means P was already cached —
i.e. *someone else sent it*. Formalized as a hypothesis test over hit-vs-miss
TTFT distributions.

**Their findings.** Global cross-user sharing at 7 of 17 audited providers
(OpenAI among them). Timing alone reveals whether a prefix was seen. The
channel also leaked architecture (evidence OpenAI's embedding model is
decoder-only).

**Their stated limitation (our opening).**
> "We were unable to execute practical prompt extraction attacks."

They prove a prefix *was* cached; they never *recover* it. Black-box only.

## What we add (the 14-795 project)
1. **Build the attack they didn't** — token/block-by-block prefix recovery
   against a shared cache (EarlyBird-style, Song et al. arXiv:2409.20002):
   turn "something leaked" into "this is what leaked."
2. **Ground truth via white-box** — serve the same workload locally on vLLM
   with prefix caching on, read vLLM's real cache metrics for true hit/miss
   labels, and calibrate the timing attack against them. This separates *how
   often the cache hit* from *how often timing detected the hit* — something a
   black-box audit cannot do.
3. **Price each defense** — measure the security-vs-TTFT/throughput curve for
   cache partitioning, per-user isolation, constant-time responses.
4. **Re-audit providers** — the published numbers are from Sep–Oct 2024;
   re-running the auditor is cheap and any drift is itself a finding.

## Why white-box / why vLLM
Black-box gives you timing but never the label. Running vLLM ourselves
(`--enable-prefix-caching`, `/metrics` exposed) gives ground-truth cache state,
so every precision number is anchored to reality rather than inferred from
timing. The local testbed calibrates the attack; the calibrated attack is then
carried back to APIs we're authorized to test.

## Experiment map
| Exp | File | What it does | Block-size sensitive? |
|-----|------|--------------|-----------------------|
| 01 | experiments/01_prove_sidechannel.py | Prove hit/miss TTFT separation (KS test) | No |
| 02 | experiments/02_recover_prefix.py | Token/block-by-block prefix recovery | Yes (see below) |
| 03 | experiments/03_calibration_sweep.py | Calibrate timing threshold vs ground truth | No |
| 04 | experiments/04_defense_salting.py | Cost of prefix salting defense | No |
| 05 | experiments/05_suffix_collision.py | Same-prefix/different-suffix behavior | No |
| 06 | experiments/06_salt_granularity.py | Salt granularity vs leakage | No |
| 09 | experiments/09_block_recovery.py | Block-granular recovery (ROCm fallback) | Yes |

## The block_size caveat (hardware-driven)
Token-by-token recovery (Exp 02) wants `block_size=1` so each cached token is
independently observable. On gfx1100 ROCm that's not guaranteed: the modern
vLLM V1 engine dropped `block_size=1`, and the ROCm paged-attention path wants
block size 16. So the ROCm profile defaults to 16 and recovery is
*block-granular* (Exp 09). Exp 01 and the audit are unaffected — they work at
any block size. True token-granular recovery is the one case for the pinned
vllm==0.6.3.post1 path (config/default.yaml), a much harder gfx1100 build.

## Scope / ethics
Run only against our own vLLM instance or external APIs we're authorized to
test. External-API path is gated behind an explicit `--allow-external` flag and
wired to no provider by default.
