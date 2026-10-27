# Morning brief — vLLM prefix-cache study

## The one thing you must know
The pitch has to change. **vLLM v0.11.0 already uses SHA-256 for the block hash by default**
(builtin `hash()` was removed as an option). I verified this in the real source. So:

- The "vLLM's hash can be hacked" angle your teammate mentioned was true for **old** vLLM
  only. **Do not** build the proposal on a weak hash — it's wrong for current vLLM and a
  reviewer will catch it.
- Good news: the **timing side channel doesn't need a weak hash.** A cache HIT leaks shared
  content regardless of hash strength. The novelty just moves to where it's actually
  defensible.

## What I did overnight (all real, all CPU, nothing fabricated)
1. **Cloned real vLLM v0.11.0** and verified our hash re-implementations **byte-for-byte**
   against the actual source (`verify_against_source.py` → "ALL BYTE-EQUAL"). Both the block
   hash and the multimodal `MultiModalHasher` match exactly.
2. **exp1 (corrected):** block-hash mechanism, hash-chain blast radius, the
   `NONE_HASH`/`PYTHONHASHSEED` determinism story (incl. a nice "pinning the seed *removes*
   the only entropy" point), real hash-cost microbench.
3. **exp2 (corrected):** the multimodal id is `blake3(model_id, raw-decoded-media, kwargs)` —
   not what the first draft assumed. Exact-content membership oracle; auto cross-model
   isolation; audio hits only if the decode path is byte-identical; EXIF-UUID override path.
4. **exp4 (new):** reconstruction query complexity. Block-chaining makes recovery **linear**
   (`V·n`), ~**78 million×** cheaper than naive — but only for **low-entropy/templated**
   secrets; high-entropy content stays safe. Clean, quotable numbers.
5. **exp3 (GPU harness, ready):** now also has an **audio** function and correct salt-based
   framing. I could not run it — no GPU here, and I can't log into or bill your Modal account.

## Your move (5 min each)
```
modal run modal_exp3_ttft.py            # text TTFT hit/miss + KS/KDE + salt cost
modal run modal_exp3_ttft.py::mm        # image TTFA hit/miss
modal run modal_exp3_ttft.py::audio     # audio TTFA + re-encode test
```
Paste the JSON back and I'll drop the real timings into the proposal deck.

## Files (in outputs/pcache/)
- `FINDINGS.md` — full write-up with the corrected novelty story + proposal skeleton
- `exp1_hash_analysis.py`, `exp2_multimodal_ident.py`, `exp4_reconstruction_sim.py` — runnable, real output
- `verify_against_source.py` — proves byte-equality vs real vLLM v0.11.0
- `modal_exp3_ttft.py` — GPU harness (text / image / audio)
- `RESULTS_*.txt` — captured real output of every CPU run

## Honest limits
No GPU timing was run and no timing numbers were invented. I can't provision GPUs or touch
your Modal billing. The reconstruction "queries" are exact-oracle counts (information cost),
separate from — and complementary to — the timing-reliability numbers exp3 will produce.
