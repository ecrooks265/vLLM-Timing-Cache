# Project handoff — vLLM prefix-cache side-channel study

This is a complete brief for another assistant to pick up the work. Everything referenced
lives in `~/Documents/pcache/` on the user's Mac. Read this top to bottom first.

---

## 1. What the project is

A course project for CMU **14-795 (AI Applications in Information Security)**. It extends a
published paper from commercial APIs to **self-hosted vLLM**:

- **Base paper:** Gu et al., *"I Know What You Asked"*, ICML 2025 (arXiv:2502.07776). It shows
  that LLM serving systems reuse the KV cache across users, a cache hit answers faster, and that
  timing difference is observable from the outside. Their audit was black-box and did **not**
  extract prompts.
- **This project's angle:** measure the same timing channel on vLLM, and cover the three things
  the paper and later vLLM work did not: (a) the **image and audio** channels, (b) a precise
  statement of **what an attacker can actually recover**, (c) the **latency cost of vLLM's own
  defense** (`cache_salt`).
- **Framing is defensive/measurement:** all experiments run on the team's **own** vLLM instance,
  where they control every tenant and know the ground-truth cache state. No third-party or
  production target. Keep this framing.

Deliverable due **Oct 2, 2026**: a Project Proposal (slides + talk). The team wrote an earlier
"Idea Portfolio" deck; this is idea #3 taken forward.

---

## 2. Status: what is done vs. not

**Done (on CPU, in this folder, with real captured output):**

- `exp1_hash_analysis.py` — how vLLM keys the block cache; hash-chain behavior; determinism;
  cost microbenchmark. Output in `RESULTS_exp1.txt`.
- `exp2_multimodal_ident.py` — how image/audio get their cache identifier. Output in
  `RESULTS_exp2.txt`.
- `exp4_reconstruction_sim.py` — query-count simulation of block-by-block recovery. Output in
  `RESULTS_exp4.txt`.
- `verify_against_source.py` — imports the **real vLLM v0.11.0** hashing code and confirms the
  re-implementations above are **byte-identical** to it. Output: "ALL BYTE-EQUAL".
- `FINDINGS.md` — the full write-up with the corrected story (read this second, after this file).

> Note: the `RESULTS_*.txt` files were produced in the cloud session. If they are not in the
> folder, just re-run the three scripts — they need only `python3`, `numpy`, `scipy`, `blake3`,
> `cbor2`, `pillow` and print their results.

**Not done — needs a GPU, which is the whole ask for the next assistant:**

- `modal_exp3_ttft.py` — the actual timing measurement on a GPU. This is the one piece that
  produces the headline numbers (hit vs. miss latency, and the cost of the defense). It has
  never been run. See section 4.

---

## 3. Key facts the next assistant must not get wrong

These were verified against the vLLM v0.11.0 source and overturn the first version of the plan:

1. **The default block hash is SHA-256, not Python's `hash()`.**
   `vllm/config/cache.py` → `prefix_caching_hash_algo` defaults to `"sha256"`; the built-in hash
   is no longer selectable. **Do not pitch "the hash is weak / hackable"** — that was only true
   of old vLLM. The timing channel does not depend on hash strength: a hit leaks because the
   content matched, whatever the hash is.

2. **The side channel is a confirmation/membership oracle, not a general read-out.**
   - Text: can confirm a known prefix was sent; can also **recover low-entropy / templated**
     secrets block-by-block (cost is linear, not exponential — see `exp4`). High-entropy text is
     safe.
   - Image / audio: **confirmation only.** The media key is a hash of the whole decoded input
     (`vllm/multimodal/hasher.py`, blake3 over decoded pixels/samples + model id + processor
     args). One bit changes the key, so there is no partial-match search → unknown media cannot
     be reconstructed, only matched against something you already have (or a small candidate set).
   - Audio has an extra caveat: two "same" clips only collide if the full decode is bit-identical;
     a re-encode or resample generally misses. `modal_exp3_ttft.py::audio` is meant to test this.

3. **The defense is `cache_salt`** (a per-tenant/per-session string added to the first block's
   key). With distinct salts, nothing is shared and the oracle closes for all modalities. Its
   cost (lost cache reuse → higher TTFT/TTFA) is exactly what `exp3` is supposed to measure.

4. **Prior art on the text channel already exists** — PrefixWall (arXiv:2603.10726) and
   "Governing the KV Cache" (arXiv:2608.09225). So the novelty must sit on **multimodal +
   defense cost + the recovery-cost model**, not on the text channel alone. Cite these.

---

## 4. The one job to finish: run exp3 on a GPU

`modal_exp3_ttft.py` uses [Modal](https://modal.com) (serverless GPU). The team has a Modal
account. The cloud assistant could not run it because Modal's servers are blocked by the
session's network policy — it must run from a real terminal with normal internet (the user's Mac,
or any machine with a GPU and internet).

**Commands (also in `RUN_ME.txt`):**
```
cd ~/Documents/pcache
pip3 install modal          # once
modal token new             # once — browser login to the team's Modal account
modal run modal_exp3_ttft.py            # text channel (cheap GPU)
modal run modal_exp3_ttft.py::mm        # image channel
modal run modal_exp3_ttft.py::audio     # audio channel (set AUDIO_MODEL if Qwen2-Audio isn't available)
```

**What it outputs:** each function prints a JSON blob and also writes it to a Modal volume
`pcache-out`. Pull the files locally with:
```
modal volume get pcache-out exp3_text.json .
modal volume get pcache-out exp3_mm.json .
modal volume get pcache-out exp3_audio.json .
```
Each JSON reports, per prefix length and per `cache_salt` setting: median hit latency, median
miss latency, a KS-test statistic and p-value (separability), a KDE-overlap leakage metric, and
the server's ground-truth cache-hit-rate for cross-checking.

**Verify the run is real, not degenerate:** the ground-truth hit-rate should be high on the
"hit" trials and ~0 on the "miss" trials; if hit and miss latencies overlap completely, check the
server actually had prefix caching on and the prefixes were block-aligned (block size 16).

**Rough cost:** the text run is small (an L4-class GPU, minutes); the image/audio runs use a
bigger GPU. Expect a few dollars of Modal credit total, not more — but the team should confirm
their own budget.

---

## 5. After the numbers exist

1. Fold the real latencies into `FINDINGS.md` (there are placeholders marked for the TTFT/TTFA
   numbers and the salt-on-vs-off cost).
2. Build the Oct 2 proposal deck. A natural arc: problem → how vLLM keys the cache → scope
   correction (SHA-256, section 3 above) → threat model → research questions → the exp3 results →
   recovery-cost model (exp4) → defense + its cost → feasibility/timeline. Reuse the team's CMU
   template (`~/Documents/…` Idea Portfolio deck) for visual consistency.
3. There is a published capability-boundary chart (confirm vs. reconstruct across the three
   modalities) the team can reuse as a slide.

---

## 6. Authorship note

The write-ups and any slides should be in the team's own voice for submission. Treat the
material here as research notes and verified facts to build from, not final submission text.
