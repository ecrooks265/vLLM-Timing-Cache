# Experiments — replicating Gu et al. on vLLM (white-box)

How to reproduce the base study on this stack, what each experiment maps to in
the paper, and what we add by being white-box. Run with `config/replication.yaml`
(ungated Qwen-0.5B) now; swap `model.name` to Llama-3.1-8B for the paper's model
once HF access lands — nothing else changes.

## The method (what "replicate" means here)
The paper detects prompt caching as a timing side channel: send a prefix, measure
time-to-first-token (TTFT); a cache **hit** is faster than a **miss** because the
prefill is skipped. It frames detection as a two-sample test over hit-vs-miss TTFT
distributions and reports the **average precision (AP)** of a timing classifier.

Black-box, they could only *infer* the hit/miss label from timing. We serve vLLM
ourselves and read its real prefix-cache counters from `/metrics`, so every label
is **ground truth** — the AP we report is measured, not inferred. That is the whole
point of the white-box testbed.

## Cache levels on local vLLM (important nuance)
vLLM's prefix cache is **process-global** — it is not partitioned by the OpenAI
`user` field. So:
- **per_user** (same user primes + probes): confirms the timing signal exists.
- **global** (different users): a hit proves the cache is shared *across users* —
  the real-world threat. On stock vLLM this is the **default**, so the channel is
  present cross-user out of the box. That absence of isolation is itself a finding
  and the baseline our defense experiments (04/06) measure against.

## Experiment map
| Exp | Paper element | Command | Replicate / extend |
|-----|---------------|---------|--------------------|
| 01 prove side channel | Detection + AP (Tbl 1) | `python scripts/run_experiment.py --config config/replication.yaml --audit` | **Replicate**, white-box |
| 03 calibration sweep | — (they couldn't) | `python experiments/03_calibration_sweep.py config/replication.yaml` | **Extend**: AP vs prefix length w/ ground truth |
| 05 suffix collision | "same prefix, diff suffix" | `python experiments/05_suffix_collision.py` | Replicate |
| 02 / 09 prefix recovery | their stated limitation | `python scripts/run_experiment.py --config config/replication.yaml --recover` | **Extend**: the attack they didn't build |
| 04 / 06 defenses | proposed, unmeasured | `python experiments/04_defense_salting.py` | **Extend**: price each defense |

## Run the core replication (Exp 01)
```bash
# 1. server up on the replication config (Qwen, ungated)
CACHELAB_CONFIG=/workspace/config/replication.yaml \
  docker compose -f docker/docker-compose.rocm.yml up -d
curl http://127.0.0.1:8000/health          # -> 200

# 2. harness venv (client only — no vLLM, no GPU)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. run the audit (writes runs/<ts>/requests.jsonl + summary.json)
python scripts/run_experiment.py --config config/replication.yaml --audit

# 4. figures (writes runs/<ts>/figures/)
python scripts/make_figures.py
```
Console prints per level: `AP=… KS p=… n=…`, and `PASS` when
`KS p < alpha AND AP >= 0.70`.

## Outputs
- `runs/<ts>/requests.jsonl` — one row per request: TTFT (client) + **ground-truth**
  hit/miss from `/metrics`, cache level, condition.
- `runs/<ts>/summary.json` — KS test + AP + best precision/recall per level.
- `runs/<ts>/figures/` — `ecdf_<level>`, `hist_<level>`, `pr_<level>`,
  `audit_ap`, `calibration` (PNG+SVG) and `figures.md` (results table).

## Faithfulness checklist (vs the paper)
- [x] Same statistic (two-sample KS) and headline metric (average precision).
- [x] NUMSAMPLES = 250, warmup discarded.
- [x] Prefix caching on; hit = primed prefix, miss = unique prefix.
- [x] Ground-truth labels (improves on black-box inference).
- [ ] Exact model (Llama-3.1-8B) — pending HF access; Qwen-0.5B stands in now.
- [ ] block_size=1 token granularity — ROCm V1 uses 16 (see README caveat); affects
      only recovery granularity (Exp 02/09), not detection (Exp 01).
