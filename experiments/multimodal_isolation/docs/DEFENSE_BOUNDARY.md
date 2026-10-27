# Defense boundary and evidence provenance

## Scope

The new experiment tests whether an attacker who already owns an exact candidate image can infer prior submission by another salt namespace. A successful binary membership test does not recover an unknown image or quantify its semantic contents. No information-theoretic leakage rate in bits is estimated.

## Historical context (kept separate from new first-content timing)

| Modality | Model | vLLM | Historical salt-on evidence | Boundary |
|---|---|---|---|---|
| Text | Qwen/Qwen2.5-3B-Instruct | 0.11.0 | 3 server repetitions, 30 observations per arm, AP approximately 0.51–0.54, latency difference near zero | This tested setup suppresses the text prefix timing signal. It does not prove universal text confidentiality. |
| Images | Qwen/Qwen2.5-VL-3B-Instruct | 0.11.0 | 3 repetitions, 15 per arm, AP 1.0 in saved results, median latency difference approximately 65–67 ms | Legacy generated-image evidence. New natural-image protocol is reported separately. |
| Audio | Qwen/Qwen2-Audio-7B-Instruct | 0.11.0 | 3 repetitions, 15 per arm, AP 0.968–1.0, median difference approximately 56–60 ms | Legacy audio evidence only. Lossless format conversion did not necessarily reuse the same entry. |

Sources: legacy/exp3_repeat_text.json, legacy/exp3_repeat_mm.json, legacy/exp3_repeat_audio.json and the corresponding RESULTS files. Original AUC used the opposite orientation. We deliberately do not copy legacy AUC values into the new study. Legacy timing stopped at the first SSE chunk, unlike the new full-stream first-content measurement. Historical absolute timing must not be pooled or compared as a controlled performance change.

## New evidence

Formal run: main-20261027T023818Z-bd4e43c4. Complete metadata and all seven technical analysis gates passed. See FINAL_RESULTS_zh.md and results/runs/main-20261027T023818Z-bd4e43c4/analysis.json for measured results. Main configuration uses Qwen2.5-VL-3B, vLLM 0.11.0, A100 40GB, 40 held-out leaf photographs, separate 8-image calibration, 12-image performance set, three repetitions and shared/distinct salts.

KV hit counters distinguish observed KV reuse from a residual timing signal. They do not directly label the processor or encoder cache hit. The mechanism claim therefore remains limited without separately validated instrumentation or controlled ablation. No observer patch is installed in official timing runs.

## Limits on statistical interpretation

Analysis bootstraps whole images and repetitions, preserving paired passes. There are only three repetitions, so session-level uncertainty is weakly estimated. If every resample has perfect separation, a degenerate empirical bootstrap interval of [1,1] describes these observations and is not a guarantee of perfect population performance. Calibration images never enter the held-out scores.

One image domain, one model/version/device, localhost timing, single serial client, controlled salting via request parameter. No commercial API audit, throughput benchmark, network-jitter study, arbitrary-image reconstruction or complete isolation fix is claimed.
