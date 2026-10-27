# Multimodal KV-salt isolation study

Ruilin's extension tests whether a tenant can detect a known image submitted under another `cache_salt`, and measures the remaining latency benefit of reuse. It complements the existing text-recovery and compute-contention experiments.

## Measured image result

vLLM **0.11.0**, Qwen2.5-VL-3B-Instruct, **A100 40GB**, 336×336 images, eager mode and one generated token. Model revision, full flags, hardware and frozen source are saved with the run.

Forty held-out public bean-leaf photographs, eight separate calibration images and twelve separate performance images. Three repetitions, two fresh complementary attack sessions per mode/repetition. Each image receives only one attacker probe in each server session. Thirty fresh server sessions and 1,704 successful requests in total. This is a single visual domain.

| Salt mode | AUC in each repetition | AP | Paired unseen−primed latency, 95% bootstrap CI | Calibrated TPR | Observed FPR |
|---|---|---|---|---|---|
| Shared | 1.0 / 1.0 / 1.0 | 1.0 | 34.45 ms [33.53, 35.33] | 71.7% | 0% |
| Distinct | 1.0 / 1.0 / 1.0 | 1.0 | 34.52 ms [33.54, 35.69] | 68.3% | 0% |

All distinct-salt evaluation requests hit only 16 common-template KV tokens, whether victim-primed or unseen. Shared-salt primed requests hit 160 tokens. The residual timing signal therefore persists without the additional image-prefix KV hits observed under shared salt. Static source inspection supports a separate multimodal reuse path, but this study does not directly count processor/encoder hits.

AUC measures ranking. The pre-calibrated threshold misses some primed images, so AUC=1 does **not** mean 100% attack detection. The empirical AUC bootstrap interval [1,1] does not guarantee population perfection. Bootstrap resamples images and repetitions, with only three repetitions available.

## Reuse latency

Independent performance sessions use the separate twelve-image set. Values below are median first-content latency in milliseconds.

| Salt mode | Cold | Same tenant | Cross tenant | Paired cross-tenant saving, 95% CI |
|---|---:|---:|---:|---|
| Shared | 82.41 | 44.74 | 44.39 | 37.04 [33.33, 41.25] |
| Distinct | 79.80 | 46.74 | 45.74 | 33.99 [32.19, 36.89] |

The matched distinct-minus-shared cross-tenant difference is 0.71 ms, CI [-3.35, 3.51]. This does not establish equivalence or a cost-free defense. Throughput and a complete multimodal isolation fix were not measured.

## Files and CPU-only reproduction

Everything is in [experiments/multimodal_isolation](../experiments/multimodal_isolation). Read its [README](../experiments/multimodal_isolation/README.md), [detailed report](../experiments/multimodal_isolation/docs/FINAL_RESULTS_zh.md), [protocol](../experiments/multimodal_isolation/docs/PROTOCOL.md), and [historical modality boundary](../experiments/multimodal_isolation/docs/DEFENSE_BOUNDARY.md).

```bash
cd experiments/multimodal_isolation
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/analyze.py results/runs/main-20261027T023818Z-bd4e43c4
python scripts/finalize_results.py results/runs/main-20261027T023818Z-bd4e43c4
```

These commands analyze saved data without renting a GPU. Modal pilot/main launch commands are separate and billable.

Historical text and audio evidence is retained in `legacy/`, with its older first-chunk timing definition and original models. Do not pool those measurements with new first-content results or the repository's ROCm experiments. The new attack tests membership of an exact known image, not reconstruction. Localhost serial requests, one model/version/device and limited session repetitions constrain generalization. Brief read-only hardware/source checks during execution may have introduced transient system disturbance. No observer patch was installed in timing runs.
