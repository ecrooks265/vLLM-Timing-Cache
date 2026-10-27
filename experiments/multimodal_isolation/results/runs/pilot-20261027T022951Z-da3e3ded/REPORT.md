# New image isolation results

**PILOT ONLY — not final evidence.**

| Salt mode | ROC AUC | AP | Paired cold−cached ms | TPR | FPR |
|---|---:|---:|---:|---:|---:|
| distinct | 1.000 | 1.000 | 33.94 | 1.000 | 0.000 |

| Salt mode | Cold ms | Same-tenant ms | Cross-tenant ms | Paired cross saving ms |
|---|---:|---:|---:|---:|
| distinct | 80.28 | 47.61 | 47.65 | 32.63 |

All timing is client-side localhost first-content latency. AP baseline is 0.5 for this balanced design.
No claim of zero leakage follows from a nonsignificant test. KV counters do not directly observe encoder hits.
One leaf-photo dataset, one VLM, one vLLM version. This is membership testing, not image reconstruction.

See analysis.json for confidence intervals, per-repetition results, and performance values.
