# New image isolation results

Measured new run; legacy results are separate.

| Salt mode | ROC AUC | AP | Paired cold−cached ms | TPR | FPR |
|---|---:|---:|---:|---:|---:|
| shared | 1.000 | 1.000 | 34.45 | 0.717 | 0.000 |
| distinct | 1.000 | 1.000 | 34.52 | 0.683 | 0.000 |

| Salt mode | Cold ms | Same-tenant ms | Cross-tenant ms | Paired cross saving ms |
|---|---:|---:|---:|---:|
| shared | 82.41 | 44.74 | 44.39 | 37.04 |
| distinct | 79.80 | 46.74 | 45.74 | 33.99 |

All timing is client-side localhost first-content latency. AP baseline is 0.5 for this balanced design.
No claim of zero leakage follows from a nonsignificant test. KV counters do not directly observe encoder hits.
One leaf-photo dataset, one VLM, one vLLM version. This is membership testing, not image reconstruction.

See analysis.json for confidence intervals, per-repetition results, and performance values.
