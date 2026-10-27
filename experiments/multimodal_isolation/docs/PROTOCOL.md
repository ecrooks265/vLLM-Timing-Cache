# Measurement and inference protocol

## Cache and tenant semantics

`shared`: victim and attacker use the same namespace. `distinct`: they use different namespaces. These are experimental namespace relationships, not proof that an API key/user field enforces identity. Trusted servers must assign salts for real isolation; allowing an adversary to choose a victim salt would violate that assumption.

The attacker knows a candidate image and observes only its own latency. It does not use privileged counters for its decision. Counters are validation evidence. No concurrent background requests in this study; global metric deltas cannot provide per-request ground truth under other traffic.

## Image crossover

For each repetition and salt mode, two fresh servers receive complementary, balanced labels. Every evaluation/calibration image is victim-primed in exactly one pass and unseen in the other. Attack probe count is exactly one per image per server. Unseen trials make a sham prime on a reserved warmup image to avoid simply omitting victim activity. Cache warmth of that sham is a remaining difference; all requests complete serially, no intentional contention.

Both tenant namespaces and execution paths are warmed with images excluded from test/performance sets. After every request the entire stream is drained. All data enters the model as normalized RGB 336×336 PNG bytes; hashes identify the exact submitted content. The same image/normalization is reused across modes and repetitions. Data source assets are not kept as expiring signed URLs in the manifest.

## Timing

Clock: `perf_counter`. Request payload/encoding is prepared before starting the timer. Record first JSON SSE chunk, first nonempty `delta.content`, and full completion after DONE. Role-only chunks do not count as content. HTTP requests' SSE reader uses chunk_size=1 to avoid 512-byte client buffering. CPU overhead is part of localhost client-observed latency. max_tokens=1; no claim about long generation.

Metrics are read before and after complete stream consumption, with a short bounded wait for query updates. Missing or non-updating metrics are marked missing_or_delayed, never zero. Query updates must be demonstrated by pilot; delayed batched metric publication can still complicate attribution. Preserve raw before/after counters. No assumption that common chat-template tokens produce zero KV hits.

## Independent calibration

Each session threshold is learned exclusively from calibration images by maximizing balanced accuracy. Threshold direction is frozen: cached predicts lower first-content latency. Held-out evaluation images are never used to tune it. Report TPR, FPR and n. AP/ROC AUC use cached=positive and score=-latency, so 1 is strong evidence and .5 is chance. Reverse direction found in old `legacy/` output must not be compared numerically without explicit conversion.

For each mode report paired image cold−cached median difference, per-repetition metrics, and 95% bootstrap intervals. Bootstrap samples whole image IDs and whole repetitions, preserving two-pass pairs. With three repetitions the across-session uncertainty is weakly characterized; report all three. No significance claim is based on selecting a favorable hypothesis family after seeing results. No nonsignificant p-value proves absence of leakage. A balanced AP baseline is 0.5 only for this experimental class balance, not all deployments.

## Cost experiment

Performance images never appear in attack sessions. Cold/same/cross are in three independent servers per mode/repetition, with the same normalized images. Latency paired by image and repetition. Same-tenant repeated requests are allowed only in the dedicated performance protocol. Do not repeat attack probes to reduce noise: they populate the attacker's cache. Report per-rep cost medians and paired savings; no throughput claim.

## Failure and output handling

Any HTTP, stream, or server-start failure aborts the run and preserves metadata/logs. No silent omission/replacement/retry. Run state says failed until all intended sessions finish. Analysis gates partial runs from formal reporting. Output paths carry unique IDs and never overwrite legacy or previous runs. Pilot data are excluded from final measurements.
