# D: Direct mechanism evidence — optional, separate from timing

The original unverified implementation is preserved in:

- legacy/modal_exp3_ttft.py: patch_check(), mm_direct(), probe_direct().
- legacy/_vllm_patch/encoder_cache_manager.py and cache.py.

Do not invoke these as an already verified experiment. No direct-result file was present in the original snapshot. Existing read-before/read-after file counters may count method invocations rather than unique requests. Processor-cache class selection must be checked; only instrumenting one subclass can falsely imply silence. Scheduler checks can occur repeatedly for one request. Missing files must not mean zero hits. Instrumentation writes may affect timing.

## Required preflight before spending

1. Download/read pristine source for exactly installed vLLM 0.11.0, compare against patched copies. Only observer changes allowed. Compile both modules. Record source version and hashes.
2. Verify actual classes created by processor_cache_from_config()/engine_receiver_cache_from_config()/worker_receiver_cache_from_config(). Instrument active paths, not guessed locations.
3. Prefer events with request_id, process_id, cache layer, media-key digest, decision and event time. Distinguish a scheduler lookup from a unique media computation. No prompt/image bytes in logs.
4. Place events into a per-run location. Ensure completion/flush before readback. Repeated checks aggregated by request, not naively called request hits.
5. Prove cold image produces a miss/computation, same-tenant repeat can reuse, different-tenant repeat is labeled independently, and new image differs. A log path existing is not proof.

## Small mechanism-only run

Use 4 images, then 10 if pilot is valid, A100, max_tokens=1. Same processor setting/context as timing run. Run baseline and processor-cache-off as two separate sessions. Do not perform 400/1200-image floods by default. If eviction is required, demonstrate the observed eviction rather than assume a fixed flood count guarantees it.

Collect KV delta plus active processor/encoder events per request. Compare contents and salt relationships. The question is whether a different-salt request reuses victim-produced multimodal state, not whether a method was called.

## Acceptance and reporting

Directly observed mechanism requires request-linked events that distinguish cached/unseen conditions and known negative controls. Otherwise retain "consistent with multimodal caching; attribution inferred from ablations". A diagnostic counter patch is not a security fix. Restoring a genuine tenant-isolation fix and measuring its overhead is future work, not part of tonight's required completion.

Timebox investigation to one hour / remaining budget; if unresolved, finish B/C and ship a documented limitation. Do not install observer patches into the official timing run.
