# Public release scope and provenance

This directory contains Ruilin's multimodal study, completed on 2026-10-26. The image main run uses vLLM 0.11.0 and Qwen2.5-VL-3B on A100 40GB. It is independent of the repository's existing ROCm/text-recovery harness.

Included: runnable source and configs, eight offline protocol/analysis tests, public normalized images and source/hash manifest, completed pilot/main raw requests, environment metadata, exact run-source snapshots, figures and reports. `legacy/` retains the original pcache research code, saved measurements and notes, including historical text/audio evidence. Its included files are byte-preserved and listed in LEGACY_INCLUDED_SHA256.json. Unvalidated legacy patches and historical claims remain historical evidence, not validated components of the new runner.

Excluded: local terminal/server logs, account budget ledger and launcher receipts, personal coordination notes, redundant ZIP, virtual environments, Python caches and unrelated individual Student Lecture materials. No credentials or model weights are included. Analysis JSON source_run fields use portable relative paths; original numeric request records and frozen source files are unchanged.

The local image main was analyzed with 1,000 bootstrap resamples. Classification thresholds were calibrated before held-out evaluation. AUC=1.0 does not imply the calibrated detector detects every primed image. Only the two shipped run directories are completed evidence. Regenerating analysis is CPU-only; launching Modal functions incurs compute charges.

Dataset: AI-Lab-Makerere/beans, public validation split, dataset metadata declares MIT. The exact dataset revision, original source URLs and original/normalized file hashes are in data/manifest.json. The photographs cover bean leaves, a single visual domain. No private user media is included.

The project source does not introduce a new project-wide license. Original upstream code excerpts in static_source_evidence.json retain their source identity and are evidence for vLLM 0.11.0, not direct runtime cache-hit observations.
