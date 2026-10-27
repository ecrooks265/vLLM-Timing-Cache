# Budget and execution limits

User budget: maximum $40 for this study. Expected small-study target roughly $5–15, contingent on model load/startup time. This is not a promise of actual billing.

Observed 2026-10-26 on https://modal.com/pricing:

- A100 40 GB: $0.000583/s = $2.0988/hour GPU only.
- L4 (previous attempts only): $0.000222/s = $0.7992/hour GPU only.
- CPU: $0.0000131/physical core/s.
- RAM: $0.00000222/GiB/s.
- Volumes: $0.09/GiB/month (plan allowances may apply).

No reliance on unused account credits, no paid-plan upgrade, no non-preemptible multiplier, no explicit region, user explicitly chose A100 after L4 attempts. Single A100 40 GB, cpu=2, memory=32 GiB. The conservative local planning assumption is $3.00/hour for function compute; check provider pricing/account rates before launching elsewhere. CPU image builds/storage/network may be billed separately. Model weights are cached under pcache-out/multimodal_starter/model_cache and may have ongoing storage charges outside invocation caps.

Each remote invocation has an absolute function timeout of 7500 seconds, retries=0, max_containers=1; experiment body deadline is pilot 1800/main 7200 seconds. Startup has a 600-second timeout. The local wrapper holds a lock for the entire invocation, including download. Before launch it reserves the entire 7500 seconds at the configured conservative rate plus $1 build allowance: currently $7.25 for A100 ($3.60 for earlier L4 attempts). Failed invocations keep their full reservation. Local ledger refuses another run if cumulative reservations exceed $40. Do not delete/reset the ledger to bypass this guard.

**This is a conservative local launch guard, not an account-level provider hard cap.** Other tasks, manually bypassing the wrapper, simultaneous runs from another copied folder, a different effective price, image-build time beyond the allowance, and long-term storage can exceed the tracked estimate. For a true provider cap use an available account/environment budget facility manually; do not assume Starter has one. Never say "exact spend" from elapsed time alone; verify in Modal billing.

Use `python scripts/run_stage.py pilot --check` without charging. Formal run requires technical pilot gate, not positive experimental results. If pilot startup dominates, reduce sessions or run attack only first. Budget is a ceiling, not a target. Stop the study app when work is interrupted; do not stop other apps.

The wrapper checks once after five minutes: if its exact study app still has zero allocated tasks and no result receipt, it stops that app rather than waiting indefinitely. This also bounds a stalled allocation/build attempt; inspect the saved log before retrying. It never switches to a different GPU automatically.

User initially requested waiting for L4, then explicitly chose A100. The active entrypoint and configs use A100 40 GB. Use `--wait-for-gpu` to disable only the five-minute queue cancellation. Function timeouts, no retries, serialization, and budget reservations remain unchanged.

As of the formal launch: local cumulative reservations are $21.70 (two L4 attempts plus A100 pilot and main). This is NOT actual expenditure. At the observed published rates, 2 physical CPU cores and 32 GiB RAM plus A100 40 GB cost approximately $2.45/hour if all reserved CPU/RAM are billed, below the $3/hour planning rate. No account credits are assumed.
