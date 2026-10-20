# cache-timing-lab — documentation

- **[VERSIONS.md](VERSIONS.md)** — the exact known-good version combination
  for this environment and the WSL glue that makes it work.
- **[SETUP.md](SETUP.md)** — get the vLLM server running on the 7900 XTX via
  Docker on WSL2, and pick up testing any time.
- **[REPLICATION.md](REPLICATION.md)** — results of the base-study
  replication on this stack, with figures.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — reproduce the study on this
  stack: protocol mapping, commands, outputs, faithfulness checklist.
- **[RESEARCH.md](RESEARCH.md)** — the side-channel paper (Gu et al.), what we
  extend, the experiment map, and the block_size caveat.
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — gfx1100 / ROCm / WSL / Docker
  gotchas and fixes.

Quick env check: `bash scripts/doctor.sh`
