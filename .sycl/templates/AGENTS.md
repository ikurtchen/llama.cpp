# Project Agents — CUDA → SYCL Migration

This project uses the **sycl-agent** framework to migrate CUDA kernels to SYCL 2020 for
Intel Xe2/Xe3 GPUs (Arc Pro B60/B70 on Battlemage; Crescent Island on Xe3P), driven by a
single orchestrator agent plus on-demand skills. The active target is `target.platform` in
`.sycl/config.json`.

## Entry point
Invoke the **sycl-agent** agent. It sequences the workflow and loads skills as needed:
- **cuda-analysis** — scan/parse CUDA, extract algorithms, build the inventory.
- **sycl-migration** — hand-write SYCL 2020, set up the SYCL build, author + run unit tests.
- **sycl-profiler** — env detection, roofline, profiling, benchmarking (via the runner).
- **sycl-optimization** — profile-guided, one-change-at-a-time tuning loop.
- **sycl-reference** — sharded, cited Xe2/Xe3 + SYCL knowledge base other skills draw on.

## Where things live
- Agent + skills: see your engine's directory (`.github/`, `.claude/`, or `.opencode/`)
- Runtime state (source of truth): `.sycl/state/*.json` — start at `.sycl/state/project.json`
- Human dashboard (generated): `.sycl/PROGRESS.md`
- Execution config (local/remote runner): `.sycl/config.json`
- Knowledge & workflow refs: `.sycl/references/`
- Durable lessons: `.sycl/state/lessons.md`
- Audit logs (rotated): `.sycl/logs/`

## Rules
- Never use legacy auto-translators (dpct/SYCLomatic) — hand-write SYCL from the algorithm.
- Follow SYCL 2020; optimize for the active Intel Xe2/Xe3 target in `.sycl/config.json`.
- Route all build/test/profile/benchmark through `.sycl/scripts/run.sh` (local or remote GPU host).
- State JSON is authoritative; regenerate `.sycl/PROGRESS.md` after every change.
- Read `.sycl/references/state-model.md` before writing any state.
