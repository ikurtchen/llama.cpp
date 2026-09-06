# Project Agents — CUDA → SYCL Migration

This project uses the **sycl-agent** framework to migrate CUDA kernels to SYCL 2020 and integrate
them as a **fully functional SYCL/XPU backend** of this project for
Intel Xe2/Xe3 GPUs (Arc Pro B60/B70 on Battlemage; Crescent Island on Xe3P), driven by a
single orchestrator agent plus on-demand skills. The active target is `target.platform` in
`.sycl/config.json`.

## Entry point
Invoke the **sycl-agent** agent. It sequences the workflow and loads skills as needed:
- **cuda-analysis** — scan/parse CUDA, extract algorithms, build the inventory.
- **sycl-migration** — hand-write SYCL 2020, set up the SYCL build, author + run unit tests.
- **sycl-integration** — make SYCL a real backend: backend seam, device runtime layer, host `icpx`
  port sweep, host-library replacement, op registration/packaging, e2e validation, readiness level.
- **sycl-profiler** — env detection, roofline, profiling, benchmarking (via the runner).
- **sycl-optimization** — profile-guided, one-change-at-a-time tuning loop.
- **sycl-review** — close a phase: run its exit gate, adjudicate what the gate cannot check, record
  the verdict. A phase is never skipped — it is closed, or waived out loud.
- **sycl-reference** — sharded, cited Xe2/Xe3 + SYCL knowledge base other skills draw on.

## Where things live
- Agent + skills: see your engine's directory (`.github/`, `.claude/`, or `.opencode/`)
- Runtime state (source of truth): `.sycl/state/*.json` — start at `.sycl/state/project.json`
- Phase-exit contract (what each phase delivered, gate results, waivers, frozen scope):
  `.sycl/state/phases.json`
- Backend readiness + the last-mile surfaces: `.sycl/state/integration/`
- Human dashboard (generated): `.sycl/PROGRESS.md`
- Execution config (local/remote runner): `.sycl/config.json`
- Knowledge & workflow refs: `.sycl/references/`
- Durable lessons: `.sycl/state/lessons.md`
- Audit logs (rotated): `.sycl/logs/`
- Evidence behind every claim: `.sycl/reports/evidence/` — indexed for humans in `.sycl/reports/AUDIT.md`

## Rules
- Never use legacy auto-translators (dpct/SYCLomatic) — hand-write SYCL from the algorithm.
- Follow SYCL 2020; optimize for the active Intel Xe2/Xe3 target in `.sycl/config.json`.
- Route all build/test/profile/benchmark through `.sycl/scripts/run.sh` (local or remote GPU host).
- State JSON is authoritative; regenerate `.sycl/PROGRESS.md` after every change.
- **No claim without a record.** Capture every test/benchmark you cite with
  `.sycl/scripts/evidence.sh record` and store its run id in the matching `evidence` field;
  `.sycl/scripts/evidence.sh verify` must pass before leaving `scaffold`, `integrate`, `optimize`
  and `done`.
- **No phase without a gate.** Enter each phase with `.sycl/scripts/evidence.sh enter <phase>` and
  leave it with `.sycl/scripts/evidence.sh gate <phase>` — the only thing that may mark a phase
  exited. `verify` asks whether what you said is true; `gate` asks whether you did the thing the
  phase exists to do. A criterion you cannot meet is **waived** with a reason and a blast radius,
  never passed over in silence, and never fixed by moving a threshold or re-cutting the inventory.
- **The deliverable is a working backend, not a kernel layer.** Never fork the project into a
  standalone SYCL tree: the gates are the project's **own** build, **own** test suite and **own**
  entrypoint. "Runs on Intel GPU" may only be written once readiness L3 is backed by evidence.
- Read `.sycl/references/state-model.md` before writing any state, and
  `.sycl/references/evidence-model.md` before writing any result.
- Honor the opt-in `guardrails` in `.sycl/config.json`: on a GPU hang, only kill the process —
  never reset/reload the GPU driver (stop and report if it won't die); and never skip kernels or
  stop early for difficulty (`no_lazy_completion`).
