---
name: sycl-optimization
description: >
  Optimize a migrated SYCL kernel for Intel Xe2/Xe3 (Battlemage Arc Pro B-series / Crescent Island) using a
  profile-guided search loop: classify the bottleneck, apply ONE optimization at a time, keep it
  only if it is faster AND still correct (else git-revert), and record every trial. Use when:
  a kernel misses its roofline expectation, tuning work-group/sub-group size, vectorization,
  native math, XMX offload, escaping a local optimum, or deciding when to stop. Correctness is a
  hard gate; every trial is logged to .sycl/state/optimization/<id>.json.
---

# SYCL Optimization

Improve performance **without ever sacrificing correctness**, on the highest-impact kernels first
(order comes from `.sycl/state/profile/e2e.json`). This skill is a disciplined search, not a random
one — it remembers what it tried and why it stopped.

## Required reading
- `sycl-reference` → `references/sycl-optimization-catalog.md` (profiling-signal → strategy lookup
  table + anti-patterns; its columns feed this skill's ANALYZE/combination/exploratory phases).
- `sycl-reference` → `references/intel-gpu-hardware.md` (roofline, ridge point).
- `sycl-reference` → `references/intel-gpu-hardware-metrics-bmg.md` /
  `references/intel-gpu-hardware-metrics-cri.md` (counters, thresholds, and the `XVE_STALL_*` crosswalk)
  when a DIAGNOSE trigger cites a specific counter/threshold and you need to read it off the CSV.
- The kernel's deep-profile result from `sycl-profiler` §4 (bottleneck class + dominant stall +
  register-spill flag) — this drives step 0.5 DIAGNOSE.
- `references/search-strategy.md` (this skill) — how to find good *combinations*.

For a hard bottleneck, consult `sycl-reference` → `references/intel-gpu-software-repos.md` for how a
comparable kernel is tuned in a local upstream repo (oneDNN, sycl-tla, Triton XPU) — borrow the
approach (tiling, sub-group width, `joint_matrix` use), never the source.

## The loop (per kernel)
```
0. SETUP     git checkpoint branch; load baseline + bottleneck class from profiler; ensure a
             kernel-level benchmark + a JIT (non-AOT) profiling build exist (write them if missing)
0.5 DIAGNOSE deep-profile the KERNEL BENCHMARK (sycl-profiler §4) with unitrace: HW metric counters +
             stall sampling + (when useful) IGC shader/asm dump. Combine the three into a root cause.
1. ANALYZE   rank candidate optimizations by the DIAGNOSE root cause (not by roofline class alone)
2. APPLY     make ONE code change (one hypothesis)
3. VALIDATE  run.sh build -> run.sh test (correctness is a HARD GATE)
4. MEASURE   run.sh bench on the KERNEL BENCHMARK across its shape set (median of warm-up + iters)
5. DECIDE    faster AND correct -> git commit (KEEP); else git revert (REJECT + log root cause)
6. RECORD    append the trial to optimization/<id>.json; regen PROGRESS.md
7. REPEAT    next hypothesis, until a stop criterion fires
```
Every step's shell work goes through `.sycl/scripts/run.sh` (local or remote GPU host).

**Measure and profile the kernel, never the e2e workload.** The e2e run (`.sycl/state/profile/e2e.json`)
is only for *ranking* which kernel to optimize. Once a kernel is selected, both MEASURE (step 4) and
DIAGNOSE (step 0.5) run against the **standalone single-kernel benchmark** (`bench_<id>`), not the full
application. Re-running the whole workload to time one kernel is slow, noisy, and is what makes unitrace
core-dump on large traces — the small single-kernel driver profiles cleanly every time.

### SETUP: ensure the kernel benchmark + JIT profiling build (step 0)
Before the loop, confirm the kernel has:
- **A kernel-level benchmark** (`bench_<id>`). If missing, **write one** using the `sycl-profiler` §2
  methodology and `sycl-migration/templates/kernel_bench.cpp`: a single-kernel driver over
  representative shapes — popular **model configs** (real hidden/head/seq/batch sizes the kernel
  serves), a couple of **general** saturating sizes, and **edge** cases (tiny, non-power-of-2,
  masked/boundary, near-memory-limit). This is the measurement target for the whole loop.
- **A JIT (non-AOT) build of the profiling target.** Build `bench_<id>` without `-fsycl-targets=spir64_gen`
  so the runtime IGC shader dump used by stall→asm annotation is non-empty (an AOT build dumps at build
  time — see `sycl-profiler` §4). If the project ships AOT, either keep a JIT profiling build alongside it,
  or capture a build-time dump and feed `profile_metrics.sh` with `SHADER_DUMP=prebuilt SHADER_DUMP_DIR=<dir>`.

## Diagnose before you tune (step 0.5)
Do **not** pick optimizations from the coarse roofline class alone — get evidence from `sycl-profiler`
deep profiling and combine the lenses. The mapping from evidence to first move:

| Evidence (counters + stall + asm) | Root cause | Try first |
|---|---|---|
| memory-bound roofline, high `SendStall`, no spills | uncoalesced / too much traffic | coalescing, `vec4` loads, fuse passes, cut redundant loads |
| high `SendStall` **and** register spills in asm | register pressure hurting latency hiding | shrink live set, smaller tiles/WG, `[[intel::grf]]` tuning |
| compute-bound, high `PipeStall`/`DistStall` | low ILP / dependency chains | unroll, interleave independent work, reorder |
| high `SbidStall` | few in-flight async ops | more outstanding loads, software pipeline |
| low occupancy (counter), no stall dominance | not enough threads resident | WG size, reduce SLM/registers per WG |
| idle XMX on a matmul (asm shows no DPAS) | not using matrix engines | route to `joint_matrix` / oneMKL / oneDNN |
| high `SyncStall` | over-synchronization | fewer/rebalanced barriers |

Record the chosen root cause in `optimization/<id>.json` per trial so a reverted trial teaches the
next one.

**Re-diagnose only when the bottleneck likely shifted — not after every trial.** Each trial is
*verified* cheaply by the correctness gate (step 3) + one benchmark (step 4); that is enough to
keep-or-revert. A full deep profile is a *diagnosis*, not a verification, so re-run it only when the
limiter has probably moved:
- after a trial that **kept a large win** (e.g. you fixed the dominant `SendStall` — the kernel may now
  be compute- or occupancy-bound);
- when a trial's result is **surprising** vs the prediction (helped or hurt unexpectedly);
- when the current root cause's **checklist is exhausted** and you need the next dominant stall.

**Deterministic cadence fallback (so the loop never runs blind on a stale diagnosis).** The three
triggers above are judgement calls; back them with a mechanical rule that fires even when nothing
*looked* surprising: force a re-diagnosis after **5 trials since the last profile** or **3 consecutive
rejects on the current branch**, whichever comes first (also re-diagnose whenever you checkpoint-branch,
since you restore an earlier kernel). Use the cheap tier for these forced checks — a `metrics`-only pass
first, escalating to a full deep profile only if the limiter actually moved.

For a quick mid-loop re-check, prefer the cheap metrics-only pass over a full deep profile:
`PROFILE_MODE=metrics METRICS_GROUPS=<suspect>` (e.g. `MemoryProfile`, `DeviceCacheProfile`) — it skips
stall sampling and the IGC dump. Small keeps and rejects need **no** re-diagnosis: the existing
root cause still holds.

**Retain audit artifacts for baseline and final.** Keep the full deep-profile output — metric CSVs
**and** the IGC shader/asm dump — for two points so a human can audit the before/after:
`.sycl/reports/opt_<id>/baseline/` (the step 0.5 diagnosis on the migrated kernel) and
`.sycl/reports/opt_<id>/final/` (a deep profile of the best kept commit). Run the helper with
`OUT_DIR` set so these are not overwritten by intermediate runs, e.g.:
```bash
.sycl/scripts/run.sh profile \
  "OUT_DIR=.sycl/reports/opt_<id>/baseline \
   bash framework/skills/sycl-profiler/scripts/profile_metrics.sh <build-dir>/bench_<id>"
# ... optimize ...
.sycl/scripts/run.sh profile \
  "OUT_DIR=.sycl/reports/opt_<id>/final \
   bash framework/skills/sycl-profiler/scripts/profile_metrics.sh <build-dir>/bench_<id>"
```
Record both paths in `optimization/<id>.json` (`diagnosis.artifacts.baseline` / `.final`). Intermediate
per-trial profiles may use a scratch dir and be discarded.


## Rules
- **One hypothesis per trial.** Stacking changes hides which one helped or regressed.
- **Correctness gate first.** If the unit test fails after a change, revert immediately — never keep a
  faster-but-wrong kernel.
- **Profile-guided, not blind.** Choose the next optimization from the DIAGNOSE root cause (counters
  \+ stall reason + asm), not by guessing. Memory-bound/`SendStall` → coalescing, vec4, fewer passes,
  fusion. Compute-bound/`PipeStall`/`SbidStall` → native math, unrolling, ILP, XMX. Occupancy-bound →
  WG size, register pressure (watch for spills in the asm dump).
- **Respect the anti-pattern catalog.** Do not re-try things known to regress on Intel Xe (SLM caching of
  read-only data, forced sub-group size on simple kernels, 64-bit index math, etc.). If you try one
  anyway to confirm, log the result.
- **The compiler is often smarter than manual tuning.** When unsure, measure both and keep the winner.
- **Never benchmark or diagnose via the full e2e workload.** MEASURE and DIAGNOSE always run the
  standalone kernel benchmark (`bench_<id>`); the e2e profile is only for choosing *which* kernel to
  optimize. Always profile that kernel benchmark with unitrace (metrics + stall) — unitrace does not
  crash on a single-kernel driver, so there is no excuse to skip it. Build the profiling target JIT so
  the shader dump works.

## Finding good combinations (the hard part)
See `references/search-strategy.md`. In short:
1. **Greedy first**: apply individually-winning optimizations one at a time; keep each winner.
2. **Combination trials**: once you have 2–3 individual winners that plausibly interact (e.g. WG-size
   \+ vec4), test them together — interactions can be super- or sub-additive. Mark these trials
   `combination: true`.
3. **Checkpoint branching** to escape local optima: from the best commit, branch and explore a
   different strategy family; keep the branch only if it beats the current best.
4. **Exploratory acceptance**: occasionally keep a neutral-but-simpler change (simplicity criterion)
   to open new paths — but never a regression.

## Stop criteria (record `stop_reason`)
- `target-met` — achieved ≥ the roofline-derived expectation (e.g. ≥60–70% of relevant peak).
- `diminishing-returns` — several consecutive trials yield < a small threshold (e.g. <2%).
- `checklist-exhausted` — no untried, applicable strategy remains.
- `correctness-block` — the only remaining ideas break correctness; send the kernel back to migrate.

## State (own these)
- `.sycl/state/optimization/<id>.json` — the trial log (validate against `optimization.schema.json`):
  baseline, target, best (value+commit), every trial (hypothesis, bottleneck class, commit, build,
  correctness, metric before/after, keep/revert, root cause), and `stop_reason`.
- Update the kernel detail `performance` block (`optimized`, `fraction_of_peak`, `meets_expectation`).
- Regenerate PROGRESS.md; log each kept/reverted trial.
- Append durable Intel Xe2/Xe3 findings (with environment) to `.sycl/state/lessons.md`.

## Output (return to orchestrator)
- Kernel `<id>`: baseline → best (metric, fraction of peak).
- Trials: kept/reverted counts; notable anti-patterns confirmed.
- `stop_reason`; whether it now meets expectation.

## Assets
- `references/search-strategy.md` — greedy + combination + checkpoint-branching search recipe.
