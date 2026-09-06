---
name: sycl-integration
description: >
  Turn a migrated SYCL kernel layer into a fully functional SYCL/XPU backend of the project itself —
  so the project's own build, runtime, dispatcher and entrypoint run the real end-to-end workload on
  an Intel GPU, not just standalone kernel tests. Covers the backend seam and walking skeleton, the
  SYCL/XPU device runtime layer (device/context/queue/event, USM allocator + memory pool), the
  host-code `icpx` port sweep, host library replacement (Thrust/CUB→oneDPL, cuBLAS→oneMKL,
  NCCL→oneCCL), framework op registration (PyTorch XPU custom ops, TF, DGL), Python packaging,
  end-to-end workload validation, and the Backend Readiness Level (L0–L4). Use when: adding a
  SYCL/XPU backend, wiring migrated kernels into the real build/runtime, making `import <pkg>` work
  on XPU, registering torch.ops for XPU, replacing cudaMalloc with USM, running the real workload
  end-to-end on Intel GPU. Kernel translation → sycl-migration; kernel tuning → sycl-optimization.
---

# SYCL Backend Integration (last mile)

Migrating kernels proves the **math** runs on an Intel GPU. It does not make the **project** run on
an Intel GPU. This skill closes that gap: it makes SYCL a real backend of the project — selected by
the project's own build, driven by the project's own runtime, reached by the project's own
entrypoint — and it makes the depth actually achieved **measurable and declared** instead of
discovered by a reviewer an hour after the report lands.

> **The failure this skill exists to prevent.** A kernel layer validated only by standalone SYCL
> binaries under `dev/sycl/` is **L0**. It is a real deliverable, but the sentence "project X now
> runs on Intel GPU" is false at L0 and every reader will assume it is true. Integration is not a
> tidy-up at the end of migration; it is the deliverable the user actually wanted.

## Hard rules
- **Never fork the project.** SYCL is a *second backend inside the project*, selected by a switch,
  sharing the project's host code, build, tests, and entrypoint. A parallel SYCL-only tree with its
  own `main`, its own headers and its own tolerances is a **failed** integration, not a partial one.
- **The gate is the project's own artifact.** "It builds" means *the project's own build target*
  (`libcudf.so`, `train_gpt2`, `import cupy`, `pip install -e .`) succeeds with the backend enabled —
  not that a test binary compiled.
- **Declare the target level before you start, measure the achieved level at the end.** Both go in
  `.sycl/state/integration/readiness.json`. The report prints them side by side. Under-delivery is
  allowed; silent under-delivery is not.
- **No claim without a record.** Every readiness gate is backed by a captured run
  (`.sycl/scripts/evidence.sh record build|integration|e2e …`) whose run id is stored on the gate.
  A gate marked `pass` with no record fails `evidence.sh verify` (check E13).
- **Waivers are explicit.** Anything stubbed, disabled, or scoped out (OptiX, DLSS, OpenXR, TensorRT,
  multi-GPU NCCL, an unported ufunc family) is recorded as a **waiver** with a reason and a blast
  radius. A silent stub is a defect; a recorded waiver is a scoped engagement.
- All builds/tests/workload runs go through `.sycl/scripts/run.sh` — never call `icpx`/`cmake`/`pip`
  directly.
- **Concurrent GPU work is forbidden.** Multiple agents must never run `run.sh build|test|profile|bench|exec`
  at the same time on the same GPU; `run.sh` serializes these automatically, and you must not bypass it.
- **Integration difficulty is never a reason to stop.** If a surface is genuinely a systems project
  (a JIT codegen backend, a memory manager), lower the *target level*, record the residual work
  package with an estimate, and finish everything below it — do not quietly leave the backend dead.

## Required reading
- `references/backend-readiness.md` — the L0–L4 ladder, the gate for each level, and the claim rules.
- `references/integration-archetypes.md` — pick the project's archetype; it names the surfaces,
  the order to do them in, and the traps that sank each one before.
- `references/device-runtime-layer.md` — CUDA runtime/driver → SYCL/USM mapping (device, context,
  stream, event, allocator, pool, multi-device), and the host library replacement table.
- `references/host-port-sweep.md` — compiling the project's **host** `.cc/.cpp` files with `icpx`:
  nvcc-isms, `__host__/__device__` leakage, CUDA types in public headers, the mechanical checklist.
- `references/e2e-validation.md` — how to prove the real workload is *correct*, not just alive
  (loss-curve diff, golden output, accuracy metric, tolerance selection, determinism traps).
- `sycl-reference` → `references/sycl-build-guide.md` (icpx/AOT flags, oneMKL linking) and
  `references/intel-gpu-software-repos.md` (how oneDNN / PyTorch XPU / vLLM-XPU structure a real
  Intel-GPU backend — extract patterns, never copy source).
- `.sycl/state/project.json` (`kind`, `build_system`, `e2e_cases`, `integration`) and
  `.sycl/state/integration/index.json`.

---

## Procedure

Integration runs in two phases of the workflow, and the split matters:

| Phase | What this skill does | Exit gate |
|---|---|---|
| **`scaffold`** (before `migrate`) | Plan the integration, then build the **walking skeleton**: backend switch, build wiring, device runtime layer, one trivial op reaching the GPU through the project's own path | **L1** + runtime smoke |
| **`integrate`** (after `migrate`) | Drive the remaining surfaces, register every migrated kernel, run the project's own tests, then the **real workload** vs a reference | **L2 → target level (usually L3)** |

Doing the skeleton *first* is the entire design. Every project in the prior programme that ended at
L0 had "wire it up" as the last task on the list, and it was never reached. A skeleton built before
migration means every kernel lands in a **live** backend, and the last mile is a short walk instead
of a second project.

### Phase `scaffold`

#### S1. Classify the archetype and pick the target level
Read `references/integration-archetypes.md` and choose one of:

| # | Archetype | Example shape | Typical reachable level |
|---|---|---|---|
| A | C/C++ application with a backend seam | single driver + per-kernel headers | **L4** — cheapest |
| B | C++ library, CMake + its own test suite | `libfoo.so` + gtest/pytest | **L3** |
| C | Framework custom-op extension | PyTorch/TF operators built by setuptools | **L3** |
| D | Python array / JIT library | `import pkg` dispatches to generated kernels | **L2–L3** |
| E | Runtime / JIT-codegen system | the project *generates* device code at runtime | **L1–L2** (codegen backend is a systems project) |
| F | Graph / parser-driven engine | JSON/graph model → layer factory → pipeline | **L2–L3** |

Then set `integration.target_level` — the level you commit to reaching — using the rule in
`backend-readiness.md` §"Choosing a target". If `config.integration.target_level` is set, it is
authoritative (user override). Record the archetype, the reason, and the target in
`.sycl/state/integration/readiness.json`.

#### S2. Enumerate the integration surfaces
Run the scanner, then read the sources it points at:
```bash
bash .sycl/skills/sycl-integration/scripts/scan_integration.sh .        # path varies by engine layout
```
It reports CUDA **runtime/driver API** call sites (`cudaMalloc`, `cudaMemcpy`, `cudaStream*`,
`cudaEvent*`, `cudaSetDevice`…), host library usage (Thrust/CUB/cuBLAS/cuFFT/cuRAND/NCCL/RMM),
build-system CUDA hooks, framework registration points (`TORCH_LIBRARY`, `REGISTER_OP`,
`DeviceAPI`, extension `setup.py`), NVIDIA-only dependencies (OptiX/DLSS/TensorRT/NVML/OpenXR), and
the count of **host** translation units that must compile with `icpx`.

Write one work item per surface into `.sycl/state/integration/index.json`. The seven canonical
surfaces (skip the ones the project does not have — say so, do not silently drop them):

| id | Surface | Question it answers |
|----|---------|---------------------|
| `build` | Build-system integration | Does the project's own build produce its own artifact with the backend on? |
| `runtime` | Device runtime layer | Who owns the device, queue, stream, event, sync? |
| `memory` | Allocator / memory manager | What replaced `cudaMalloc`/RMM/the framework caching allocator? |
| `dispatch` | Registration & dispatch | How does the project's call path *reach* a SYCL kernel? |
| `hostlib` | Host library call sites | What replaced Thrust/CUB/cuBLAS/cuFFT/cuRAND/NCCL on the host side? |
| `package` | Packaging / import path | Does `import pkg` / `find_package` / linking work for a user? |
| `nvonly` | NVIDIA-only feature paths | What is stubbed or waived, and what breaks as a result? |

Each item gets `status` (`pending|in-progress|done|waived|deferred`), an `effort` estimate, a
`blocks_level` (the lowest readiness level it blocks), and — when done — its `evidence`.

#### S3. Build the walking skeleton
The smallest change that makes the backend **real**. In order:

1. **Backend switch** — one place (a `backend.hpp` / CMake option / env or build flag) that selects
   CUDA or SYCL. Not scattered `#ifdef`s at call sites. Name it the way the project names its other
   backends (`FOO_ENABLE_SYCL`, `USE_XPU`, `kDGLSYCL`, `device_type='xpu'`).
2. **Device runtime layer** — implement the project's device abstraction over SYCL:
   device enumeration/selection, a context, a queue per "stream", events, synchronisation, USM
   allocation + free, H2D/D2H/D2D copies, and error translation. Full mapping in
   `references/device-runtime-layer.md`. This is the piece that is *always* missing at L0 and it is
   cheap to write early and expensive to retrofit.
3. **Build wiring** — extend the project's existing build with a SYCL path (an `icpx -fsycl` rule
   beside the `nvcc` rule, or a `CMAKE_CXX_COMPILER=icpx` + `-fsycl` target). The artifact produced
   must be the project's own, under its own name.
4. **One op through the real path** — pick the simplest migrated-or-trivial kernel (a copy, a fill,
   an axpy) and route it through the project's **actual** dispatch path end to end: project API →
   dispatcher → SYCL kernel → result checked on the host. If the project has no kernel yet, write a
   throwaway one; the point is the *path*, not the math.

#### S4. Prove L1 and the runtime smoke, then hand back to `migrate`
```bash
.sycl/scripts/evidence.sh record build backend --label scaffold-l1 \
    -- "<the project's own build command with the SYCL backend enabled>"
.sycl/scripts/evidence.sh record integration runtime --label scaffold-smoke \
    --reference "host-computed expected result" \
    -- "<the smoke command that drives one op through the project's dispatch path>"
```
Store both run ids on the corresponding gates in `readiness.json`, set `achieved_level: "L1"`, log,
regenerate PROGRESS.md, commit. If the skeleton cannot be built (archetype E, or a hard blocker),
record it as a surface with `status: deferred`, an estimate, and lower `target_level` **now** — so
migration proceeds knowing the truth, rather than discovering it at the end.

> **Scaffold is time-boxed by design.** If the skeleton is not standing after a reasonable effort,
> that is itself the finding: record the blocker, set `target_level: "L0"` with the reason, and go
> migrate. What is forbidden is proceeding with an *unstated* L0.

### Phase `integrate`

#### I1. Land every migrated kernel in the live backend
Every kernel from `migrate` must be reachable through the dispatch surface, not only through its
`dev/sycl/<k>` test binary. Walk `kernels/index.json` and, for each `migrated` kernel, confirm the
project's own call path reaches it (registration entry, dispatch table row, layer-factory case,
`torch.ops` binding). Anything unreachable is a defect in `dispatch`, not a kernel defect — record it
on the surface, fix it here.

#### I2. Complete the surfaces
Work the `index.json` items, lowest `blocks_level` first, one at a time, committing each:
- **`hostlib`** — replace host-side CUDA library calls; see the table in
  `references/device-runtime-layer.md` §"Host libraries".
- **`memory`** — replace the memory manager (RMM, caching allocator, framework pool) with a USM
  equivalent; keep the project's own allocator interface so call sites are unchanged.
- **`build`/host sweep** — compile the project's host translation units with `icpx`; the mechanical
  checklist is `references/host-port-sweep.md`. On large libraries this is broad but shallow — do it
  as one sweep, not per-kernel.
- **`package`** — the user-facing entry: `import pkg` succeeds, the extension builds via the
  project's own `setup.py`/`pip install -e .`, `find_package`/CMake export works.
- **`nvonly`** — stub or disable, and record a **waiver** for each with reason + blast radius.

After each surface: `run.sh` build + the project's own tests, then commit.

#### I3. Prove L2 — the project's own test suite runs on the backend
Not the migrated-kernel unit tests: the **project's** suite (`ctest`, `pytest`, `make test`,
`gtest`), run with the SYCL/XPU backend selected.
```bash
.sycl/scripts/evidence.sh record integration suite --label l2-suite \
    --reference "<project's own test suite, unmodified>" \
    -- "<the project's own test command, backend=sycl>"
```
A partial pass is a legitimate L2 **only** if the failures are enumerated in `readiness.json`
`gates.L2.exceptions` with a reason each. "Most tests pass" is not a gate.

#### I4. Prove L3 — the real workload, against a reference
This is the level that licenses the sentence "project X runs on Intel GPU". Run the project's real
entrypoint on a real workload to completion and compare against a reference per
`references/e2e-validation.md`. **Every** resolved e2e case is a gate (config
`benchmark.e2e_cases` if non-empty, else `project.json` `e2e_cases`).
```bash
.sycl/scripts/evidence.sh record e2e baseline --label baseline \
    --reference "<loss curve vs gpt2_124M_debug_state.bin | golden output | accuracy metric>" \
    --tolerance "<the project's own tolerance>" \
    -- "<the project's real entrypoint command>"
```
Store the run ids in `profile/e2e.baseline.json` and on `gates.L3`. On a correctness failure,
bisect to the responsible kernel and send **that kernel** back to `migrate` — do not loosen the
workload tolerance.

#### I5. Record readiness, waivers, and the residual work package
Compute `achieved_level` **from the gates that actually have records** — never assert it. Then write
the residual package: for every surface not `done`, an estimate and what it unlocks. This is what
turns "we did not finish" into a scoped, sellable next step, and it is what the report's integration
section is generated from. Format and effort rubric: `references/backend-readiness.md`
§"Residual work package".

Run `evidence.sh verify` before leaving the phase. Then `phase = profile-e2e`.

> **L4** is not earned in this phase — it is earned at `done`, when the post-optimization e2e run
> produces a measured baseline→final speedup on the real workload. The `done` phase sets it.

---

## What "done" looks like per level

| Level | The honest sentence you may write |
|---|---|
| L0 | "The N kernels are migrated and validated on Intel GPU by standalone tests." |
| L1 | "…and the project builds with a SYCL backend." |
| L2 | "…and the project's own test suite passes on Intel GPU." |
| L3 | "**<project> runs on Intel GPU**, producing correct results on <workload>." |
| L4 | "…and is X.X× faster end-to-end than the initial SYCL baseline." |

Nothing above the achieved level may appear in the report, the summary, or a commit message.

## Anti-patterns (each one was observed, each one cost a project)
- **Standalone tree declared as integration.** `dev/sycl/` builds, 20/20 pass, entrypoint never run.
  → L0. Say L0.
- **Infrastructure without the sweep.** Device-type enum, CMake module and dispatch macros committed;
  190 host files never compiled with `icpx`. Infrastructure is not integration.
- **Bypassing the production path.** Manual fprop/bprop loop instead of the parser-driven pipeline;
  the objects are real, the path is not the one users take. → L1/L2, not L3.
- **Correctness by liveness.** The workload ran without crashing and no reference was compared.
  Running is not correct — L3 needs a reference (`e2e-validation.md`).
- **Silent stubs.** An NVIDIA-only path quietly disabled and never mentioned. Record a waiver.
- **Regex-translating a long tail.** A best-effort source translator with no test exercising a single
  case of the long tail. Coverage is then *unknown*, and unknown must be reported as unknown.
- **Re-cutting the surface list to hit 100%.** If a surface is dropped, mark it `waived`/`deferred`
  with its original id — never delete the row.
