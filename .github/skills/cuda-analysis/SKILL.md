---
name: cuda-analysis
description: >
  Enumerate CUDA GPU kernels in a project and extract each kernel's ALGORITHM (intent, not
  syntax), the CUDA-specific constructs it uses, its input/output spec, and a migration risk
  rating. Use when: scanning a CUDA codebase for kernels, building the migration inventory,
  identifying warp shuffles / shared memory / atomics / cooperative groups / CUB / cuBLAS /
  cuDNN / Thrust usage, or deciding which kernels are hardest to port. Produces the kernel
  index and per-kernel detail files that drive the rest of the migration.
---

# CUDA Analysis

Turn a CUDA codebase into a structured migration inventory. You extract *what each kernel computes*
and *what CUDA-specific machinery it relies on* — you do **not** write SYCL here (that is
`sycl-migration`).

## When to use
- Inventory phase: discover every CUDA kernel and record it in `.sycl/state/kernels/`.
- Any time you need to understand a kernel's algorithm and CUDA constructs before porting.

## Procedure

### 1. Discover kernels
Run the scan helper (adjust the root as needed):
```bash
bash <skill-path>/scripts/scan_cuda.sh <project-root>
```
It finds `.cu`/`.cuh` files, `__global__`/`__device__` definitions, `<<<...>>>` launches, and library
usage (CUB, cuBLAS, cuDNN, Thrust, cuFFT, cuSPARSE). Treat its output as a starting list; confirm by
reading the sources — generated launchers or templates can hide kernels.

### 2. Extract the algorithm (intent, not syntax)
For each kernel, read the source and write a short **algorithm summary**: what it computes, the data
layout, the parallelization strategy (per-element, per-row, tiled, reduction, scan, stencil, GEMM…),
and any numerical subtleties (accumulation order, tolerances, fast-math). Understanding the algorithm
is what enables a correct hand-written SYCL port later.

### 3. Flag CUDA-specific constructs
Record which of these each kernel uses (they drive risk and the SYCL mapping — see the
`sycl-reference` skill's `sycl-kernel-patterns.md`):
- **Warp-level**: `__shfl_*`, `__ballot_sync`, `__any/all_sync`, cooperative groups, warp size 32.
- **Shared memory**: `__shared__`, dynamic shared memory, bank-conflict-sensitive access.
- **Atomics**: `atomicAdd/CAS/...`, atomic reductions.
- **Memory**: `__constant__`, textures/surfaces, `cudaMallocManaged`, pinned memory.
- **Libraries**: cuBLAS→oneMKL, cuDNN→oneDNN, Thrust→oneDPL, CUB→group algorithms, cuFFT→oneMKL DFT.
- **Math**: intrinsics (`__expf`, `__fdividef`), fast-math assumptions.
- **Launch**: grid/block dims, dynamic parallelism, streams/graphs, multi-GPU.

### 4. Capture the IO spec
Record inputs/outputs: names, dtypes, shapes/sizes, and which buffers are read vs written. This is
what the unit test and the accuracy reference will be built against.

### 5. Rate risk
Risk measures **migration difficulty / how much care the port needs** — it ranks migration order and
signals how rigorous the correctness reference must be. It is **NOT** an eligibility filter: a `high`
risk rating never means "skip this kernel." Every non-skipped kernel gets migrated regardless of risk;
high-risk kernels just go later (lowest risk first) and demand a more careful reference.
- **low** — element-wise / simple map; no warp/shared/atomics; direct nd_range mapping.
- **medium** — reductions, shared-memory tiling, atomics, standard library calls.
- **high** — warp-shuffle algorithms, cooperative groups, custom quantized GEMM, dynamic parallelism,
  texture/surface use, or anything without a clear reference for correctness. Still migrated — with
  extra care and a solid oracle. If (and only if) no trustworthy reference can be established, mark it
  `needs-reference` and surface it; never silently skip a kernel because it looks hard.

### 6. Write state
- Append each kernel to `.sycl/state/kernels/index.json` (`id`, `source`, `status: "pending"`,
  `risk`, `detail` pointer). Validate against `.sycl/schemas/kernel-index.schema.json`.
- Create `.sycl/state/kernels/<id>.json` (validate against `kernel.schema.json`) with the algorithm
  summary, `cuda_constructs`, `io_spec`, and initial `reference.kind` guess.
- Regenerate the dashboard: `bash .sycl/scripts/gen-progress.sh`.
- Log: `.sycl/scripts/log.sh sycl-agent inventory info "inventoried N kernels"`.

## Output (return to orchestrator)
- Kernel count by risk (low/medium/high) and any kernels lacking a clear correctness reference.
- Recommended migration order (lowest risk first). **Do not propose skips based on risk/difficulty** —
  the default is to migrate every kernel. Only flag a kernel as a `skipped` candidate for a concrete
  non-difficulty reason (dead/unreachable code, not actually a GPU kernel, or an explicit user
  exclusion), and state that reason. A kernel with no correctness reference is `needs-reference`
  (surfaced), not `skipped`.

## Assets
- `scripts/scan_cuda.sh` — grep-based discovery of kernels and CUDA library usage.
