---
name: triton-analysis
description: >
  Enumerate Triton GPU kernels (@triton.jit) in a project and extract each kernel's ALGORITHM
  (intent, not syntax), the Triton-specific constructs it uses (block pointers, masks, tl.dot,
  reductions, atomics, autotune configs, constexpr meta-params, num_warps/num_stages), its
  input/output spec (including the launch grid + meta-params), and a migration risk rating. Use
  when: scanning a Triton codebase for kernels, building the migration inventory for Triton
  sources, identifying tl.load/tl.store masking / tl.dot / tl.atomic_* / autotune usage, or
  deciding which Triton kernels are hardest to port. Produces the kernel index and per-kernel
  detail files (source_lang=triton) that drive the rest of the migration.
---

# Triton Analysis

Turn a Triton codebase into a structured migration inventory. You extract *what each kernel computes*
and *what Triton-specific machinery it relies on* — you do **not** write SYCL here (that is
`sycl-migration`). This is the Triton counterpart of `cuda-analysis`; it writes the **same** kernel
index/detail files, tagged `source_lang: "triton"`.

## When to use
- Inventory phase when the project has Triton kernels (`@triton.jit`, `import triton`,
  `triton.language as tl`). A project may have both CUDA and Triton — run this alongside
  `cuda-analysis` and merge into one index.
- Any time you need a Triton kernel's algorithm + constructs before porting.

## Background
Load `sycl-reference` → `references/triton-patterns.md` for the Triton→SYCL mental model (one Triton
program == one SYCL work-group; block pointers/masks == index math + bounds guards; `tl.dot` == XMX).
It informs how you classify constructs and risk here.

## Procedure

### 1. Discover kernels
Run the scan helper (adjust the root as needed):
```bash
bash <skill-path>/scripts/scan_triton.sh <project-root>
```
It finds `@triton.jit` functions, `tl.*` usage, `@triton.autotune`/`triton.Config`, and kernel
launches (`kernel[grid](...)`). Treat its output as a starting list; confirm by reading the sources —
kernels are often wrapped by an `autograd.Function` or a plain Python launcher that builds the grid.

### 2. Extract the algorithm (intent, not syntax)
For each kernel, read the source and write a short **algorithm summary**: what it computes, the
**tile shape** (from `BLOCK_*: tl.constexpr`), the data layout + strides (from `tl.make_block_ptr` or
manual `offs` math), the parallelization strategy (per-element, per-row, tiled GEMM, reduction/
softmax, attention, scan), and numerical subtleties (accumulate dtype via `tl.dot(out_dtype=...)`,
fast-math, masking fill values). Recovering the tiling intent is what enables a correct SYCL port.

### 3. Flag Triton-specific constructs
Record which of these each kernel uses (they drive risk and the SYCL mapping — see
`triton-patterns.md`):
- **Program/tile model**: `tl.program_id`, `tl.num_programs`, `BLOCK_*: tl.constexpr`, `tl.arange`.
- **Addressing**: `tl.make_block_ptr` / `tl.advance`, manual `offs` pointer math, `boundary_check`.
- **Masked memory**: `tl.load(..., mask=, other=)`, `tl.store(..., mask=)`, `eviction_policy`.
- **Matmul**: `tl.dot` (→ XMX/DPAS; flag `out_dtype`, input dtypes — GEMM/attention candidates).
- **Reductions/scans**: `tl.sum/max/min(axis=)`, `tl.cumsum`, softmax idioms.
- **Atomics**: `tl.atomic_add/max/min/cas`.
- **Math/dtype**: `tl.exp/log/sqrt/sigmoid`, `tl.math.*` fast paths, `tl.float16`/`tl.bfloat16`.
- **Autotune/launch**: `@triton.autotune` configs, `num_warps`, `num_stages`, the `grid` lambda.

### 4. Capture the IO spec
Record inputs/outputs: names, dtypes, shapes/sizes, strides, and which pointers are read vs written.
Also capture the **launch spec**: the `grid` expression and the `tl.constexpr` meta-parameters (tile
sizes, `num_warps`, `num_stages`) — these define the tiling the SYCL port must reproduce. If the
project has a **PyTorch reference** (eager implementation + `torch.testing.assert_close` test), record
its location: it is the ready-made correctness oracle (see `sycl-migration`).

### 5. Rate risk
Risk measures **migration difficulty / how much care the port needs** — it ranks migration order and
signals how rigorous the correctness reference must be. It is **NOT** an eligibility filter: a `high`
risk rating never means "skip this kernel." Every non-skipped kernel gets migrated regardless of risk;
high-risk kernels just go later (lowest risk first) and demand a more careful reference.
- **low** — element-wise / simple map over a 1-D block; masks only; no `tl.dot`, no cross-lane reduce.
- **medium** — block reductions / softmax, atomics, single `tl.dot` without a fused epilogue, standard
  autotuned tile kernels.
- **high** — tiled GEMM/attention with a K-loop of `tl.dot` + epilogue + `num_stages` pipelining
  (**sycl-tla candidates** — flag them), custom quantization/MoE, anything without a clear reference.
  Still migrated — with extra care and a solid oracle. If (and only if) no trustworthy reference can be
  established, mark it `needs-reference` and surface it; never silently skip a kernel because it looks
  hard.

### 6. Write state
- Append each kernel to `.sycl/state/kernels/index.json` (`id`, `source`, `status: "pending"`,
  `source_lang: "triton"`, `risk`, `detail` pointer). Validate against
  `.sycl/schemas/kernel-index.schema.json`.
- Create `.sycl/state/kernels/<id>.json` (validate against `kernel.schema.json`) with the algorithm
  summary, `source_lang: "triton"`, `triton_constructs`, `io_spec` (incl. launch/meta), an initial
  `reference.kind` guess (prefer the project's torch reference), and — for tiled `tl.dot` kernels — a
  `target_style` hint (`sycl-tla` candidate) with a one-line reason. Leave the final target-style
  decision to `sycl-migration`.
- Regenerate the dashboard: `bash .sycl/scripts/gen-progress.sh`.
- Log: `.sycl/scripts/log.sh sycl-agent inventory info "inventoried N triton kernels"`.

## Output (return to orchestrator)
- Kernel count by risk (low/medium/high), which are `tl.dot`/tiled (**sycl-tla candidates**), and any
  lacking a clear correctness reference.
- Recommended migration order (lowest risk first). **Do not propose skips based on risk/difficulty** —
  the default is to migrate every kernel. Only flag a kernel as a `skipped` candidate for a concrete
  non-difficulty reason (dead/unreachable code, not actually a GPU kernel, or an explicit user
  exclusion), and state that reason. A kernel with no correctness reference is `needs-reference`
  (surfaced), not `skipped`.

## Assets
- `scripts/scan_triton.sh` — grep-based discovery of `@triton.jit` kernels, `tl.*` usage, autotune
  configs, and launches.
