# Project instructions (human-authored overrides)

This file is **optional** and **human-owned**. `sycl-agent` reads it on start and treats it as
project-specific guidance that **takes precedence over skill defaults** where they conflict. The
agent never rewrites this file — edit it by hand to steer the workflow for a special project.

Delete the examples below and keep only what applies. Leave the file empty (or delete it) to use
pure defaults.

## Guardrails (what overrides cannot do)
Custom instructions may tighten, relax, or redirect *how* work is done, but they must **not** disable
the framework's integrity guarantees: correctness is always gated against a reference, one change is
verified at a time, and every accepted change is committed and logged. If an instruction here would
break those, the agent surfaces the conflict instead of silently obeying.

## Examples of what to put here

### Build / test commands
<!-- Exact commands if the project's build is non-standard; the agent uses these verbatim via run.sh.
     Default: SYCL is a backend that mirrors the CUDA structure and extends the existing build
     (e.g. a SYCL rule beside the nvcc/CMake one) rather than a standalone sycl/ project. -->
- Build: `make -C dev/sycl <kernel>` (SYCL rule sits beside the `dev/cuda` `nvcc` rule)
- Test:  `./dev/sycl/<kernel>` (same args / version-select as the CUDA harness)

### Correctness reference
<!-- Point the SYCL tests at an existing CPU reference / harness, or set project-wide tolerances. -->
- Reuse `tests/cpu_reference.hpp::reference_<kernel>()` as the oracle; do not author new references.
- Tolerance: fp32 rtol=1e-4 atol=1e-5 (this workload accumulates large sums).

### Kernels
<!-- Force-skip, force-priority, or annotate specific kernels. -->
- Skip `legacy_debug_kernel` (dead code, do not migrate).
- Always optimize `attention_fwd` first regardless of e2e ranking (latency-critical path).

### Target / platform
<!-- Cross-compile intent, or notes the detect phase should respect. -->
- Cross-compiling for b70 from a b60 runner: set `target.on_mismatch: warn`.

### Conventions / constraints
<!-- Coding conventions, forbidden APIs, licensing notes, domain rules. -->
- Mirror the CUDA layout: SYCL kernel headers sit beside their `.cuh` counterparts with the same
  launcher signatures; share the host/driver code, don't fork a separate SYCL `main`.
- No `sycl::half` in the public API — convert at the boundary.
