# AGENTS.md Override Permissions

**Created:** 2026-02-23
**Authorized by:** User

## Override Scope

This document overrides the restrictions in `/localdisk/kurt/workspace/code/xpu/llama.cpp/AGENTS.md` for the SYCL kernel optimization pipeline.

## Permissions Granted

The AI agent is authorized to:

1. **Write new code** — Implement new features and optimizations
2. **Generate large code blocks** — Create templates, refactored code, and new implementations
3. **Modify kernel implementations** — Make algorithmic changes to SYCL kernels
4. **Implement new features** — Add missing quantization types, kernels, and optimizations

## Tasks Unblocked

| Task ID | Description | Original Block Reason |
|---------|-------------|----------------------|
| task_014 | XMX/joint_matrix compute path for mmq | oneAPI 2025.3 API incompatibility |
| task_039 | Consolidate mmvq 31 duplicated functions | AGENTS.md - large refactoring |
| task_040 | Consolidate dmmv duplicated K-quant code | AGENTS.md - large refactoring |
| task_041 | Add missing quant type support | AGENTS.md - feature addition |
| task_042 | Add missing feature gaps | AGENTS.md - feature addition |

## Scope Limitation

This override applies ONLY to:
- Files within `ggml/src/ggml-sycl/`
- The optimization tasks defined in `syclgen/kernel_optimization_task_spec.json`

All other AGENTS.md restrictions remain in effect for non-SYCL code.
