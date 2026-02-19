---
description: Apply optimization to a SYCL kernel (runs as subtask)
subtask: true
---

Optimize the SYCL kernel: $ARGUMENTS

Steps:
1. Read syclgen/hw_spec_b60.md for target hardware specs
2. Read syclgen/optimization_guide.md for optimization techniques
3. Read the kernel optimization spec from kernel_optimization_spec.json
4. Apply the specified optimization to the kernel
5. Ensure correctness is preserved — output must match original
6. Add comments explaining the optimization
7. Report what was changed and expected improvement
