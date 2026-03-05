---
description: Implement a SYCL kernel migration (runs as subtask)
subtask: true
---

Implement SYCL kernel migration for: $ARGUMENTS

Steps:
1. Read the source CUDA kernel file carefully
2. Understand all CUDA-specific constructs used (threadIdx, blockIdx, shared memory, etc.)
3. Create the equivalent SYCL implementation using appropriate SYCL constructs
4. Handle memory management (buffers/accessors or USM)
5. Ensure thread indexing is correctly mapped (CUDA grid → SYCL nd_range)
6. Add proper error handling and queue management
7. Write the SYCL file to the target location specified
8. Update any build system files (CMakeLists.txt, Makefile) if needed

Report back: the file path created, key CUDA→SYCL mappings made, and any concerns.
