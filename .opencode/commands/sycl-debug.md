---
description: Debug a failing SYCL kernel (runs as subtask)
subtask: true
---

Debug the SYCL kernel: $ARGUMENTS

Steps:
1. Read the failing test output and error messages
2. Read the SYCL source file and the original CUDA kernel
3. Compare the implementations line by line
4. Identify the bug (common issues: wrong indexing, missing synchronization, incorrect memory scope)
5. Fix the SYCL code
6. Re-run the test: ./syclgen/remote_run.sh test $1
7. Report: what was wrong, what was fixed, test result after fix
