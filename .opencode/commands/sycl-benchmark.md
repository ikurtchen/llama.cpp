---
description: Benchmark SYCL kernels via remote_run.sh (runs as subtask)
subtask: true
---

Benchmark SYCL kernels: $ARGUMENTS

Steps:
1. Read syclgen/remote_run.md for test infrastructure documentation
2. Upload code to server: ./syclgen/remote_run.sh -s b60 -t upload
3. Build: ./syclgen/remote_run.sh -s b60 -t build -e ZES_ENABLE_SYSMAN=1
4. Run benchmarks: ./syclgen/remote_run.sh -s b60 -t benchmark -e ZES_ENABLE_SYSMAN=1
5. Report the FULL output with timing numbers
