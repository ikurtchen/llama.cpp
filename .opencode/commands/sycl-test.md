---
description: Test a SYCL kernel via remote_run.sh (runs as subtask)
subtask: true
---

Run the SYCL kernel test for: $ARGUMENTS

Steps:
1. Read syclgen/remote_run.md to understand the test infrastructure
2. Run the test command: ./syclgen/remote_run.sh test $1
3. Capture and analyze the output
4. If the test fails, identify the root cause from error messages
5. Report: PASS/FAIL, output summary, and any error analysis
