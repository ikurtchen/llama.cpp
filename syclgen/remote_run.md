---
description: Run tests on remote ssh server
agent: build
subtask: true
---

Run tests on a remote SSH server using script: `syclgen/remote_run.sh`.

$1 is the remote server name
$2 is the task to run
$3 is the other arguments to pass to the script

**IMPORTANT NOTES**:
* For server name (`-s`), use `h20` for **Nvidia GPU** server to test CUDA, `b60` for **Intel GPU** server to test SYCL.
* When upload code to **Intel GPU**, the remote directory (`--remote`) is "/intel/kurt".
* When run tests on **Intel GPU**, always add `-e ZES_ENABLE_SYSMAN=1` to support to get free memory of GPU by `sycl::aspect::ext_intel_free_memory`.
* When upload code to **Nvidia GPU**, the remote directory (`--remote`) is "/ssd/kurt".
* When run tests on **Nvidia GPU**, always add `-e CUDA_VISIBLE_DEVICES="0"` to avoid using all GPUs.

Examples:

* Upload code to Nvidia GPU server `h20` and build:
```bash
bash syclgen/remote_run.sh -s h20 -t upload,build --local "/localdisk/kurt/workspace/code/xpu/llama.cpp" --remote "/ssh/kurt/"
```

* Run kernel unit test on Nvidia GPU server `h20`:
```bash
bash syclgen/remote_run.sh -s h20 -t unit -k CPY,ROPE -e CUDA_VISIBLE_DEVICES="0"
```

* Run integration test on Nvidia GPU server `h20`:
```bash
bash syclgen/remote_run.sh -s h20 -t run -e CUDA_VISIBLE_DEVICES="0"
```

* Upload code to Intel GPU server `b60` and build:
```bash
bash syclgen/remote_run.sh -s b60 -t upload,build --local "/localdisk/kurt/workspace/code/xpu/llama.cpp" --remote "/intel/kurt/"
```

* Run kernel unit test on Intel GPU server `b60`: 
```bash
bash syclgen/remote_run.sh -s b60 -t unit -k CPY,ROPE -e ZES_ENABLE_SYSMAN=1
```

* Run integration test on Intel GPU server `b60`:
```bash
bash syclgen/remote_run.sh -s b60 -t run -e ZES_ENABLE_SYSMAN=1
```

* Run custom command on Intel GPU server `b60`: assume $3 is `--command "cat /intel/kurt/llama.cpp/logs/b60_build_20260202_133422.log | tail -100"`
```bash
bash syclgen/remote_run.sh -s b60 -t custom --command "cat /intel/kurt/llama.cpp/logs/b60_build_20260202_133422.log | tail -100"
```