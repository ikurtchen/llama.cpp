
## profile-e2e: unitrace crashes deterministically on this runner during model load

`unitrace` (overview mode AND slice `--start-paused` mode, both with and without `--call-logging`)
segfaults during `llama-cli`'s model-loading phase on this Intel B70 runner, regardless of workload
size -- reproduced on both Qwen3-8B-Q8_0 and Qwen3-0.6B-Q8_0. This is not the documented "large
workload OOMs unitrace" failure mode; it crashes even on the smallest model at the very first
spinner frame. Root cause not further isolated (out of scope for this migration); treat unitrace as
currently unusable for e2e workload tracing on this host.

**Fallback used** (per sycl-profiler skill SS5, "in-app device timers"): added a lightweight,
`GGML_SYCL_PROFILE=1`-gated per-op timer directly in `ggml_backend_sycl_graph_compute`
(ggml/src/ggml-sycl/ggml-sycl.cpp) that brackets each op's dispatch with `ctx->stream().wait()` and
accumulates wall-clock by `ggml_op_desc()`, dumped cumulatively at the end of every `graph_compute`
call (so the last dump in the log is the full-run total). Confirmed a clean 8456/8456 regression
after adding it (profiling is off by default; zero overhead when the env var is unset).

**Gotcha 1 -- two loaded instances of ggml-sycl's statics.** An `atexit()`-registered dump saw an
always-empty table even though the *same-named* static was clearly being populated (confirmed via
temporary debug prints showing `table_size` growing during the run). ggml's dynamic backend loading
apparently creates a second loaded copy of `libggml-sycl.so`'s file-scope statics distinct from the
one `graph_compute` runs in, so `atexit()` fired against the *other* instance's empty copy. Fix:
never rely on `atexit()` for this kind of instrumentation in ggml-sycl; dump from the exact same call
site that recorded the data (here: unconditionally at the end of `graph_compute` itself).

**Gotcha 2 -- clock skew between the local dev sandbox and the remote runner silently skips
recompiles.** After editing `ggml-sycl.cpp` and pushing via `run.sh build`, the object file was
*newer* than the just-edited source (remote wall clock was hours ahead of the local sandbox clock
at push time), so `make`/`ninja` saw the source as "not modified" and skipped recompilation entirely
-- the stale binary kept running with no error. Symptom: a code change appears to have zero effect.
Fix: `touch` the source file on the remote host (or otherwise force its mtime past the existing
object's) immediately before every build when local/remote clocks might disagree.

## optimize/mmvq: unitrace segfaults specifically on oneMKL-calling kernels

Building on the earlier unitrace finding (segfaults during e2e model load): the crash reproduces on
a **single-kernel** `test-backend-ops perf` driver too, but only for the kernel that calls into
oneMKL's GEMM dispatch (the old dequant+GEMM MUL_MAT q8_0 path). The exact same driver, same
`--metric-query` flags, profiling the new oneMKL-free fused kernel instead, completes cleanly with
real HW counters. This narrows the earlier "unitrace crashes on this runner" finding to something
that fires specifically when the profiled binary calls oneMKL -- worth checking first if unitrace
crashes on a new kernel: does it call oneMKL/oneDNN?

## optimize/mmvq: mul_mat q8_0 decode fast path (fused mat-vec kernel)

For Q8_0 MUL_MAT with a small number of RHS columns (n<=8, i.e. decode/n=1 and speculative-decode-
sized batches), the existing dequantize-to-F32-scratch + oneMKL GEMM path forces ~2x(ne00*ne01)
bytes of pure F32 scratch traffic (write the dequantized matrix, then read it back for the GEMM) for
a single output vector -- e.g. ~470 MB moved for m=4096,k=14336. Replacing it with a fused kernel
that reads the quantized `block_q8_0` bytes directly and accumulates the dot product in registers
(one work-group per output row, `sycl::reduce_over_group` for the final reduction) cuts this to
O(ne00*ne01) quantized bytes only, no scratch buffer, no extra kernel launch, no GEMM call.
Measured: 2.58x on the n=1 shape (1140us -> 442us), unaffected n=512/prefill shape (unchanged code
path). Deep-profile of the fused kernel (unitrace --metric-query, VectorEngineStalls group) shows
XVE_STALL ~87% dominated by SBID (~77%) and SendWr (~54%) stalls despite only ~131+41 GB/s achieved
(well under B70's ~456 GB/s peak) -- i.e. still latency-bound on scattered per-lane scalar loads
(`blk.qs[l]`, `yv[l]`), not bandwidth-saturated. A vectorized-load rewrite is a plausible next step
(residual work), not pursued here due to time-boxing.
