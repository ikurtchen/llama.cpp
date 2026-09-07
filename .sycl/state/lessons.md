
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

## run.sh: touch + build must be ONE invocation, not two separate run.sh calls

`run.sh` performs an automatic rsync push (local->remote) before *every* invocation, including
`build`. If you `touch` a remote file in one `run.sh exec` call and then run a build in a *separate*
`run.sh build` call afterward, that second call's own pre-command sync re-pushes the local file and
resets its remote mtime back to the (older, sandbox-clock) local mtime -- silently undoing the touch
and causing the build to skip recompilation even though the file content genuinely changed. Symptom
is identical to the earlier "clock skew" lesson (code change appears to have zero effect) but the
root cause here is the sync itself, not clock drift. Fix: always combine `touch <file> && cmake
--build ...` into a **single** `run.sh build "..."` call, so only one sync happens before the
touch-then-build sequence runs atomically on the remote side.

## optimize/flash-attn-vec: GQA-decode FLASH_ATTN_EXT is occupancy-bound, not compute/memory-bound

For decode shapes (nb=1, few Q rows -- e.g. nh=8 heads with nr23=[1,1] launches only 8 work-groups
total), the existing one-work-group-per-Q-row online-softmax kernel cannot fill a B70's execution
units: `unitrace --metric-query` showed GPU_BUSY ~100% but XVE_ACTIVE only ~2.66% and
XVE_THREADS_OCCUPANCY_ALL only ~3.49%. This is the same problem CUDA's `fattn-vec.cuh` solves with
`parallel_blocks`/flash-decoding. A first hypothesis (removing a redundant barrier per KV iteration,
since `reduce_over_group`'s result is already uniform across work-items) measured *zero* wall-clock
change -- confirming the kernel is occupancy-bound, not barrier-bound, before spending more effort on
barrier micro-optimizations. The fix that worked: split the KV range into `nsplit` chunks per Q row
(launching `nrows*nsplit` work-groups instead of `nrows`), write each split's partial online-softmax
state (m, l, unnormalized acc) to a scratch buffer, then merge with a small second "combine" kernel
(one work-group per row, redundant per-thread loop over the tiny nsplit values -- cheaper than a
barrier-based reduction at this size). Speedup scales directly with nsplit: 15.02x/7.56x/3.85x for
nsplit=16/8/4. Falls back to the original single-pass kernel when nrows is already large enough
(prefill), confirmed unaffected. General takeaway: when a kernel-benchmark measures near-zero
GFLOPS/GB-s AND unitrace shows GPU_BUSY high but XVE_ACTIVE low, check the work-group *count* against
the launch shape before assuming a memory/compute bottleneck -- it may just be too few work-groups
for the shape at hand, especially for GQA-decode-style small-batch kernels.

## optimize/mmvq: a fast-path routing threshold on a shape dimension needs a semantic check, not just a bound

The mmvq fused-kernel fast path was gated on `ne11 <= 8` (RHS column count), intended to mean "this
is decode, or a handful of speculative-decode/beam candidates". But `ne11` is just a tensor dimension
-- it cannot distinguish "up to 8 decode candidates" from "a 6-token prompt prefill", which produces
the identical shape. Every isolated `test-backend-ops perf` benchmark used to validate the trial
(m=4096,k=14336 at n=1..8,512) showed the fused kernel winning or tying, because none of those shapes
represent a *real* short-prompt prefill matmul. Only a genuine end-to-end `llama-cli` run (6-token
prompt) exposed a ~61% prompt-throughput regression, caught during the `done`-phase e2e
re-verification -- well after the trial had already been marked "kept" and the phase gated closed.

Fix: tightened the threshold to `ne11 == 1`, the only value that is *unambiguous* (a batch dimension
can never be a multi-token prefill). General takeaway: when a fast path is gated by comparing a shape
dimension against a bound, ask whether that dimension's *value* alone actually disambiguates the two
cases the fast/slow paths are meant to separate. If it doesn't (as here, batch-size vs sequence-length
both show up as "small ne11"), no set of single-shape microbenchmarks will catch the misrouting --
only a real, whole-model e2e run exercises the call patterns where the ambiguity actually resolves the
wrong way. This is why the `done` phase's e2e re-verification is a hard gate, not a formality: it is
the only check that can catch this entire class of bug.

## optimize/mmvq: an isolated single-shape win does not generalize to the deployed model's dims -- and a stale remote build can make a real regression look fixed

Follow-up to the lesson above. The `ne11==1` threshold fix above did stop the prefill misrouting, but
a second, independent problem was still hiding underneath it: the fused mmvq kernel's original 2.58x
win (trial #1) was measured only at `m=4096,k=14336` -- the dims of a *different, larger* model than
the one actually being deployed (Qwen3-0.6B, `n_embd=1024, n_ff=3072`). Once the fix was in place and
n=1 unambiguously meant "single-token decode", a real `llama-cli` e2e run still showed a genuine
Generation regression (72.0 -> 66.6 t/s), even though every single-token benchmark at the *original*
validation shape still looked fine. Adding an n=1 `test-backend-ops` case at the real model's dims
showed why: the fused kernel loses to oneMKL's dequant+GEMM path in 3 of 4 realistic (m,k)
combinations -- the win never generalized past the shape it was born from. The fused fast path was
reverted entirely; oneMKL GEMM is used unconditionally for all n. General takeaway: validate a kept
optimization at the *actual deployed model's* dims before trusting it, not just at whatever shape the
kernel-level benchmark happened to use -- a real speedup on one model's matmul dims can be a real
regression on another's, for the exact same op and the exact same code path.

A second, orthogonal trap compounded this investigation: an early re-measurement after the threshold
fix reported Generation 77.7 t/s (an apparent *improvement*, taken as confirmation the fix worked) --
but this was a stale-binary artifact. The remote host's incremental `cmake --build` can silently treat
an already-compiled `.o` as up to date and skip recompilation even after a source edit is synced, if
the synced file's mtime lands behind the existing `.o`'s mtime (confirmed via an identical relinked-
binary MD5 across a `rm -f` + rebuild cycle with no source change). This produced a confident-looking
but entirely wrong "problem solved" measurement, and cost significant time to unravel. General
takeaway: on this remote runner, always `touch` the changed source file(s) immediately before a
`cmake --build`, and confirm the build log shows `Building CXX object ...` (not just `Built target
...`) for every file you just edited -- before trusting *any* measurement taken after a code change.
Never conclude a performance investigation from a build you did not personally watch recompile the
changed file.
