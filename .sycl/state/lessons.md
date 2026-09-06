
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
