#!/usr/bin/env bash
# End-to-end WORKLOAD profiling (phase: profile-e2e) — understand the kernels in the full workload
# and RANK them by their share of wall-clock, so sycl-optimization knows which kernels to fix first.
# This is the complement of profile_metrics.sh (which zooms into ONE kernel): here we look at the whole
# app and attribute time across kernels, capturing BOTH host overhead and device (kernel) time.
#
# Tracing a large workload start-to-finish routinely OOMs / core-dumps unitrace, so the safe way to
# profile it is to collect only a SLICE of time — e.g. one Transformer layer or one model iteration —
# using unitrace sessions: start the app with collection paused (--start-paused --session <name>),
# then bracket the region of interest with --resume / --pause / --stop from a second shell. The slice
# is small enough not to crash unitrace yet representative enough to rank the kernels.
#
# Two capture strategies (pick per workload):
#   overview  — trace the whole run in one shot. Only for SMALL / short workloads (a handful of
#               iterations); will crash on big ones. Scope it with INCLUDE_KERNELS when possible.
#   slice     — start the app under --start-paused --session <name> and collect only the bracketed
#               window. Crash-safe; the recommended path for real workloads.
#
# Both strategies collect --host-timing (host API overhead) AND --device-timing / --device-timeline
# (per-kernel device time) so the resulting summary shows host vs device and a per-kernel ranking.
#
# Deterministic tooling, exactly like profile_metrics.sh: the unitrace binary is resolved from
# $UNITRACE_HOME (exported by run.sh from .sycl/config.json .tools.unitrace_home) using the same
# installed-vs-source layout — no reliance on PATH.
#
# Usage (run on the Intel GPU runner via .sycl/scripts/run.sh):
#
#   # 1) Small workload — one-shot overview:
#   .sycl/scripts/run.sh profile \
#     "bash framework/skills/sycl-profiler/scripts/profile_e2e.sh overview ./app --iters 3"
#
#   # 2) Real workload — crash-safe time slice. Start the app PAUSED in the background, let it warm
#   #    up to the region of interest, resume for one layer/iteration, then stop:
#   .sycl/scripts/run.sh exec \
#     "bash framework/skills/sycl-profiler/scripts/profile_e2e.sh slice mysess ./app" &   # (async)
#   #    ...wait for the app to reach steady state (a fixed sleep or a log marker)...
#   .sycl/scripts/run.sh exec "bash framework/.../profile_e2e.sh resume mysess"           # start collecting
#   #    ...one Transformer layer / one iteration worth of wall-clock...
#   .sycl/scripts/run.sh exec "bash framework/.../profile_e2e.sh stop   mysess"           # stop + flush
#
# The output text file carries "=== Host Timing ===" and "=== Device Timing ===" summaries; parse the
# device-timing per-kernel totals into .sycl/state/profile/e2e.json (schema e2e-profile.schema.json)
# with `ranking` sorted by descending wallclock_pct — that ranking is the optimization order.
#
# Env knobs (all optional; run.sh exports UNITRACE_HOME / UNITRACE_USE_INSTALLED):
#   UNITRACE_HOME          root of the unitrace tree (required to resolve the binary)
#   UNITRACE_USE_INSTALLED true=installed layout (bin/), false=source tree (build/)
#   INCLUDE_KERNELS        comma-separated substrings — collect ONLY these kernels (shrinks the trace,
#                          cuts crash risk; e.g. INCLUDE_KERNELS=attention,rmsnorm)
#   EXCLUDE_KERNELS        comma-separated substrings — drop these kernels from collection
#   OUT_DIR                output dir (default .sycl/reports/e2e_<ts>)
set -uo pipefail

UNITRACE_HOME="${UNITRACE_HOME:-}"
UNITRACE_USE_INSTALLED="${UNITRACE_USE_INSTALLED:-false}"
INCLUDE_KERNELS="${INCLUDE_KERNELS:-}"
EXCLUDE_KERNELS="${EXCLUDE_KERNELS:-}"
OUT="${OUT_DIR:-.sycl/reports/e2e_$(date +%Y%m%d_%H%M%S)}"

resolve_unitrace_paths() {
  if [[ "$UNITRACE_USE_INSTALLED" == true ]]; then
    UNITRACE_BIN="$UNITRACE_HOME/bin/unitrace"
  else
    UNITRACE_BIN="$UNITRACE_HOME/build/unitrace"
  fi
}
have_bin() { [[ -n "$1" && -x "$1" ]] || command -v "$1" >/dev/null 2>&1; }

usage() {
  echo "usage: profile_e2e.sh <overview|slice|resume|pause|stop> ..." >&2
  echo "  overview <app> [args...]          # one-shot host+device trace (small workloads only)" >&2
  echo "  slice    <session> <app> [args..] # run app --start-paused --session; collect a bracketed slice" >&2
  echo "  resume   <session>                # begin collecting on a paused session" >&2
  echo "  pause    <session>                # pause collecting" >&2
  echo "  stop     <session>                # stop + flush the session trace" >&2
}

if [[ $# -lt 1 ]]; then usage; exit 2; fi
MODE="$1"; shift

if [[ -z "$UNITRACE_HOME" ]]; then
  echo "profile_e2e: UNITRACE_HOME is unset — set .tools.unitrace_home in .sycl/config.json." >&2
  echo "             Falling back to in-app device timers to build the ranking (see sycl-profiler §5)." >&2
  exit 3
fi
resolve_unitrace_paths
if ! have_bin "$UNITRACE_BIN"; then
  echo "profile_e2e: unitrace binary not found at $UNITRACE_BIN" >&2
  echo "             (check .tools.unitrace_home / .tools.unitrace_use_installed). Falling back to" >&2
  echo "             in-app device timers to build the ranking (see sycl-profiler §5)." >&2
  exit 3
fi

# Assemble optional kernel-scoping args once (shared by overview + slice). Scoping the collection is
# the second lever (besides the time slice) for keeping the trace small enough not to crash unitrace.
scope_args=()
[[ -n "$INCLUDE_KERNELS" ]] && scope_args+=(--include-kernels "$INCLUDE_KERNELS")
[[ -n "$EXCLUDE_KERNELS" ]] && scope_args+=(--exclude-kernels "$EXCLUDE_KERNELS")

# The host+device timing lens: --host-timing (host API call overhead) + --device-timing /
# --kernel-submission / --device-timeline (per-kernel device execution). --chrome-kernel-logging emits
# a Perfetto JSON alongside the text summary for a visual timeline.
timing_args=(--call-logging --host-timing --device-timing --kernel-submission --device-timeline --chrome-kernel-logging)

case "$MODE" in
  overview)
    [[ $# -ge 1 ]] || { echo "profile_e2e overview: need <app> [args...]" >&2; exit 2; }
    mkdir -p "$OUT"
    OUTFILE="$OUT/e2e_overview.txt"
    echo "E2E overview -> $OUT"
    echo "  unitrace : $UNITRACE_BIN"
    [[ ${#scope_args[@]} -gt 0 ]] && echo "  scope    : ${scope_args[*]}"
    echo "WARNING: a full-run trace can OOM/core-dump unitrace on large workloads — if this crashes,"
    echo "         use the 'slice' mode (session --start-paused/--resume/--stop) or set INCLUDE_KERNELS." >&2
    "$UNITRACE_BIN" "${timing_args[@]}" "${scope_args[@]}" \
      --output-dir-path "$OUT" -o "$OUTFILE" "$@" \
      || { echo "profile_e2e: overview trace failed (crash? permissions?). Try 'slice' mode." >&2; exit 4; }
    echo "E2E overview complete: $OUTFILE"
    echo "Host overhead is under '=== Host Timing ==='; per-kernel device time under '=== Device Timing ==='."
    echo "Rank kernels by device-time share into .sycl/state/profile/e2e.json (descending wallclock_pct)."
    ;;

  slice)
    [[ $# -ge 2 ]] || { echo "profile_e2e slice: need <session> <app> [args...]" >&2; exit 2; }
    SESSION="$1"; shift
    mkdir -p "$OUT"
    OUTFILE="$OUT/e2e_slice_${SESSION}.txt"
    echo "E2E slice (session=$SESSION) -> $OUT"
    echo "  unitrace : $UNITRACE_BIN"
    [[ ${#scope_args[@]} -gt 0 ]] && echo "  scope    : ${scope_args[*]}"
    echo "Started PAUSED. From another shell: 'profile_e2e.sh resume $SESSION' to begin collecting a"
    echo "slice (e.g. one layer/iteration), then 'profile_e2e.sh stop $SESSION' to flush the trace."
    "$UNITRACE_BIN" --start-paused --session "$SESSION" "${timing_args[@]}" "${scope_args[@]}" \
      --output-dir-path "$OUT" -o "$OUTFILE" "$@" \
      || { echo "profile_e2e: slice trace failed for session=$SESSION." >&2; exit 4; }
    echo "E2E slice complete: $OUTFILE"
    echo "Host overhead under '=== Host Timing ==='; per-kernel device time under '=== Device Timing ==='."
    echo "Rank kernels by device-time share into .sycl/state/profile/e2e.json (descending wallclock_pct)."
    ;;

  resume) [[ $# -ge 1 ]] || { echo "profile_e2e resume: need <session>" >&2; exit 2; }
          echo "== resume session $1 =="; exec "$UNITRACE_BIN" --resume "$1" ;;
  pause)  [[ $# -ge 1 ]] || { echo "profile_e2e pause: need <session>" >&2; exit 2; }
          echo "== pause session $1 =="; exec "$UNITRACE_BIN" --pause "$1" ;;
  stop)   [[ $# -ge 1 ]] || { echo "profile_e2e stop: need <session>" >&2; exit 2; }
          echo "== stop session $1 =="; exec "$UNITRACE_BIN" --stop "$1" ;;

  -h|--help|help) usage; exit 0 ;;
  *) echo "profile_e2e: unknown mode '$MODE'" >&2; usage; exit 2 ;;
esac
