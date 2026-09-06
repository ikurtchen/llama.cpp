#!/usr/bin/env bash
# Deep GPU diagnosis for ONE kernel/workload. Three complementary lenses, combined to pick the
# *right* optimization instead of guessing (see sycl-optimization: DIAGNOSE):
#   1. HW metric counters (--metric-query|--metric-sampling --group <G>)  which resource is the bottleneck
#   2. aggregate stall breakdown (--group VectorEngineStalls)  the stall mix for the whole kernel
#   3. per-IP stall sampling (--stall-sampling) + IGC asm ....  which instructions stall & why (+spills)
#
# Deterministic tooling: everything is resolved from $UNITRACE_HOME (exported by run.sh from
# .sycl/config.json .tools.unitrace_home) using the same installed-vs-source layout as the reference
# run_unitrace.sh, so the unitrace binary, analyzeperfmetrics.py, and the metrics config all come
# from one known root — no reliance on PATH.
#
# Run on the Intel GPU runner, e.g.:
#   .sycl/scripts/run.sh profile "bash framework/.../profile_metrics.sh <build-dir>/test_<id>"
#
# Env knobs (all optional; run.sh exports UNITRACE_HOME / UNITRACE_USE_INSTALLED / METRICS_PLATFORM):
#   UNITRACE_HOME          root of the unitrace tree (required to resolve the binary + scripts)
#   UNITRACE_USE_INSTALLED true=installed layout (bin/, etc/), false=source tree (build/, scripts/)
#   PROFILE_MODE           all|metrics|stall (default all)
#                          - metrics: only the --metric-query group pass (fast follow-up to inspect
#                            e.g. MemoryProfile / DeviceCacheProfile after a first full run)
#                          - stall  : only the stall-sampling + IGC asm annotation pass
#   METRICS_GROUPS         comma-separated list of counter groups to query (commas only — spaces
#                          break when this is passed through the SSH command line)
#                          (default "VectorEngineStalls,ComputeBasic"; add MemoryProfile,
#                           DeviceCacheProfile, ComputeExtended, L3 as the diagnosis needs)
#   METRIC_MODE            query|sampling (default query) — how the counter groups are collected:
#                          - query   : --metric-query, one aggregate counter value per kernel
#                                      instance (best for "which resource is the bottleneck")
#                          - sampling: --metric-sampling, time-based samples over the run (a
#                                      counter timeline, pairs with --devices-to-sample); mirrors
#                                      the reference run_unitrace.sh trace_metrics_sample()
#   METRIC_SAMPLING_INTERVAL sampling interval for METRIC_MODE=sampling (default = SAMPLING_INTERVAL)
#   METRICS_PLATFORM       metrics config subdir (default bmg for Xe2 B60/B70)
#   SAMPLING_INTERVAL      stall/metric sampling interval (default 50)
#   SAMPLE_DEVICE          device index to sample (default 0)
#   KERNEL_NAME            target kernel for per-IP asm annotation (auto-detected if empty)
#   SHADER_DUMP            auto|on|off|prebuilt — capture an IGC asm dump for annotation (default auto)
#                          - auto/on : runtime IGC dump on the stall run (JIT builds only; an AOT build
#                                      dumps at BUILD time so the runtime dump is empty)
#                          - prebuilt: reuse an existing SHADER_DUMP_DIR from a build-time (AOT) dump;
#                                      the helper does NOT wipe it and does NOT set IGC env at runtime
#                          - off     : skip asm annotation
#   SHADER_DUMP_DIR        where the IGC dump goes / is read from (default <OUT_DIR>/shader_dump;
#                          wiped each run unless SHADER_DUMP=prebuilt)
#   OUT_DIR                output dir (default .sycl/reports/deep_<ts>)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSPOSE_PY="$SCRIPT_DIR/transpose_csv.py"
ANNOTATE_PY="$SCRIPT_DIR/annotate_asm_stalls.py"

UNITRACE_HOME="${UNITRACE_HOME:-}"
UNITRACE_USE_INSTALLED="${UNITRACE_USE_INSTALLED:-false}"
PROFILE_MODE="${PROFILE_MODE:-all}"
METRICS_GROUPS="${METRICS_GROUPS:-VectorEngineStalls,ComputeBasic}"
METRIC_MODE="${METRIC_MODE:-query}"
METRICS_PLATFORM="${METRICS_PLATFORM:-bmg}"
INTERVAL="${SAMPLING_INTERVAL:-50}"
METRIC_SAMPLING_INTERVAL="${METRIC_SAMPLING_INTERVAL:-$INTERVAL}"
DEV="${SAMPLE_DEVICE:-0}"
KERNEL_NAME="${KERNEL_NAME:-}"
SHADER_DUMP="${SHADER_DUMP:-auto}"
OUT="${OUT_DIR:-.sycl/reports/deep_$(date +%Y%m%d_%H%M%S)}"
SHADER_DUMP_DIR="${SHADER_DUMP_DIR:-$OUT/shader_dump}"

# --- deterministic path resolution (installed tree vs source tree) ------------------------------
resolve_unitrace_paths() {
  if [[ "$UNITRACE_USE_INSTALLED" == true ]]; then
    UNITRACE_BIN="$UNITRACE_HOME/bin/unitrace"
    ANALYZE_PY="$UNITRACE_HOME/bin/analyzeperfmetrics.py"
    METRICS_CONFIG_DIR="$UNITRACE_HOME/etc/unitrace/metrics/config"
  else
    UNITRACE_BIN="$UNITRACE_HOME/build/unitrace"
    ANALYZE_PY="$UNITRACE_HOME/scripts/metrics/analyzeperfmetrics.py"
    METRICS_CONFIG_DIR="$UNITRACE_HOME/scripts/metrics/config"
  fi
}

have_bin() { [[ -n "$1" && -x "$1" ]] || command -v "$1" >/dev/null 2>&1; }

# Transpose every metrics CSV that unitrace just wrote and does not yet have a *_transposed.csv.
# unitrace emits one wide "long" CSV per group (metrics as rows under a 3-line banner); the
# transposed view (one row per kernel instance, one column per metric) is what humans and downstream
# tooling actually read — so we produce it for EVERY metrics CSV, mirroring run_unitrace.sh's
# auto_transpose. Pure-csv (no pandas), so it works even before the runtime deps are installed.
transpose_new_metrics_csvs() {
  have_bin python3 || return 0
  [[ -f "$TRANSPOSE_PY" ]] || { echo "profile_metrics: transpose_csv.py not found at $TRANSPOSE_PY — skipping transpose." >&2; return 0; }
  local csv out
  while IFS= read -r csv; do
    [[ -z "$csv" ]] && continue
    out="${csv%.csv}_transposed.csv"
    [[ -f "$out" ]] && continue
    if python3 "$TRANSPOSE_PY" --skip 3 -o "$out" "$csv"; then
      echo "   transposed -> $(basename "$out")"
    else
      echo "profile_metrics: transpose failed for $(basename "$csv")" >&2
    fi
  done < <(find "$OUT" -maxdepth 1 \( -name '*.metrics.*.csv' -o -name '*.*.metrics.csv' \) ! -name '*_transposed.csv' 2>/dev/null)
}

if [[ $# -lt 1 ]]; then
  echo "usage: profile_metrics.sh <application> [args...]" >&2
  exit 2
fi

case "$PROFILE_MODE" in
  all|metrics|stall) ;;
  *) echo "profile_metrics: invalid PROFILE_MODE='$PROFILE_MODE' (use all|metrics|stall)" >&2; exit 2 ;;
esac

case "$METRIC_MODE" in
  query|sampling) ;;
  *) echo "profile_metrics: invalid METRIC_MODE='$METRIC_MODE' (use query|sampling)" >&2; exit 2 ;;
esac

if [[ -z "$UNITRACE_HOME" ]]; then
  echo "profile_metrics: UNITRACE_HOME is unset — set .tools.unitrace_home in .sycl/config.json." >&2
  echo "              Falling back to VTune or roofline+timing for this kernel." >&2
  exit 3
fi
resolve_unitrace_paths
if ! have_bin "$UNITRACE_BIN"; then
  echo "profile_metrics: unitrace binary not found at $UNITRACE_BIN" >&2
  echo "              (check .tools.unitrace_home / .tools.unitrace_use_installed). Falling back to" >&2
  echo "              VTune or roofline+timing for this kernel." >&2
  exit 3
fi

mkdir -p "$OUT"
# Split METRICS_GROUPS on commas only (spaces are unsafe once this is passed through SSH).
# NOTE: do NOT name this array GROUPS — that is a bash builtin (the caller's group IDs) and
# assigning to it is silently ignored, which would leave the group list as "0".
IFS=',' read -r -a GROUP_LIST <<< "$METRICS_GROUPS"
echo "Deep profile -> $OUT  (mode=$PROFILE_MODE)"
echo "  unitrace : $UNITRACE_BIN"
echo "  groups   : ${GROUP_LIST[*]}"
echo "  platform : $METRICS_PLATFORM   interval=$INTERVAL   device=$DEV"

# 1) Hardware metric counters, one CSV per group. VectorEngineStalls gives the full stall mix for the
#    whole kernel (Active vs each *Stall %); ComputeBasic gives EU activity, thread occupancy,
#    achieved bandwidth and cache hit rates; add MemoryProfile / DeviceCacheProfile / ComputeExtended
#    for a deeper look. Answers "which resource is the bottleneck". Skipped when PROFILE_MODE=stall.
if [[ "$PROFILE_MODE" != stall ]]; then
  n=${#GROUP_LIST[@]}; i=0
  for g in "${GROUP_LIST[@]}"; do
    g="${g//[[:space:]]/}"   # tolerate stray spaces around commas (e.g. "A, B")
    [[ -z "$g" ]] && continue
    i=$((i + 1))
    if [[ "$METRIC_MODE" == sampling ]]; then
      # Time-based counter sampling (a timeline of the group's counters), mirroring the reference
      # run_unitrace.sh trace_metrics_sample(): --metric-sampling + --sampling-interval +
      # --devices-to-sample. Shows how a counter evolves across the run rather than one aggregate
      # value per kernel instance.
      echo "== [$i/$n] metric sampling (--metric-sampling --group $g, interval=$METRIC_SAMPLING_INTERVAL dev=$DEV) =="
      "$UNITRACE_BIN" --metric-sampling --sampling-interval "$METRIC_SAMPLING_INTERVAL" \
        --group "$g" --devices-to-sample "$DEV" --chrome-kernel-logging \
        --output-dir-path "$OUT" -o "$OUT/metrics_${g}.csv" "$@" \
        || echo "profile_metrics: metric-sampling group=$g failed (permissions? try root / i915 perf enabled)" >&2
    else
      # One aggregate counter value per kernel instance — answers "which resource is the bottleneck".
      echo "== [$i/$n] metric counters (--metric-query --group $g) =="
      "$UNITRACE_BIN" --metric-query --group "$g" --chrome-kernel-logging \
        --output-dir-path "$OUT" -o "$OUT/metrics_${g}.csv" "$@" \
        || echo "profile_metrics: metric-query group=$g failed (permissions? try root / i915 perf enabled)" >&2
    fi
    # Transpose this group's freshly-written metrics CSV (one per group).
    transpose_new_metrics_csvs
  done
fi

if [[ "$PROFILE_MODE" == metrics ]]; then
  echo "Deep profile complete (metrics only): $OUT"
  echo "Inspect the per-group CSVs (e.g. metrics_MemoryProfile.csv) to see which resource is the limiter."
  exit 0
fi

# 2) Per-IP stall sampling: WHY and WHERE the EUs stall — SendStall (memory latency), SbidStall
#    (scoreboard/async dependency), PipeStall/DistStall (ALU latency / ILP), ControlStall (branch),
#    SyncStall (barrier), InstrFetchStall (icache). We also capture a clean IGC shader dump on this
#    same run so the sample IPs can be mapped onto the exact assembly (register spills, DPAS density,
#    SIMD width, uncoalesced sends). The dump dir is wiped first so it holds exactly ONE dump set for
#    this kernel — a hard requirement for analyzeperfmetrics.py's -k/-s asm annotation.
STALL_OUT="$OUT/stall.csv"
stall_env=()
if [[ "$SHADER_DUMP" == prebuilt ]]; then
  if [[ ! -d "$SHADER_DUMP_DIR" ]] || [[ -z "$(find "$SHADER_DUMP_DIR" -maxdepth 1 -name '*.asm' 2>/dev/null)" ]]; then
    echo "profile_metrics: SHADER_DUMP=prebuilt but no *.asm in $SHADER_DUMP_DIR" >&2
    echo "              Point SHADER_DUMP_DIR at a build-time IGC dump (AOT builds dump at build time," >&2
    echo "              not at run time), or use a JIT build with SHADER_DUMP=on." >&2
  else
    echo "== using prebuilt shader dump: $SHADER_DUMP_DIR (no runtime IGC dump) =="
  fi
elif [[ "$SHADER_DUMP" != off ]]; then
  rm -rf "$SHADER_DUMP_DIR"; mkdir -p "$SHADER_DUMP_DIR"
  stall_env=(env "IGC_ShaderDumpEnable=1" "IGC_DumpToCustomDir=$SHADER_DUMP_DIR")
fi
echo "== stall sampling (--stall-sampling, IGC dump -> ${SHADER_DUMP_DIR}) =="
"${stall_env[@]}" "$UNITRACE_BIN" --stall-sampling --sampling-interval "$INTERVAL" \
  --devices-to-sample "$DEV" --chrome-kernel-logging \
  --output-dir-path "$OUT" -o "$STALL_OUT" "$@" \
  || echo "profile_metrics: stall-sampling failed (needs stall-sampling-capable driver/HW, often root)" >&2

# 3) Annotate the assembly with per-IP stalls. Needs (a) the stall metrics CSV unitrace just wrote,
#    (b) a single target kernel name, and (c) a shader dump dir containing that kernel's asm.
STALL_METRICS_CSV="$(find "$OUT" -maxdepth 1 -name 'stall.metrics.*.csv' -o -name 'stall.*.metrics.csv' 2>/dev/null | head -1)"

resolve_kernel_name() {
  # explicit override wins
  [[ -n "$KERNEL_NAME" ]] && return 0
  # otherwise derive from the asm files IGC dumped into the (freshly wiped) dump dir
  local names=()
  mapfile -t names < <(find "$SHADER_DUMP_DIR" -maxdepth 1 -name '*.asm' -printf '%f\n' 2>/dev/null \
    | sed -E 's/\.asm$//; s/^[0-9a-fA-F]+_//' | sort -u)
  if [[ ${#names[@]} -eq 1 ]]; then
    KERNEL_NAME="${names[0]}"
    return 0
  fi
  return 1
}

if [[ -z "$STALL_METRICS_CSV" ]]; then
  echo "profile_metrics: no stall metrics CSV produced — skipping asm annotation." >&2
elif [[ "$SHADER_DUMP" == off ]]; then
  echo "profile_metrics: SHADER_DUMP=off — skipping asm annotation." >&2
elif ! have_bin "python3"; then
  echo "profile_metrics: python3 not available — skipping asm annotation." >&2
elif [[ ! -f "$ANALYZE_PY" ]]; then
  echo "profile_metrics: analyzeperfmetrics.py not found at $ANALYZE_PY — skipping asm annotation." >&2
elif ! python3 -c 'import pandas, matplotlib' 2>/dev/null; then
  echo "profile_metrics: analyzeperfmetrics.py needs pandas + matplotlib, which are not importable —" >&2
  echo "              skipping asm annotation. Install them on this runner with: .sycl/scripts/run.sh setup" >&2
else
  asm_count=$(find "$SHADER_DUMP_DIR" -maxdepth 1 -name '*.asm' 2>/dev/null | wc -l)
  if [[ "$asm_count" -eq 0 ]]; then
    echo "profile_metrics: no *.asm in $SHADER_DUMP_DIR (IGC dump empty?) — skipping asm annotation." >&2
    echo "              Most common cause: the binary was built AOT (-fsycl-targets=spir64_gen), so" >&2
    echo "              IGC compiled it at BUILD time and the runtime dump is empty. Rebuild the" >&2
    echo "              profiling target JIT (drop the AOT flags) and re-run, or capture the dump at" >&2
    echo "              build time and pass SHADER_DUMP=prebuilt SHADER_DUMP_DIR=<build-dump>." >&2
  elif ! resolve_kernel_name; then
    echo "profile_metrics: $asm_count kernels dumped in $SHADER_DUMP_DIR — cannot pick one automatically." >&2
    echo "              Re-run with KERNEL_NAME=<kernel> set, or profile a single-kernel test so the" >&2
    echo "              dump holds exactly one kernel (required for reliable IP->asm mapping)." >&2
  else
    echo "== asm stall annotation (kernel=$KERNEL_NAME) =="
    if python3 "$ANALYZE_PY" -k "$KERNEL_NAME" -s "$SHADER_DUMP_DIR" \
        -o "$OUT/stall_${KERNEL_NAME}.pdf" "$STALL_METRICS_CSV"; then
      echo "   annotated asm + report under $OUT (look for ${KERNEL_NAME}.asm.ip / .pdf)"
      # Overlay per-IP stall counts (with top-N hotspot ranks) onto the .asm.ip file emitted above,
      # producing <kernel>.asm.ip.stall — the instruction-level view used to pinpoint the exact
      # stalling instructions. Pure-csv (no pandas); mirrors run_unitrace.sh's annotate step.
      if [[ -f "$ANNOTATE_PY" ]]; then
        python3 "$ANNOTATE_PY" -k "$KERNEL_NAME" -s "$SHADER_DUMP_DIR" -d "$DEV" "$STALL_METRICS_CSV" \
          && echo "   per-IP stall overlay -> $SHADER_DUMP_DIR/*.asm.ip.stall" \
          || echo "profile_metrics: annotate_asm_stalls.py failed for kernel=$KERNEL_NAME." >&2
      else
        echo "profile_metrics: annotate_asm_stalls.py not found at $ANNOTATE_PY — skipping per-IP overlay." >&2
      fi
    else
      echo "profile_metrics: analyzeperfmetrics.py failed for kernel=$KERNEL_NAME." >&2
    fi
  fi
fi

echo "Deep profile complete: $OUT"
echo "Combine the three lenses: counters say WHICH resource, VectorEngineStalls says the stall MIX,"
echo "the annotated asm says WHERE/WHY — then pick one optimization (see sycl-optimization: DIAGNOSE)."
