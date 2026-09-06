#!/usr/bin/env bash
# Run a benchmark command with warm-up + timed iterations; print median/p90/stddev in ms.
# The command should run ONE iteration of the workload and exit.
# Usage: bench.sh --warmup 3 --iters 20 -- <command...>
set -euo pipefail

WARMUP=3; ITERS=20
while [[ $# -gt 0 ]]; do
  case "$1" in
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters)  ITERS="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[[ $# -ge 1 ]] || { echo "usage: bench.sh [--warmup N] [--iters N] -- <command...>" >&2; exit 2; }

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"

for ((i=0;i<WARMUP;i++)); do "$@" >/dev/null 2>&1 || true; done

times=()
for ((i=0;i<ITERS;i++)); do
  start=$(date +%s.%N)
  "$@" >/dev/null 2>&1
  end=$(date +%s.%N)
  times+=("$(echo "($end - $start)*1000" | bc -l)")
done

printf '%s\n' "${times[@]}" | python3 -c '
import sys, statistics
xs=sorted(float(l) for l in sys.stdin if l.strip())
n=len(xs)
med=statistics.median(xs)
p90=xs[min(n-1, int(0.9*n))]
sd=statistics.pstdev(xs) if n>1 else 0.0
print(f'"'"'{{"iters":{n},"median_ms":{med:.4f},"p90_ms":{p90:.4f},"stddev_ms":{sd:.4f},"min_ms":{xs[0]:.4f}}}'"'"')
'
