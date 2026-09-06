#!/usr/bin/env bash
# Measure how much CUDA/Triton code a migration actually covered, and how much SYCL it took.
#   loc.sh count  <file[:span|:sym,sym]>...   # ad-hoc counts
#   loc.sh spans  <file> <sym1,sym2>          # resolve a kernel symbol to its line span
#   loc.sh kernel <id> [--source SPEC]... [--sycl SPEC]... [--no-mirror] [--dry-run]
#   loc.sh all                                # measure every non-skipped kernel, then roll up
#   loc.sh rollup                             # -> .sycl/state/loc.json (report §2 input)
#   loc.sh scan                               # project-wide CUDA/Triton/SYCL totals
# Run `loc.sh all` at the end of inventory (source side) and again after migrate (SYCL side).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"   # .sycl/

exec python3 "$SCRIPT_DIR/loc.py" --sycl-dir "$SYCL_DIR" "$@"
