#!/usr/bin/env bash
# Force-rotate all logs under <.sycl>/logs regardless of size.
# Usage: rotate-logs.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SYCL_DIR/logs"
MAX_BACKUPS="${SYCL_LOG_MAX_BACKUPS:-5}"

[[ -d "$LOG_DIR" ]] || { echo "No log dir: $LOG_DIR"; exit 0; }

shopt -s nullglob
for f in "$LOG_DIR"/*.log; do
  [[ -s "$f" ]] || continue
  i="$MAX_BACKUPS"
  rm -f "${f}.${MAX_BACKUPS}" 2>/dev/null || true
  while (( i > 1 )); do
    [[ -f "${f}.$((i-1))" ]] && mv -f "${f}.$((i-1))" "${f}.${i}"
    ((i--))
  done
  mv -f "$f" "${f}.1"
  echo "rotated: $(basename "$f")"
done
