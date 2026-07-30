#!/usr/bin/env bash
# Append an audit log entry as JSONL, with size-based rotation.
# Usage: log.sh <agent> <phase> <level> "<message>"
#   level: info | warn | error
# Logs are written to <.sycl>/logs/<agent>.log next to this script's parent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"     # .sycl/
LOG_DIR="$SYCL_DIR/logs"

MAX_BYTES="${SYCL_LOG_MAX_BYTES:-5242880}"    # 5 MB
MAX_BACKUPS="${SYCL_LOG_MAX_BACKUPS:-5}"

agent="${1:-unknown}"
phase="${2:-unknown}"
level="${3:-info}"
msg="${4:-}"

mkdir -p "$LOG_DIR"
logfile="$LOG_DIR/${agent}.log"

# Rotate if the file is at/over the cap.
rotate_if_needed() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local size
  size=$(wc -c < "$f" 2>/dev/null || echo 0)
  if (( size >= MAX_BYTES )); then
    local i="$MAX_BACKUPS"
    rm -f "${f}.${MAX_BACKUPS}" 2>/dev/null || true
    while (( i > 1 )); do
      [[ -f "${f}.$((i-1))" ]] && mv -f "${f}.$((i-1))" "${f}.${i}"
      ((i--))
    done
    mv -f "$f" "${f}.1"
  fi
}

rotate_if_needed "$logfile"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Escape backslashes and double quotes for JSON.
esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

printf '{"ts":"%s","agent":"%s","phase":"%s","level":"%s","msg":"%s"}\n' \
  "$ts" "$(esc "$agent")" "$(esc "$phase")" "$(esc "$level")" "$(esc "$msg")" \
  >> "$logfile"
