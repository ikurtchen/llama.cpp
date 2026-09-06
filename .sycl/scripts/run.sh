#!/usr/bin/env bash
# Execution dispatcher: run build/test/profile/bench/env commands locally or on a remote
# Intel GPU host, per .sycl/config.json. The agent NEVER calls icpx/unitrace/tests directly —
# it always goes through this wrapper so the same workflow works with or without a local GPU.
#
# Usage:
#   run.sh env                      # print detected toolchain/GPU on the runner
#   run.sh setup                    # force-(re)install sycl-agent Python deps (auto-runs once otherwise)
#   run.sh sync                     # push the project to the remote workdir, additively (no-op when local)
#   run.sh sync --prune             # push AND delete remote files the project no longer has
#                                   #   (runner-produced trees stay protected; also SYCL_SYNC_PRUNE=1)
#   run.sh pull                     # fetch remote-generated .sycl/reports + logs + state back (no-op when local)
#   run.sh exec  "<shell command>"  # run an arbitrary command on the runner (in project dir)
#   run.sh build "<build command>"  # convenience alias for exec, tagged as a build
#   run.sh test  "<test command>"
#   run.sh profile "<profile command>"
#   run.sh bench "<bench command>"
#
# GPU concurrency safety:
#   Multiple agents/subagents can call run.sh at the same time. GPU-bound verbs (exec/build/test/
#   profile/bench) are SERIALIZED with a file lock so only ONE of them runs on the GPU at a time.
#   Concurrent callers wait with a visible message; the lock is released when the command finishes
#   or if the process is killed. Non-GPU verbs (env/setup/sync/pull) do not take the lock.
#   Configure timeout/location in .sycl/config.json under guardrails.gpu_concurrency.
#
# Sync/pull safety model (both directions are non-destructive by default):
#   push  never deletes on the remote unless --prune is passed, and even then a protect-list keeps
#         every runner-produced tree (reports, logs, golden refs, build/) alive.
#   pull  never clobbers local files: state is copied with --ignore-existing (local authored state
#         always wins) and reports/logs only take strictly newer remote copies, with any replaced
#         local file preserved under .sycl/.pull-backups/<timestamp>/.
#
# Config (.sycl/config.json) shape:
#   {
#     "runner": {
#       "mode": "local" | "remote",
#       "remote": {
#         "host": "10.239.98.41", "port": 2332, "user": "",
#         "workdir": "~/sycl-agent-runs/<project>",
#         "env_setup": "source /opt/intel/oneapi/setvars.sh",
#         "identity_file": "",
#         "env": ["ONEAPI_DEVICE_SELECTOR=level_zero:gpu", "HF_HOME=/scratch/hf"],
#         "proxy": { "http": "", "https": "", "no_proxy": "localhost,127.0.0.1" }
#       },
#       "local": { "env_setup": "...", "env": [], "proxy": { "http": "", "https": "", "no_proxy": "" } }
#     }
#   }
# When .runner.<scope>.proxy.http/https is set, every command on that runner is prefixed with the
# proxy env vars so Internet-facing steps (git clone, pip/apt, reference-repo fetch) use the proxy.
# .runner.<scope>.env is a list of "KEY=VALUE" strings exported (shell-quoted) before every command.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # .sycl/
PROJECT_DIR="$(cd "$SYCL_DIR/.." && pwd)"         # target project root
CONFIG="$SYCL_DIR/config.json"

[[ -f "$CONFIG" ]] || { echo "run.sh: missing $CONFIG" >&2; exit 2; }

# --- GPU concurrency lock -----------------------------------------------------
# A single file lock serializes all GPU-bound work (exec/build/test/profile/bench)
# across concurrent agent/subagent invocations. The lock lives on the RUNNER side
# when mode=remote (inside the remote workdir) so multiple local agents sharing one
# remote GPU do not overload it. When mode=local it lives under .sycl/ in the project.
# Non-GPU verbs (env/setup/sync/pull) never take the lock.
# Defaults are applied here; config is read after the get() helper is defined below,
# so the final values are resolved just before use in run_cmd().
GPU_LOCK_ENABLED_DEFAULT="true"
GPU_LOCK_TIMEOUT_DEFAULT="3600"
GPU_LOCK_FILE_DEFAULT="$SYCL_DIR/.gpu-run.lock"

# Acquire the lock file descriptor <lock_fd> with a timeout. Uses flock(1) when
# available (Linux), else falls back to a mkdir-based spin lock (portable).
# Prints a waiting message once, then silently retries until acquired or timeout.
_gpu_lock_acquire() {
  local lock_path="$1" timeout="${2:-3600}"
  local waited=0
  if command -v flock >/dev/null 2>&1; then
    local fd
    exec {fd}>"$lock_path" || { echo "run.sh: cannot open lock $lock_path" >&2; return 1; }
    # flock -w 0 fails immediately if locked; loop with sleeps for visibility.
    while ! flock -n "$fd" 2>/dev/null; do
      if (( waited == 0 )); then echo "[gpu-lock] waiting: another GPU job holds $lock_path" >&2; fi
      sleep 5
      waited=$((waited + 5))
      if (( waited > timeout )); then
        echo "[gpu-lock] timeout after ${timeout}s waiting for $lock_path" >&2
        return 1
      fi
    done
    # Keep fd open for the duration of the shell; caller must close it.
    GPU_LOCK_FD=$fd
    return 0
  else
    # mkdir is atomic on POSIX; spin until we create the directory.
    while ! mkdir "$lock_path" 2>/dev/null; do
      if (( waited == 0 )); then echo "[gpu-lock] waiting: another GPU job holds $lock_path" >&2; fi
      sleep 5
      waited=$((waited + 5))
      if (( waited > timeout )); then
        echo "[gpu-lock] timeout after ${timeout}s waiting for $lock_path" >&2
        return 1
      fi
    done
    GPU_LOCK_FD=""
    return 0
  fi
}

_gpu_lock_release() {
  if [[ -n "${GPU_LOCK_FD:-}" ]]; then
    flock -u "$GPU_LOCK_FD" 2>/dev/null || true
    eval "exec ${GPU_LOCK_FD}>&-" 2>/dev/null || true
  elif [[ -n "${GPU_LOCK_PATH:-}" ]]; then
    rmdir "$GPU_LOCK_PATH" 2>/dev/null || rm -f "$GPU_LOCK_PATH" 2>/dev/null || true
  fi
  unset GPU_LOCK_FD GPU_LOCK_PATH
}

# Ensure the lock is released on normal exit, error, or signal.
trap '_gpu_lock_release' EXIT

# Content hash of the runtime requirements file, used as the dependency-install marker key so a
# changed deps list triggers a one-time reinstall. cksum is POSIX and always present; missing file
# -> "0" (deps-ensure then no-ops on the [ -f requirements ] guard).
REQ_FILE="$SYCL_DIR/requirements.txt"
REQ_HASH="$( { cksum "$REQ_FILE" 2>/dev/null || echo 0; } | awk '{print $1}')"

# --- Minimal JSON reader (jq if present, else python3) ---
# python3 fallback that reads a scalar at a dot-path (empty string when absent).
pyget() { python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
cur=d
for k in sys.argv[2].strip(".").split("."):
    if isinstance(cur,dict) and k in cur: cur=cur[k]
    else: cur=""; break
print(cur if cur is not None else "")' "$CONFIG" "$1"; }

get() { # get <dot-path> -> scalar value (prefer jq, else python3)
  if command -v jq >/dev/null 2>&1; then jq -r "$1 // empty" "$CONFIG"; else
    pyget "$1"; fi
}

get_array() { # get_array <dot-path> -> one element per line (empty when absent / not an array)
  if command -v jq >/dev/null 2>&1; then
    jq -r "$1 // [] | .[]" "$CONFIG"
  else
    python3 -c 'import json,sys
d=json.load(open(sys.argv[1]));cur=d
for k in sys.argv[2].strip(".").split("."):
    cur=cur.get(k) if isinstance(cur,dict) else None
    if cur is None: break
if isinstance(cur,list):
    for x in cur:
        if x is not None: print(x)' "$CONFIG" "$1"
  fi
}

MODE="$(get '.runner.mode')"; MODE="${MODE:-local}"

remote_ssh_args() {
  local host port user ident
  host="$(get '.runner.remote.host')"
  port="$(get '.runner.remote.port')"; port="${port:-22}"
  user="$(get '.runner.remote.user')"
  ident="$(get '.runner.remote.identity_file')"
  local dest="$host"
  [[ -n "$user" ]] && dest="${user}@${host}"
  SSH=(ssh -p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
  [[ -n "$ident" ]] && SSH+=(-i "$ident")
  SSH+=("$dest")
  RSYNC_SSH="ssh -p $port"
  [[ -n "$ident" ]] && RSYNC_SSH="$RSYNC_SSH -i $ident"
  REMOTE_DEST="$dest"
  REMOTE_WORKDIR="$(get '.runner.remote.workdir')"
  REMOTE_WORKDIR="${REMOTE_WORKDIR:-~/sycl-agent-run}"
  ENV_SETUP="$(get '.runner.remote.env_setup')"
}

local_env_setup() { get '.runner.local.env_setup'; }

# Build a prefix that exports configured tool paths so profile/bench commands can reference
# $UNITRACE_HOME / $UNITRACE / $VTUNE / $XPU_SMI regardless of where each server installs them.
# unitrace is resolved deterministically from .tools.unitrace_home + .tools.unitrace_use_installed
# (installed tree -> bin/unitrace, source tree -> build/unitrace); an explicit .tools.unitrace path
# overrides the derivation. Falls back to the bare tool name (resolved via PATH) when unset.
tools_export_prefix() {
  local home installed uni mplat vtune xpu_smi
  home="$(get '.tools.unitrace_home')"
  installed="$(get '.tools.unitrace_use_installed')"; installed="${installed:-false}"
  mplat="$(get '.tools.metrics_platform')";           mplat="${mplat:-bmg}"
  vtune="$(get '.tools.vtune')";                       vtune="${vtune:-vtune}"
  xpu_smi="$(get '.tools.xpu_smi')";                   xpu_smi="${xpu_smi:-xpu-smi}"
  uni="$(get '.tools.unitrace')"
  if [[ -z "$uni" && -n "$home" ]]; then
    if [[ "$installed" == "true" ]]; then uni="$home/bin/unitrace"; else uni="$home/build/unitrace"; fi
  fi
  uni="${uni:-unitrace}"
  printf 'export UNITRACE_HOME=%q UNITRACE_USE_INSTALLED=%q UNITRACE=%q METRICS_PLATFORM=%q VTUNE=%q XPU_SMI=%q; ' \
    "$home" "$installed" "$uni" "$mplat" "$vtune" "$xpu_smi"
}

# Build a prefix that exports HTTP(S) proxy env vars when a proxy is configured for the active
# runner (.runner.<scope>.proxy). This makes every command that needs Internet access — git clone,
# pip/apt installs, fetching reference repos — go through the proxy. Emits nothing when no proxy is
# configured, so direct/offline setups are unaffected. If only one of http/https is set, the other
# mirrors it.
proxy_export_prefix() { # proxy_export_prefix <local|remote>
  local scope="$1" http https noproxy
  http="$(get ".runner.${scope}.proxy.http")"
  https="$(get ".runner.${scope}.proxy.https")"
  noproxy="$(get ".runner.${scope}.proxy.no_proxy")"
  [[ -z "$http" && -z "$https" ]] && return 0
  https="${https:-$http}"; http="${http:-$https}"
  printf 'export http_proxy=%q https_proxy=%q HTTP_PROXY=%q HTTPS_PROXY=%q' \
    "$http" "$https" "$http" "$https"
  [[ -n "$noproxy" ]] && printf ' no_proxy=%q NO_PROXY=%q' "$noproxy" "$noproxy"
  printf '; '
}

# Build a prefix that exports user-defined environment variables for the active runner
# (.runner.<scope>.env — a JSON array of "KEY=VALUE" strings). Applied to EVERY command, so users
# can inject arbitrary runner environment (ONEAPI_DEVICE_SELECTOR, cache dirs, tokens, ulimit-ish
# knobs, etc.). Emitted LAST so a user value overrides the tool/proxy defaults. Each key and value
# is shell-quoted so spaces and special characters survive the remote SSH command line. Malformed
# entries are skipped with a warning rather than corrupting the command.
env_export_prefix() { # env_export_prefix <local|remote>
  local scope="$1" pair key val out=""
  while IFS= read -r pair; do
    [[ -z "$pair" ]] && continue
    if [[ "$pair" != *=* ]]; then
      echo "run.sh: skipping malformed .runner.${scope}.env entry (need KEY=VALUE): $pair" >&2
      continue
    fi
    key="${pair%%=*}"; val="${pair#*=}"
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "run.sh: skipping .runner.${scope}.env entry with invalid variable name: '$key'" >&2
      continue
    fi
    out+="export ${key}=$(printf '%q' "$val"); "
  done < <(get_array ".runner.${scope}.env")
  printf '%s' "$out"
}

# Emit a shell snippet (run inside the SAME shell as the command, after all env exports so pip can
# reach PyPI through the configured proxy) that installs the sycl-agent Python deps ONCE per runner.
# Idempotent + cheap: after the first success a per-runner marker (keyed to the requirements content
# hash, kept under $HOME/.cache/sycl-agent — outside the synced workdir, so no `sync --prune` can
# wipe it) short-circuits the check to a single `[ -f ]`. Non-fatal: a failed install (e.g. offline) does NOT create the
# marker (so it retries next run) and never aborts the actual command. Suppressed when
# SYCL_SKIP_DEPS_ENSURE=1 (used by `run.sh setup`, which force-installs explicitly).
deps_ensure_snippet() {
  [[ "${SYCL_SKIP_DEPS_ENSURE:-0}" == 1 ]] && return 0
  local pip_args="${PIP_ARGS:-}"
  printf '{ __sdreq=.sycl/requirements.txt; __sdm="$HOME/.cache/sycl-agent/deps-%s"; if [ -f "$__sdreq" ] && [ ! -f "$__sdm" ]; then echo "[setup] installing sycl-agent Python deps (one-time)" >&2; if python3 -m pip install -q %s -r "$__sdreq"; then mkdir -p "$(dirname "$__sdm")" && : > "$__sdm"; else echo "run.sh: sycl-agent dep install failed (offline/proxy?); will retry next run" >&2; fi; fi; }; ' \
    "$REQ_HASH" "$pip_args"
}

# Files never worth shipping to the runner: VCS metadata, build output and object files (rebuilt
# there anyway), the two trees the runner itself writes (.sycl/reports, .sycl/logs) — pushing a
# locally-stale copy of those would fight with `run.sh pull` — and the pull backup vault, which is
# purely local safety history. Golden reference tensors are also held back from the main pass: they
# are runner-generated, so they get their own --ignore-existing seeding pass (see do_sync) instead
# of being overwritten by whatever copy this machine happens to hold. The GPU lock file is excluded
# too: do_sync runs on every run_cmd call, and rsync's default temp-file+rename write replaces the
# destination inode, which would silently drop any flock a concurrent job holds on the old inode
# and hand the next job a fresh, unlocked file - defeating serialization entirely.
PUSH_EXCLUDES=(
  --exclude '.git/'
  --exclude 'build/'
  --exclude '*.o'
  --exclude '.sycl/logs/'
  --exclude '.sycl/reports/'
  --exclude '.sycl/.pull-backups/'
  --exclude '.sycl/state/kernels/*/ref/'
  --exclude '.sycl/.gpu-run.lock'
)

# Filter selecting exactly the golden-reference trees under .sycl/state/ (relative to .sycl/state/
# as the transfer root). Used by both the push seeding pass and the pull, so authored state JSON is
# never in scope in either direction.
REF_ONLY_FILTER=(
  --include='kernels/'
  --include='kernels/*/'
  --include='kernels/*/ref/'
  --include='kernels/*/ref/***'
  --exclude='*'
)

# Deletion guard for `sync --prune`. rsync `protect` rules bind to the RECEIVER: a matching remote
# path survives --delete even when the local side has no counterpart. This is what makes pruning
# safe — it can only remove files that came from the project tree, never runner-produced artifacts
# (profiling reports, logs, golden reference tensors, build output).
PRUNE_PROTECT=(
  --filter 'protect .sycl/reports/***'
  --filter 'protect .sycl/logs/***'
  --filter 'protect .sycl/state/kernels/*/ref/***'
  --filter 'protect build/***'
  --filter 'protect .git/***'
  --filter 'protect *.o'
)

# Push local -> remote. ADDITIVE by default (no --delete): a remote file with no local counterpart
# is left untouched, because we cannot tell a stale leftover from an artifact the runner just
# produced under a path we don't know about. Stale sources are cheap; lost artifacts are not.
# Pass --prune (or set SYCL_SYNC_PRUNE=1) to opt into --delete for a genuinely clean mirror; the
# PRUNE_PROTECT list still shields every runner-produced tree. Pruning is NEVER implicit: run_cmd
# always calls the additive form.
do_sync() { # do_sync [--prune]
  [[ "$MODE" == "remote" ]] || { echo "[local] sync: no-op"; return 0; }
  remote_ssh_args
  local prune=0
  [[ "${1:-}" == "--prune" || "${SYCL_SYNC_PRUNE:-0}" == 1 ]] && prune=1
  local extra=()
  if (( prune )); then
    extra=(--delete "${PRUNE_PROTECT[@]}")
    echo "[remote] sync --prune -> ${REMOTE_DEST}:${REMOTE_WORKDIR} (deletes stale project files; artifacts protected)"
  else
    echo "[remote] sync -> ${REMOTE_DEST}:${REMOTE_WORKDIR} (additive, no deletions)"
  fi
  "${SSH[@]}" "mkdir -p ${REMOTE_WORKDIR}"
  rsync -az \
    -e "$RSYNC_SSH" \
    "${PUSH_EXCLUDES[@]}" \
    ${extra[@]+"${extra[@]}"} \
    "$PROJECT_DIR/" "${REMOTE_DEST}:${REMOTE_WORKDIR}/"
  # Second pass: seed golden reference tensors onto a fresh workdir WITHOUT ever replacing one the
  # runner already has. Goldens are produced remotely; the local copy is only a pulled-back mirror,
  # so a plain overwrite could push stale data over a freshly regenerated reference. --ignore-existing
  # makes the golden trees additive-only in both directions.
  if [[ -d "$SYCL_DIR/state" ]]; then
    rsync -az --ignore-existing \
      -e "$RSYNC_SSH" \
      "${REF_ONLY_FILTER[@]}" \
      "$SYCL_DIR/state/" "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/state/"
  fi
}

# Pull remote-generated outputs back to the local project so the agent and a human can read/audit
# them. Remote -> local, never with --delete, and never with a plain overwrite. Three classes, each
# with the weakest write mode that still gets the data across:
#   .sycl/reports/  profiling artifacts (metric CSVs, stall CSVs, IGC shader dumps) — --update, so a
#                   local file that is newer than the remote one (e.g. a locally annotated report)
#                   is kept; anything actually replaced is first copied to the backup dir.
#   .sycl/logs/     logs written on the runner — same --update + backup treatment.
#   .sycl/state/kernels/<id>/ref/  golden reference tensors are generated ON THE RUNNER (by running
#                   the CUDA/CPU reference) yet live under state/. Pulled with --ignore-existing and
#                   filtered down to ref/ only: authored state (*.json) is never even considered,
#                   and an existing local golden is never rewritten. Local state always wins.
# Nothing here can destroy local work: the worst case is a superseded file moved into
# .sycl/.pull-backups/<timestamp>/ instead of being lost.
do_pull() {
  [[ "$MODE" == "remote" ]] || { echo "[local] pull: no-op"; return 0; }
  remote_ssh_args
  # One timestamped backup root per pull. rsync creates it lazily, so a pull that overwrote nothing
  # leaves no directory behind.
  local backup_root="$SYCL_DIR/.pull-backups/$(date +%Y%m%d-%H%M%S)"
  echo "[remote] pull <- ${REMOTE_DEST}:${REMOTE_WORKDIR} (.sycl/reports, .sycl/logs, state ref/)"
  mkdir -p "$PROJECT_DIR/.sycl/reports" "$PROJECT_DIR/.sycl/logs" "$PROJECT_DIR/.sycl/state"
  # Trailing '/' on sources copies contents into the matching local dir. Missing remote dirs are
  # tolerated (rsync warns but we don't fail the run over absent optional output).
  rsync -az --update --backup --backup-dir="$backup_root/reports" \
    -e "$RSYNC_SSH" \
    "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/reports/" "$PROJECT_DIR/.sycl/reports/" 2>/dev/null || true
  rsync -az --update --backup --backup-dir="$backup_root/logs" \
    -e "$RSYNC_SSH" \
    "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/logs/" "$PROJECT_DIR/.sycl/logs/" 2>/dev/null || true
  # Only the kernels/<id>/ref/ golden data — the shared ref-only filter keeps authored state/*.json
  # entirely out of scope, and --ignore-existing makes this strictly additive: only files absent
  # locally are fetched, so an existing local golden is never rewritten.
  rsync -az --ignore-existing \
    -e "$RSYNC_SSH" \
    "${REF_ONLY_FILTER[@]}" \
    "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/state/" "$PROJECT_DIR/.sycl/state/" 2>/dev/null || true
  [[ -d "$backup_root" ]] && echo "[remote] pull: replaced local files preserved under ${backup_root#$PROJECT_DIR/}"
  return 0
}

run_cmd() { # run_cmd "<shell command>"
  local cmd="$1"
  # Resolve GPU lock settings now that get() is available.
  local lock_enabled lock_timeout lock_file
  lock_enabled="$(get '.guardrails.gpu_concurrency.enabled')"; lock_enabled="${lock_enabled:-$GPU_LOCK_ENABLED_DEFAULT}"
  lock_timeout="$(get '.guardrails.gpu_concurrency.wait_timeout_seconds')"; lock_timeout="${lock_timeout:-$GPU_LOCK_TIMEOUT_DEFAULT}"
  lock_file="$(get '.guardrails.gpu_concurrency.lock_file')"

  local tools; tools="$(tools_export_prefix)"
  if [[ "$MODE" == "remote" ]]; then
    remote_ssh_args
    # Resolve remote lock path now that REMOTE_WORKDIR is known.
    if [[ -z "$lock_file" ]]; then
      lock_file="${REMOTE_WORKDIR}/.sycl/.gpu-run.lock"
    elif [[ "$lock_file" != /* ]]; then
      lock_file="${REMOTE_WORKDIR}/${lock_file}"
    fi
    do_sync
    local prefix=""
    [[ -n "${ENV_SETUP:-}" ]] && prefix="${ENV_SETUP}; "
    prefix="${prefix}$(proxy_export_prefix remote)${tools}$(env_export_prefix remote)$(deps_ensure_snippet)"
    # Serialize GPU-bound work on the remote runner. The lock is created/acquired there so multiple
    # local agents sharing one remote GPU cannot overload it. We wrap the command in a small bash
    # snippet that acquires the lock, runs the command, and releases it even on failure/kill.
    if [[ "$lock_enabled" == "true" ]]; then
      local lock_remote="$lock_file"
      local lock_t="$lock_timeout"
      prefix="${prefix}mkdir -p \"$(dirname "$lock_remote")\"; "
      prefix="${prefix}exec {__gpu_lock_fd}>\"$lock_remote\"; "
      prefix="${prefix}waited=0; while ! flock -n \$__gpu_lock_fd 2>/dev/null; do "
      prefix="${prefix}if (( waited == 0 )); then echo \"[gpu-lock remote] waiting: another GPU job holds $lock_remote\" >&2; fi; "
      prefix="${prefix}sleep 5; waited=\$((waited+5)); "
      prefix="${prefix}if (( waited > $lock_t )); then echo \"[gpu-lock remote] timeout after ${lock_t}s\" >&2; exit 124; fi; "
      prefix="${prefix}done; "
      prefix="${prefix}trap 'flock -u \$__gpu_lock_fd 2>/dev/null || true' EXIT; "
    fi
    echo "[remote ${REMOTE_DEST}] $cmd"
    local rc=0
    "${SSH[@]}" "cd ${REMOTE_WORKDIR} && ${prefix}${cmd}" || rc=$?
    # Bring newly-produced reports/logs and runner-generated state (e.g. golden reference tensors)
    # back so they are readable/auditable locally and survive the remote workdir being recreated.
    # Profiling artifacts under .sycl/reports/ are how humans audit runs. The pull runs even on
    # failure so partial logs are recoverable, and it can only add or back up — never clobber.
    do_pull
    return $rc
  else
    local es; es="$(local_env_setup)"
    local prefix=""
    [[ -n "$es" ]] && prefix="${es}; "
    prefix="${prefix}$(proxy_export_prefix local)${tools}$(env_export_prefix local)$(deps_ensure_snippet)"
    # Serialize GPU-bound work locally.
    if [[ "$lock_enabled" == "true" ]]; then
      lock_file="${lock_file:-$GPU_LOCK_FILE_DEFAULT}"
      mkdir -p "$(dirname "$lock_file")"
      GPU_LOCK_PATH="$lock_file"
      _gpu_lock_acquire "$GPU_LOCK_PATH" "$lock_timeout" || return 124
    fi
    echo "[local] $cmd"
    ( cd "$PROJECT_DIR" && bash -lc "${prefix}${cmd}" )
  fi
}

env_probe() {
  run_cmd 'echo "== host =="; hostname; echo "== icpx =="; (icpx --version 2>/dev/null | head -1) || echo "icpx: not found"; echo "== sycl-ls =="; (sycl-ls 2>/dev/null) || echo "sycl-ls: not found"; echo "== unitrace =="; if [ -x "$UNITRACE" ] || command -v "$UNITRACE" >/dev/null 2>&1; then echo "unitrace: $UNITRACE"; else echo "unitrace: not found ($UNITRACE)"; fi; echo "== python deps =="; python3 -c "import pandas,matplotlib;print(\"pandas\",pandas.__version__,\"matplotlib\",matplotlib.__version__)" 2>/dev/null || echo "pandas/matplotlib: MISSING (run: .sycl/scripts/run.sh setup)"'
}

# Force a (re)install of the sycl-agent Python deps on the ACTIVE runner, then refresh the marker so
# the automatic one-time ensure stays satisfied. Normally unnecessary — run_cmd auto-installs once on
# the first command — but useful to re-install after editing requirements.txt or to preinstall
# during setup. Routed through run_cmd (inherits runner + proxy + env); SYCL_SKIP_DEPS_ENSURE=1
# suppresses the auto-ensure so we don't install twice. Extra pip args via PIP_ARGS (e.g. "--user").
do_setup() {
  local pip_args="${PIP_ARGS:-}"
  echo "[setup] force-installing sycl-agent Python deps from .sycl/requirements.txt on ${MODE} runner"
  SYCL_SKIP_DEPS_ENSURE=1 run_cmd "test -f .sycl/requirements.txt || { echo 'run.sh setup: missing .sycl/requirements.txt (re-run install.sh)'; exit 1; }; python3 -m pip install ${pip_args} -r .sycl/requirements.txt && mkdir -p \"\$HOME/.cache/sycl-agent\" && : > \"\$HOME/.cache/sycl-agent/deps-${REQ_HASH}\""
}

cmd="${1:-}"; shift || true
case "$cmd" in
  env)             env_probe ;;
  setup)           do_setup ;;
  sync)            do_sync "$@" ;;
  pull)            do_pull ;;
  exec|build|test|profile|bench)
                   [[ $# -ge 1 ]] || { echo "run.sh $cmd needs a command string" >&2; exit 2; }
                   run_cmd "$*" ;;
  *) sed -n '2,24p' "$0"; exit "${cmd:+1}" ;;
esac
