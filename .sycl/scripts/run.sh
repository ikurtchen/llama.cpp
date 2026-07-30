#!/usr/bin/env bash
# Execution dispatcher: run build/test/profile/bench/env commands locally or on a remote
# Intel GPU host, per .sycl/config.json. The agent NEVER calls icpx/unitrace/tests directly —
# it always goes through this wrapper so the same workflow works with or without a local GPU.
#
# Usage:
#   run.sh env                      # print detected toolchain/GPU on the runner
#   run.sh setup                    # force-(re)install sycl-agent Python deps (auto-runs once otherwise)
#   run.sh sync                     # push the project to the remote workdir (no-op when local)
#   run.sh pull                     # fetch remote-generated .sycl/reports + logs + state back (no-op when local)
#   run.sh exec  "<shell command>"  # run an arbitrary command on the runner (in project dir)
#   run.sh build "<build command>"  # convenience alias for exec, tagged as a build
#   run.sh test  "<test command>"
#   run.sh profile "<profile command>"
#   run.sh bench "<bench command>"
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
# hash, under $HOME/.cache/sycl-agent so the workdir's rsync --delete can't wipe it) short-circuits
# the check to a single `[ -f ]`. Non-fatal: a failed install (e.g. offline) does NOT create the
# marker (so it retries next run) and never aborts the actual command. Suppressed when
# SYCL_SKIP_DEPS_ENSURE=1 (used by `run.sh setup`, which force-installs explicitly).
deps_ensure_snippet() {
  [[ "${SYCL_SKIP_DEPS_ENSURE:-0}" == 1 ]] && return 0
  local pip_args="${PIP_ARGS:-}"
  printf '{ __sdreq=.sycl/requirements.txt; __sdm="$HOME/.cache/sycl-agent/deps-%s"; if [ -f "$__sdreq" ] && [ ! -f "$__sdm" ]; then echo "[setup] installing sycl-agent Python deps (one-time)" >&2; if python3 -m pip install -q %s -r "$__sdreq"; then mkdir -p "$(dirname "$__sdm")" && : > "$__sdm"; else echo "run.sh: sycl-agent dep install failed (offline/proxy?); will retry next run" >&2; fi; fi; }; ' \
    "$REQ_HASH" "$pip_args"
}

do_sync() {
  [[ "$MODE" == "remote" ]] || { echo "[local] sync: no-op"; return 0; }
  remote_ssh_args
  echo "[remote] sync -> ${REMOTE_DEST}:${REMOTE_WORKDIR}"
  "${SSH[@]}" "mkdir -p ${REMOTE_WORKDIR}"
  # Push local -> remote, excluding heavy/transient dirs. .sycl/ IS synced (state travels with it),
  # but .sycl/reports/ and .sycl/logs/ are produced ON THE REMOTE, so they are excluded here: with
  # --delete an included-but-locally-absent path would be wiped on the remote, destroying the very
  # profiling artifacts / logs we later want to pull back. Fetch them with `run.sh pull`.
  rsync -az --delete \
    -e "$RSYNC_SSH" \
    --exclude '.git/' --exclude 'build/' --exclude 'build_sycl/' --exclude '*.o' \
    --exclude '.sycl/logs/' --exclude '.sycl/reports/' \
    "$PROJECT_DIR/" "${REMOTE_DEST}:${REMOTE_WORKDIR}/"
}

# Pull remote-generated outputs back to the local project so the agent and a human can read/audit
# them. Remote -> local, WITHOUT --delete (never destroy local history). Covers three classes:
#   .sycl/reports/  profiling artifacts (metric CSVs, stall CSVs, IGC shader dumps)
#   .sycl/logs/     any logs written on the runner
#   .sycl/state/kernels/<id>/ref/  golden reference tensors are generated ON THE RUNNER (by running
#                   the CUDA/CPU reference) yet live under state/, which the push mirrors with
#                   --delete. We pull back ONLY the ref/ trees (not the authored *.json), so a
#                   locally-authored state file can never be clobbered — no mtime/--update guessing.
do_pull() {
  [[ "$MODE" == "remote" ]] || { echo "[local] pull: no-op"; return 0; }
  remote_ssh_args
  echo "[remote] pull <- ${REMOTE_DEST}:${REMOTE_WORKDIR} (.sycl/reports, .sycl/logs, state ref/)"
  mkdir -p "$PROJECT_DIR/.sycl/reports" "$PROJECT_DIR/.sycl/logs" "$PROJECT_DIR/.sycl/state"
  # Trailing '/' on sources copies contents into the matching local dir. Missing remote dirs are
  # tolerated (rsync warns but we don't fail the run over absent optional output).
  rsync -az \
    -e "$RSYNC_SSH" \
    "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/reports/" "$PROJECT_DIR/.sycl/reports/" 2>/dev/null || true
  rsync -az \
    -e "$RSYNC_SSH" \
    "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/logs/" "$PROJECT_DIR/.sycl/logs/" 2>/dev/null || true
  # Only the kernels/<id>/ref/ golden data — include the dirs that lead to it, take everything under
  # ref/, and exclude all else so authored state/*.json is never touched.
  rsync -az \
    -e "$RSYNC_SSH" \
    --include='kernels/' --include='kernels/*/' \
    --include='kernels/*/ref/' --include='kernels/*/ref/***' \
    --exclude='*' \
    "${REMOTE_DEST}:${REMOTE_WORKDIR}/.sycl/state/" "$PROJECT_DIR/.sycl/state/" 2>/dev/null || true
}

run_cmd() { # run_cmd "<shell command>"
  local cmd="$1"
  local tools; tools="$(tools_export_prefix)"
  if [[ "$MODE" == "remote" ]]; then
    remote_ssh_args
    do_sync
    local prefix=""
    [[ -n "${ENV_SETUP:-}" ]] && prefix="${ENV_SETUP}; "
    prefix="${prefix}$(proxy_export_prefix remote)${tools}$(env_export_prefix remote)$(deps_ensure_snippet)"
    echo "[remote ${REMOTE_DEST}] $cmd"
    local rc=0
    "${SSH[@]}" "cd ${REMOTE_WORKDIR} && ${prefix}${cmd}" || rc=$?
    # Bring newly-produced reports/logs and runner-generated state (e.g. golden reference tensors)
    # back so they survive the next push's --delete and are readable locally. Profiling artifacts
    # under .sycl/reports/ are how humans audit runs. The pull runs even on failure so partial logs
    # are recoverable.
    do_pull
    return $rc
  else
    local es; es="$(local_env_setup)"
    local prefix=""
    [[ -n "$es" ]] && prefix="${es}; "
    prefix="${prefix}$(proxy_export_prefix local)${tools}$(env_export_prefix local)$(deps_ensure_snippet)"
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
  sync)            do_sync ;;
  pull)            do_pull ;;
  exec|build|test|profile|bench)
                   [[ $# -ge 1 ]] || { echo "run.sh $cmd needs a command string" >&2; exit 2; }
                   run_cmd "$*" ;;
  *) sed -n '2,21p' "$0"; exit "${cmd:+1}" ;;
esac
