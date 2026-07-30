#!/usr/bin/env bash
# Probe the Intel GPU toolchain on THIS host (run via `.sycl/scripts/run.sh exec` to probe remote).
# Prints simple key=value lines the agent can parse into project.json.env.
set -uo pipefail

kv() { printf '%s=%s\n' "$1" "$2"; }

kv host "$(hostname 2>/dev/null || echo unknown)"

if command -v icpx >/dev/null 2>&1; then
  kv icpx_available true
  kv icpx_version "$(icpx --version 2>/dev/null | head -1)"
else
  kv icpx_available false
fi

kv oneapi_env "${ONEAPI_ROOT:-${SETVARS_COMPLETED:-unset}}"

if command -v sycl-ls >/dev/null 2>&1; then
  kv sycl_ls_present true
  # First GPU device name.
  gpu="$(sycl-ls 2>/dev/null | grep -iE 'gpu' | head -1 | sed 's/^[[:space:]]*//')"
  kv gpu_device "${gpu:-unknown}"
else
  kv sycl_ls_present false
fi

# Platform + arch guess from device name/id. Emits detected_platform (b60|b70|cri|unknown) and
# detected_arch (xe2|xe3p|unknown). The agent compares detected_platform against target.platform in
# config.json and applies target.on_mismatch. Extend the patterns as new devices ship.
g="$(echo "${gpu:-}" | tr 'A-Z' 'a-z')"
if   echo "$g" | grep -qE 'b70|0xe223|bmg-g31';                    then plat=b70; arch=xe2
elif echo "$g" | grep -qE 'b60|0xe222|bmg-g21';                    then plat=b60; arch=xe2
elif echo "$g" | grep -qE 'crescent|cri|xe3p|xe3';                 then plat=cri; arch=xe3p
elif echo "$g" | grep -qE 'b[0-9]{2,3}|battlemage|arc pro';        then plat=unknown; arch=xe2
else                                                                    plat=unknown; arch=unknown
fi
kv detected_platform "$plat"
kv detected_arch "$arch"

# Configured tool paths (exported by run.sh from .sycl/config.json .tools) override PATH lookup.
have_tool() { local t="$1"; [[ -n "$t" ]] && { [[ -x "$t" ]] || command -v "$t" >/dev/null 2>&1; }; }
have_tool "${UNITRACE:-unitrace}" && kv unitrace_available true || kv unitrace_available false
have_tool "${VTUNE:-vtune}"       && kv vtune_available true    || kv vtune_available false
have_tool "${XPU_SMI:-xpu-smi}"   && kv xpu_smi_available true   || kv xpu_smi_available false
kv unitrace_path "${UNITRACE:-unitrace}"
kv unitrace_home "${UNITRACE_HOME:-}"

# Frequency pinning status (best-effort via xpu-smi).
XPU_SMI_BIN="${XPU_SMI:-xpu-smi}"
if have_tool "$XPU_SMI_BIN"; then
  frange="$("$XPU_SMI_BIN" config -d 0 -t 0 2>/dev/null | grep -iE 'frequency' | head -2 | tr '\n' ' ')"
  kv gpu_freq_info "${frange:-unknown}"
fi
