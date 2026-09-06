#!/usr/bin/env bash
# Capture and audit EVIDENCE for every claim sycl-agent makes.
#
#   state/*.json says what the agent concluded.  evidence/ says why anyone should believe it.
#   The rule: NO CLAIM WITHOUT A RECORD.
#
# A run record is one command, on one machine, at one commit, with a verdict that comes from the
# process exit code — not from the agent's reading of the output. The raw stdout+stderr is kept
# byte-for-byte and hashed, so a log that is later edited or lost in a sync is detected rather than
# trusted. See .sycl/references/evidence-model.md.
#
# Usage:
#   evidence.sh env [--label L]                      snapshot the machine; prints the fingerprint id
#   evidence.sh record <class> <subject> [opts] -- <cmd...>
#                                                    run <cmd> through the runner, capture the record
#   evidence.sh annotate <run-id> --set k=v ...      add measurements parsed by the agent
#   evidence.sh attach   <run-id> --artifact PATH    attach + hash an extra artifact
#   evidence.sh show     <run-id>                    print a record (and where its log is)
#   evidence.sh list     [--class C] [--subject S]   list records
#   evidence.sh diagnosis <kernel-id>                build opt_<id>/DIAGNOSIS.md (baseline vs final)
#   evidence.sh manifest                             (re)hash every artifact into manifest.json
#   evidence.sh verify [--strict]                    mechanical audit; non-zero exit on any gap
#   evidence.sh audit                                regenerate .sycl/reports/AUDIT.md
#   evidence.sh enter <phase>                        stamp entry into a workflow phase
#   evidence.sh gate  <phase> [--dry-run]            THE PHASE-EXIT GATE: check that phase's
#                                                    completion criteria and mark it exited only
#                                                    if they are met. `verify` asks "is what you
#                                                    said true?"; `gate` asks "did you do the thing
#                                                    this phase exists to do?" — a phase skipped in
#                                                    silence passes verify and fails gate.
#
#   class:   test | bench | e2e | build | integration | profile
#   subject: kernel id, the e2e case label ("baseline"/"final"/"case-<name>"), or for
#            build/integration the gate subject ("backend", "runtime", "suite", "<surface>")
#
# record options:
#   --label L                 story point: baseline | final | migrate | trial-<n> | case-<name>
#   --phase P                 workflow phase (defaults to project.json .phase)
#   --shape S                 problem size for this run (one record per shape)
#   --shape-class C           model | general | edge
#   --dtype D  --seed N       input spec — with --shape this regenerates the exact inputs
#   --reference "..."         correctness oracle used, WITH its source path
#   --tolerance "rtol=..,atol=.."   the INHERITED tolerance
#   --tolerance-source "..."  where it was inherited from (proves it was not loosened)
#   --warmup N --iters N      benchmark policy actually used
#   --via VERB                run.sh verb (default per class); --local to bypass the runner
#   --artifact PATH           extra file this run produced (repeatable; hashed)
#   --set key=value           write into .measurements (dotted keys ok: tolerance.atol=1e-6)
#   --no-gate                 measurement-only run: verdict is "recorded", not pass/fail
#   --no-propagate            always exit 0 (default: exit with the command's own status)
#   --note TEXT
#
# Env knobs: SYCL_EVIDENCE_LOG_MAX_BYTES (default 2000000), SYCL_EVIDENCE_DIR
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # .sycl/
PROJECT_DIR="$(cd "$SYCL_DIR/.." && pwd)"
REPORTS="$SYCL_DIR/reports"
EVID="${SYCL_EVIDENCE_DIR:-$REPORTS/evidence}"
STATE="$SYCL_DIR/state"
RUN_SH="$SCRIPT_DIR/run.sh"
AUDIT_PY="$SCRIPT_DIR/evidence_audit.py"
LOG_MAX="${SYCL_EVIDENCE_LOG_MAX_BYTES:-2000000}"

die() { echo "evidence: $*" >&2; exit 2; }
now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stamp()   { date -u +%Y%m%dT%H%M%SZ; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum   >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else echo ""; fi
}
sha256_str() {
  if command -v sha256sum >/dev/null 2>&1; then printf '%s' "$1" | sha256sum | awk '{print $1}'
  elif command -v shasum   >/dev/null 2>&1; then printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
  else echo "0000000000000000"; fi
}
bytes_of() { wc -c < "$1" | tr -d ' '; }

need_python() { command -v python3 >/dev/null 2>&1 || die "python3 is required"; }

# Join argv into one shell command string, quoting only what needs it.
join_cmd() {
  local out="" a
  for a in "$@"; do
    if [[ "$a" =~ [[:space:]\'\"\\\$\`] ]]; then out+=" $(printf '%q' "$a")"; else out+=" $a"; fi
  done
  printf '%s' "${out# }"
}

git_commit() { (cd "$PROJECT_DIR" && git rev-parse --short HEAD 2>/dev/null) || echo ""; }
git_dirty()  {
  local s; s="$(cd "$PROJECT_DIR" && git status --porcelain 2>/dev/null)" || { echo "false"; return; }
  [[ -n "$s" ]] && echo "true" || echo "false"
}

json_get() {  # json_get <file> <dotted.path> ; empty string when absent
  need_python
  python3 - "$1" "$2" <<'PY' 2>/dev/null || true
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for k in sys.argv[2].split('.'):
    if isinstance(d,dict) and k in d: d=d[k]
    else: sys.exit(0)
print(d if not isinstance(d,(dict,list)) else json.dumps(d))
PY
}

# ------------------------------------------------------------------ env fingerprint --------------
# A benchmark number without its machine state is not evidence. The fingerprint is content-hashed,
# so two runs on the same machine in the same state share an id and become comparable; a driver
# upgrade or an unpinned clock changes the id and the audit sees the comparison is invalid.
cmd_env() {
  local label=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --label) label="$2"; shift 2 ;;
      *) die "env: unknown option $1" ;;
    esac
  done
  need_python
  mkdir -p "$EVID/env"

  local raw=""
  if [[ -x "$RUN_SH" ]]; then raw="$(bash "$RUN_SH" env 2>&1)"; fi
  [[ -n "$raw" ]] || raw="$(uname -a 2>&1)"

  local freq mode host icpx gpu arch platform
  freq="$(json_get "$STATE/project.json" env.freq_pinned)"
  mode="$(json_get "$SYCL_DIR/config.json" runner.mode)"
  host="$(json_get "$SYCL_DIR/config.json" runner.remote.host)"
  icpx="$(json_get "$STATE/project.json" env.icpx)"
  gpu="$(json_get "$STATE/project.json" env.gpu)"
  arch="$(json_get "$STATE/project.json" env.arch)"
  platform="$(json_get "$STATE/project.json" env.platform)"

  # Hash the SUBSTANCE only: drop timestamps/paths/pids so an unchanged machine keeps its id.
  local norm
  norm="$(printf '%s\n' "$raw" | sed -E 's/[0-9]{4}-[0-9]{2}-[0-9]{2}T?[0-9: ]*//g; s/pid[= ][0-9]+//gi; s/\/tmp\/[^ ]*//g')"
  local fid; fid="env-$(sha256_str "${norm}|${icpx}|${gpu}|${arch}|${platform}|${freq}" | cut -c1-8)"
  local out="$EVID/env/$fid.json"

  RAW="$raw" python3 - "$out" "$fid" "$(now_utc)" "$label" "$mode" "$host" "$icpx" "$gpu" "$arch" "$platform" "$freq" <<'PY'
import json,os,sys
out,fid,ts,label,mode,host,icpx,gpu,arch,platform,freq = sys.argv[1:12]
def b(v):
    if v in ("true","True"): return True
    if v in ("false","False"): return False
    return v or "unknown"
rec = {
  "schema":"sycl-agent/env-fingerprint","id":fid,"first_seen":ts,"last_seen":ts,"label":label,
  "runner":{"mode":mode or "local","host":host},
  "icpx":icpx or "unknown","gpu":gpu or "unknown","arch":arch or "unknown",
  "platform":platform or "unknown","freq_pinned":b(freq),
  "raw":os.environ.get("RAW",""),
}
if os.path.exists(out):
    try:
        old=json.load(open(out)); rec["first_seen"]=old.get("first_seen",ts)
    except Exception: pass
os.makedirs(os.path.dirname(out),exist_ok=True)
json.dump(rec,open(out,"w"),indent=2); open(out,"a").write("\n")
PY

  echo "$fid"
}

current_env_fingerprint() {
  # Reuse the newest fingerprint if one exists; only probe the machine when there is none.
  local latest
  latest="$(ls -1t "$EVID/env"/env-*.json 2>/dev/null | head -1)"
  if [[ -n "$latest" ]]; then basename "$latest" .json; else cmd_env; fi
}

# ------------------------------------------------------------------ record ------------------------
cmd_record() {
  [[ $# -ge 2 ]] || die "usage: evidence.sh record <class> <subject> [opts] -- <cmd...>"
  local class="$1" subject="$2"; shift 2
  case "$class" in test|bench|e2e|build|integration|profile) ;; *) die "invalid class '$class' (test|bench|e2e|build|integration|profile)" ;; esac

  local label="" phase="" via="" local_only=0 gate=1 propagate=1 note=""
  local -a artifacts=() sets=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --label) label="$2"; shift 2 ;;
      --phase) phase="$2"; shift 2 ;;
      --shape) sets+=("shape=$2"); shift 2 ;;
      --shape-class) sets+=("shape_class=$2"); shift 2 ;;
      --dtype) sets+=("dtype=$2"); shift 2 ;;
      --seed)  sets+=("seed=$2"); shift 2 ;;
      --reference) sets+=("reference=$2"); shift 2 ;;
      --tolerance) sets+=("__tolerance=$2"); shift 2 ;;
      --tolerance-source) sets+=("tolerance.source=$2"); shift 2 ;;
      --warmup) sets+=("warmup=$2"); shift 2 ;;
      --iters)  sets+=("iters=$2"); shift 2 ;;
      --via) via="$2"; shift 2 ;;
      --local) local_only=1; shift ;;
      --artifact) artifacts+=("$2"); shift 2 ;;
      --set) sets+=("$2"); shift 2 ;;
      --no-gate) gate=0; shift ;;
      --no-propagate) propagate=0; shift ;;
      --note) note="$2"; shift 2 ;;
      --) shift; break ;;
      *) die "record: unknown option $1" ;;
    esac
  done
  [[ $# -ge 1 ]] || die "record: missing command after --"
  need_python

  local cmd; cmd="$(join_cmd "$@")"
  [[ -n "$phase" ]] || phase="$(json_get "$STATE/project.json" phase)"

  # Default routing per class. Everything goes through run.sh so evidence is captured on the machine
  # that did the work — and, living under .sycl/reports/, is pulled back automatically.
  if [[ -z "$via" ]]; then
    case "$class" in
      test)  via="test" ;;
      bench) via="bench" ;;
      build) via="build" ;;
      profile) via="profile" ;;
      e2e)   via="exec" ;;
    esac
  fi

  local envfp; envfp="$(current_env_fingerprint)"
  local commit dirty; commit="$(git_commit)"; dirty="$(git_dirty)"

  local safe_subject; safe_subject="$(printf '%s' "$subject" | tr -c 'A-Za-z0-9._-' '_')"
  local dir="$EVID/$class/$safe_subject"; mkdir -p "$dir"
  local rid="$class-$safe_subject-$(stamp)-$(sha256_str "$cmd$(date +%s%N)" | cut -c1-6)"
  local logf="$dir/$rid.log" recf="$dir/$rid.json"

  local t0 t1 rc
  t0="$(date +%s.%N)"
  if [[ $local_only -eq 1 ]]; then
    ( cd "$PROJECT_DIR" && bash -lc "$cmd" ) >"$logf" 2>&1
    rc=$?
  else
    [[ -x "$RUN_SH" || -f "$RUN_SH" ]] || die "runner not found at $RUN_SH (use --local to bypass)"
    bash "$RUN_SH" "$via" "$cmd" >"$logf" 2>&1
    rc=$?
  fi
  t1="$(date +%s.%N)"

  # Cap runaway logs, but keep BOTH ends: the head holds the config banner, the tail holds the verdict.
  local truncated=false
  if [[ "$(bytes_of "$logf")" -gt "$LOG_MAX" ]]; then
    local half=$((LOG_MAX / 2))
    { head -c "$half" "$logf"; printf '\n\n...[evidence: log truncated, middle elided]...\n\n'; tail -c "$half" "$logf"; } > "$logf.tmp"
    mv "$logf.tmp" "$logf"; truncated=true
  fi

  local dur; dur="$(python3 -c "print(round(float('$t1')-float('$t0'),3))" 2>/dev/null || echo 0)"
  local lsha lbytes; lsha="$(sha256_of "$logf")"; lbytes="$(bytes_of "$logf")"

  local verdict
  if [[ $gate -eq 0 ]]; then verdict="recorded"
  elif [[ $rc -eq 0 ]]; then verdict="pass"
  else verdict="fail"; fi

  local mode host workdir
  mode="$(json_get "$SYCL_DIR/config.json" runner.mode)"
  host="$(json_get "$SYCL_DIR/config.json" runner.remote.host)"
  workdir="$(json_get "$SYCL_DIR/config.json" runner.remote.workdir)"

  local relog="${logf#"$REPORTS"/}"
  EV_SETS="$(printf '%s\n' "${sets[@]+"${sets[@]}"}")" \
  EV_ARTIFACTS="$(printf '%s\n' "${artifacts[@]+"${artifacts[@]}"}")" \
  python3 - "$recf" "$logf" "$rid" "$class" "$subject" "$label" "$phase" "$(now_utc)" "$dur" \
             "$cmd" "$via" "$local_only" "$mode" "$host" "$workdir" "$envfp" "$commit" "$dirty" \
             "$rc" "$verdict" "$relog" "$lsha" "$lbytes" "$truncated" "$note" "$REPORTS" <<'PY'
import hashlib, json, os, re, sys

(recf, logf, rid, cls, subject, label, phase, ts, dur, cmd, via, local_only, mode, host, workdir,
 envfp, commit, dirty, rc, verdict, relog, lsha, lbytes, truncated, note, reports) = sys.argv[1:27]

text = open(logf, errors="replace").read()

# --- auto-extraction -----------------------------------------------------------------------------
# Lift the numbers a reviewer would look for out of the raw log, so the common case needs no manual
# annotation. Advisory only: everything here is also readable in the log itself.
extracted, meas = {}, {}
num = r"([-+0-9][0-9.eE+-]*)"
patterns = {
    "max_abs_err":  rf"max[\s_]*abs(?:olute)?[\s_]*(?:err(?:or)?|diff)\s*[:=]?\s*{num}",
    "max_rel_err":  rf"max[\s_]*rel(?:ative)?[\s_]*(?:err(?:or)?|diff)\s*[:=]?\s*{num}",
    "mismatch_count": rf"(?:mismatch(?:e[sd])?|failures?)\s*[:=]?\s*([0-9]+)",
    "median_ms":    rf"median(?:_ms)?\s*[:=]?\s*{num}",
    "p90_ms":       rf"p90(?:_ms)?\s*[:=]?\s*{num}",
    "stddev_ms":    rf"(?:stddev|std)(?:_ms)?\s*[:=]?\s*{num}",
    "min_ms":       rf"min(?:_ms)?\s*[:=]?\s*{num}",
    "bandwidth_gbps": rf"([0-9.]+)\s*GB/s",
    "tflops":       rf"([0-9.]+)\s*TFLOP",
}
for key, pat in patterns.items():
    m = re.findall(pat, text, re.IGNORECASE)
    if m:
        try: meas[key] = float(m[-1])
        except ValueError: pass
tol = {}
for k, pat in (("rtol", rf"rtol\s*[:=]\s*{num}"), ("atol", rf"atol\s*[:=]\s*{num}")):
    m = re.findall(pat, text, re.IGNORECASE)
    if m:
        try: tol[k] = float(m[-1])
        except ValueError: pass
if tol: meas["tolerance"] = tol

# bench.sh (and any driver following it) prints a JSON stats line — take it verbatim.
for line in reversed(text.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}") and "ms" in line:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                extracted["stats_json"] = obj
                for k, v in obj.items():
                    if isinstance(v, (int, float)) and k not in meas: meas[k] = v
            break
        except Exception: pass

hits = re.findall(r"\b(PASS(?:ED)?|FAIL(?:ED)?|OK|ERROR)\b", text)
if hits: extracted["verdict_tokens"] = hits[-5:]

# --- explicit --set / typed options (authoritative over auto-extraction) -------------------------
def assign(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]: d = d.setdefault(k, {})
    try:
        v = json.loads(value)
        if not isinstance(v, (int, float, bool, list, dict)): v = value
    except Exception:
        v = value
    d[keys[-1]] = v

for raw in (os.environ.get("EV_SETS") or "").splitlines():
    raw = raw.strip()
    if not raw or "=" not in raw: continue
    k, v = raw.split("=", 1)
    if k == "__tolerance":                       # "rtol=1e-5,atol=1e-6"
        for part in v.split(","):
            if "=" in part:
                tk, tv = part.split("=", 1)
                assign(meas, f"tolerance.{tk.strip()}", tv.strip())
    else:
        assign(meas, k, v)

# --- artifacts -----------------------------------------------------------------------------------
def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

arts = []
for p in (os.environ.get("EV_ARTIFACTS") or "").splitlines():
    p = p.strip()
    if not p: continue
    full = p if os.path.isabs(p) else os.path.join(os.getcwd(), p)
    if not os.path.exists(full):
        arts.append({"path": p, "missing": True}); continue
    if os.path.isdir(full):
        arts.append({"path": os.path.relpath(full, reports), "kind": "dir",
                     "bytes": sum(os.path.getsize(os.path.join(r, f))
                                  for r, _, fs in os.walk(full) for f in fs)})
    else:
        arts.append({"path": os.path.relpath(full, reports), "kind": os.path.splitext(full)[1].lstrip(".") or "other",
                     "sha256": digest(full), "bytes": os.path.getsize(full)})

rec = {
    "schema": "sycl-agent/run-record",
    "run_id": rid, "class": cls, "subject": subject,
    "label": label or None, "phase": phase or None,
    "ts": ts, "duration_s": float(dur or 0),
    "command": cmd, "via": ("local" if local_only == "1" else f"run.sh {via}"),
    "runner": ({"mode": "local"} if (local_only == "1" or mode != "remote")
               else {"mode": "remote", "host": host or None, "workdir": workdir or None}),
    "env_fingerprint": envfp or None,
    "commit": commit or None, "dirty": dirty == "true",
    "exit_code": int(rc), "verdict": verdict,
    "log": relog, "log_sha256": lsha, "log_bytes": int(lbytes),
    "log_truncated": truncated == "true",
    "artifacts": arts, "measurements": meas, "extracted": extracted,
    "notes": note or None,
}
rec = {k: v for k, v in rec.items() if v is not None}
json.dump(rec, open(recf, "w"), indent=2, sort_keys=False); open(recf, "a").write("\n")
PY

  # The single line the agent copies into state as the evidence pointer.
  echo "run_id: $rid"
  echo "record: ${recf#"$SYCL_DIR"/}"
  echo "log:    ${logf#"$SYCL_DIR"/}"
  echo "verdict: $verdict (exit $rc) commit=${commit:-none}${dirty:+ dirty=$dirty} env=$envfp"

  [[ $propagate -eq 1 ]] && return "$rc"
  return 0
}

find_record() {  # find_record <run-id> -> path
  local rid="$1" p
  p="$(find "$EVID" -name "$rid.json" -print -quit 2>/dev/null)"
  [[ -n "$p" ]] || die "no record for run id '$rid'"
  printf '%s' "$p"
}

cmd_annotate() {
  [[ $# -ge 1 ]] || die "usage: evidence.sh annotate <run-id> --set k=v [--set k=v ...]"
  local rid="$1"; shift
  local -a sets=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --set) sets+=("$2"); shift 2 ;;
      --note) sets+=("__note=$2"); shift 2 ;;
      *) die "annotate: unknown option $1" ;;
    esac
  done
  need_python
  local recf; recf="$(find_record "$rid")"
  EV_SETS="$(printf '%s\n' "${sets[@]+"${sets[@]}"}")" python3 - "$recf" <<'PY'
import json,os,sys
recf=sys.argv[1]; rec=json.load(open(recf))
meas=rec.setdefault("measurements",{})
def assign(d,dotted,value):
    keys=dotted.split(".")
    for k in keys[:-1]: d=d.setdefault(k,{})
    try:
        v=json.loads(value)
        if not isinstance(v,(int,float,bool,list,dict)): v=value
    except Exception: v=value
    d[keys[-1]]=v
for raw in (os.environ.get("EV_SETS") or "").splitlines():
    raw=raw.strip()
    if not raw or "=" not in raw: continue
    k,v=raw.split("=",1)
    if k=="__note": rec["notes"]=v
    else: assign(meas,k,v)
json.dump(rec,open(recf,"w"),indent=2); open(recf,"a").write("\n")
print("annotated:",rec["run_id"])
PY
}

cmd_attach() {
  [[ $# -ge 1 ]] || die "usage: evidence.sh attach <run-id> --artifact PATH [--artifact PATH ...]"
  local rid="$1"; shift
  local -a arts=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --artifact) arts+=("$2"); shift 2 ;;
      *) die "attach: unknown option $1" ;;
    esac
  done
  need_python
  local recf; recf="$(find_record "$rid")"
  EV_ARTIFACTS="$(printf '%s\n' "${arts[@]+"${arts[@]}"}")" python3 - "$recf" "$REPORTS" <<'PY'
import hashlib,json,os,sys
recf,reports=sys.argv[1],sys.argv[2]; rec=json.load(open(recf))
def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as fh:
        for c in iter(lambda: fh.read(1<<20), b""): h.update(c)
    return h.hexdigest()
arts=rec.setdefault("artifacts",[])
have={a.get("path") for a in arts}
for p in (os.environ.get("EV_ARTIFACTS") or "").splitlines():
    p=p.strip()
    if not p: continue
    full=p if os.path.isabs(p) else os.path.abspath(p)
    rel=os.path.relpath(full,reports)
    if rel in have: continue
    if not os.path.exists(full): arts.append({"path":rel,"missing":True}); continue
    if os.path.isdir(full):
        arts.append({"path":rel,"kind":"dir","bytes":sum(os.path.getsize(os.path.join(r,f))
                     for r,_,fs in os.walk(full) for f in fs)})
    else:
        arts.append({"path":rel,"kind":os.path.splitext(full)[1].lstrip(".") or "other",
                     "sha256":digest(full),"bytes":os.path.getsize(full)})
json.dump(rec,open(recf,"w"),indent=2); open(recf,"a").write("\n")
print("attached:",len(arts),"artifact(s) on",rec["run_id"])
PY
}

cmd_show() {
  [[ $# -ge 1 ]] || die "usage: evidence.sh show <run-id>"
  local recf; recf="$(find_record "$1")"
  cat "$recf"
  echo "--- raw log: ${recf%.json}.log"
}

cmd_list() {
  local class="" subject=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --class) class="$2"; shift 2 ;;
      --subject) subject="$2"; shift 2 ;;
      *) die "list: unknown option $1" ;;
    esac
  done
  need_python
  python3 "$AUDIT_PY" list --sycl-dir "$SYCL_DIR" ${class:+--class "$class"} ${subject:+--subject "$subject"}
}

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

main() {
  local sub="${1:-}"; [[ $# -gt 0 ]] && shift
  case "$sub" in
    env)        cmd_env "$@" ;;
    record)     cmd_record "$@" ;;
    annotate)   cmd_annotate "$@" ;;
    attach)     cmd_attach "$@" ;;
    show)       cmd_show "$@" ;;
    list)       cmd_list "$@" ;;
    manifest|verify|audit|diagnosis|gate|enter)
                need_python
                [[ -f "$AUDIT_PY" ]] || die "evidence_audit.py not found at $AUDIT_PY"
                python3 "$AUDIT_PY" "$sub" --sycl-dir "$SYCL_DIR" "$@" ;;
    ""|-h|--help|help) usage ;;
    *) die "unknown subcommand '$sub' (env|record|annotate|attach|show|list|diagnosis|manifest|verify|audit|enter|gate)" ;;
  esac
}

main "$@"
