#!/usr/bin/env bash
# Agent analytics — collect AI-agent EFFICIENCY & COST metrics and roll them up for the
# dashboard and the final report:
#   • wall-clock TIME per phase (agent reasoning + tool/runner time), and
#   • COST per phase — token usage (BYOM) and/or GitHub AI credits / premium requests (Copilot).
#
# Two stores, same pattern as the rest of the framework:
#   .sycl/logs/metrics.jsonl   append-only event log (audit; one JSON object per line)
#   .sycl/state/metrics.json   authoritative rollup (totals + per-phase breakdown), regenerated
#                              from the event log after every write — so it is resumable and never
#                              drifts, exactly like PROGRESS.md is regenerated from state.
#
# Usage:
#   metrics.sh start <phase>                 open a timing interval for <phase>
#   metrics.sh stop  <phase>                 close it and record the elapsed seconds
#   metrics.sh usage <phase> [flags]         record cost for work done in <phase>:
#       --tokens-in N   --tokens-out N       uncached prompt / completion tokens (BYOM, token-billed)
#       --cache-read N  --cache-write N      cached input tokens read / written — priced differently
#                                            by some vendors (e.g. Anthropic: cache-write ~1.25×,
#                                            cache-read ~0.1× of base input). Kept separate so a
#                                            $-cost can be computed with the right per-class rate.
#       --requests N                         GitHub Copilot premium requests
#       --credits X                          GitHub AI credits (may be fractional)
#       --cost-usd X                         direct dollar cost, if your agent/tool reports one
#                                            (best signal — an exact price beats token estimates)
#       --model NAME    --kernel ID          optional attribution
#       --note "text"
#   metrics.sh import <usage.json>           fold in cost/usage recorded OUT-OF-BAND by the IDE/engine
#                                            (Copilot/Trae expose usage in their own UI/logs, not to the
#                                            agent) — normalize an export to a simple JSON shape and
#                                            ingest it, instead of the model self-reporting. See cmd_import.
#                                            Rows carry a `uid`, so re-importing never double-counts.
#                                            `collect-usage.sh` produces this file automatically by
#                                            reading the engine's own session store — prefer it over
#                                            self-reporting whenever the engine keeps a session log.
#   metrics.sh rollup                        recompute metrics.json from the event log
#   metrics.sh summary                       print totals + a per-phase table
#
# Timing: the rollup reports TWO durations. `session.active_wall_seconds` is the sum of bracketed
# start/stop intervals (agent reasoning + tool time). `session.elapsed_wall_seconds` is the true span
# from the first to the last event in the log — it also covers everything the agent never bracketed
# (human approval/review, model latency, idle between turns). On interactive IDEs (Copilot/Trae) the
# active sum badly undercounts real time, so elapsed is the faithful duration. Both are derived from
# timestamps and need no cooperation from the agent.
#
# Cross-check: the rollup also reads gen-progress's heartbeat trail (logs/progress.jsonl), which is
# appended after EVERY state change — far more reliably than the agent brackets phases. Those
# timestamps ANCHOR `elapsed_wall_seconds` (so it is correct even when no start/stop was called) and
# are reported independently as `session.observed_span_seconds` (first→last heartbeat) and
# `session.observed_work_seconds` (gaps summed, capped at SYCL_PROGRESS_IDLE_CAP, default 1800s, to
# exclude idle). `summary` warns when the bracketed elapsed is far below the observed span.
#
# Token/credit/$ counts are self-reported by the agent, because only the model/engine sees its own
# usage — record whatever your engine exposes; every cost field is optional. When the engine does NOT
# feed usage back to the agent (Copilot/Trae report tokens only in their own UI/logs), do NOT fabricate
# counts: leave them 0 and instead fold the real numbers in later with `metrics.sh import`. Prefer
# --cost-usd when a real dollar figure is available; else record tokens (split cached vs uncached)
# and/or credits. The event log is append-only, so a resumed run keeps accumulating.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"     # .sycl/
STATE="$SYCL_DIR/state"
LOG_DIR="$SYCL_DIR/logs"
EVENTS="$LOG_DIR/metrics.jsonl"
PROGRESS="$LOG_DIR/progress.jsonl"           # gen-progress heartbeat trail (independent timing cross-check)
ROLLUP="$STATE/metrics.json"
OPEN="$STATE/.metrics-open"                  # transient marker for the currently-open interval

# Canonical phase order (matches the workflow state machine); the rollup pre-seeds these so the
# per-phase table is stable, and still accepts any extra phase key that shows up in the log.
PHASES="detect inventory scaffold migrate integrate profile-e2e optimize done report"

mkdir -p "$LOG_DIR" "$STATE"

iso()   { date -u +%Y-%m-%dT%H:%M:%SZ; }
epoch() { date -u +%s; }
esc()   { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
append_event() { printf '%s\n' "$1" >> "$EVENTS"; }

cmd_start() {
  local phase="${1:?usage: metrics.sh start <phase>}"
  # Close any stale open interval first (e.g. a phase that never called stop before a resume/crash)
  # so its elapsed time is not lost and intervals never nest.
  if [[ -f "$OPEN" ]]; then
    local sp se now; IFS=$'\t' read -r sp se < "$OPEN"; now="$(epoch)"
    append_event "{\"ts\":\"$(iso)\",\"event\":\"phase\",\"phase\":\"$(esc "$sp")\",\"seconds\":$(( now - se )),\"note\":\"auto-closed on start\"}"
  fi
  printf '%s\t%s\n' "$phase" "$(epoch)" > "$OPEN"
  cmd_rollup
}

cmd_stop() {
  local phase="${1:?usage: metrics.sh stop <phase>}"
  if [[ ! -f "$OPEN" ]]; then
    echo "metrics.sh: no open interval to stop (phase=$phase) — call 'start $phase' first" >&2
    return 0
  fi
  local sp se now secs; IFS=$'\t' read -r sp se < "$OPEN"
  now="$(epoch)"; secs=$(( now - se ))
  append_event "{\"ts\":\"$(iso)\",\"event\":\"phase\",\"phase\":\"$(esc "$phase")\",\"seconds\":$secs,\"started\":$se,\"ended\":$now}"
  rm -f "$OPEN"
  cmd_rollup
}

cmd_usage() {
  local phase="${1:?usage: metrics.sh usage <phase> [flags]}"; shift || true
  local ti=0 to=0 crd=0 cwr=0 req=0 cr=0 usd=0 model="" kernel="" note=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tokens-in)   ti="${2:?}";     shift 2 ;;
      --tokens-out)  to="${2:?}";     shift 2 ;;
      --cache-read)  crd="${2:?}";    shift 2 ;;
      --cache-write) cwr="${2:?}";    shift 2 ;;
      --requests)    req="${2:?}";    shift 2 ;;
      --credits)     cr="${2:?}";     shift 2 ;;
      --cost-usd)    usd="${2:?}";    shift 2 ;;
      --model)       model="${2:?}";  shift 2 ;;
      --kernel)      kernel="${2:?}"; shift 2 ;;
      --note)        note="${2:?}";   shift 2 ;;
      *) echo "metrics.sh usage: unknown flag '$1'" >&2; return 2 ;;
    esac
  done
  # tokens_total counts everything processed (uncached in + cached read/write + out) so the headline
  # token figure is complete; the per-class fields remain available for accurate $-costing.
  local tt=$(( ti + to + crd + cwr ))
  append_event "{\"ts\":\"$(iso)\",\"event\":\"usage\",\"phase\":\"$(esc "$phase")\",\"tokens_in\":$ti,\"tokens_out\":$to,\"cache_read\":$crd,\"cache_write\":$cwr,\"tokens_total\":$tt,\"requests\":$req,\"credits\":$cr,\"cost_usd\":$usd,\"model\":\"$(esc "$model")\",\"kernel\":\"$(esc "$kernel")\",\"note\":\"$(esc "$note")\"}"
  cmd_rollup
}

cmd_rollup() {
  python3 - "$EVENTS" "$ROLLUP" "$PHASES" "$PROGRESS" <<'PY'
import json, os, sys, datetime
events_path, rollup_path, phases_str, progress_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
phases = phases_str.split()

def blank():
    return {"wall_clock_seconds": 0, "elapsed_wall_seconds": 0, "tokens_in": 0, "tokens_out": 0,
            "cache_read": 0, "cache_write": 0, "tokens_total": 0,
            "requests": 0, "credits": 0, "cost_usd": 0}

per = {p: blank() for p in phases}
def bucket(p):
    if p not in per:
        per[p] = blank()
    return per[p]

def parse_ts(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except Exception:
        return None

def ts_epoch(s):
    d = parse_ts(s)
    return int(d.timestamp()) if d is not None else None

first_ts = last_ts = None   # first-to-last span of the log = true elapsed wall-clock
intervals = []              # (started_epoch, ended_epoch, phase) reconstructed from phase events

if os.path.exists(events_path):
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(e.get("ts", ""))
            # Out-of-band imports may be folded in long after the run; don't let their timestamps
            # stretch the true elapsed span -- only in-run events (start/stop/usage) anchor it.
            if ts is not None and not e.get("imported"):
                if first_ts is None or ts < first_ts:
                    first_ts = ts
                if last_ts is None or ts > last_ts:
                    last_ts = ts
            b = bucket(e.get("phase", "unknown"))
            if e.get("event") == "phase":
                secs = int(e.get("seconds", 0) or 0)
                b["wall_clock_seconds"] += secs
                # Reconstruct [started, ended] for this interval so we can attribute the idle gaps
                # BETWEEN phases (approval/latency) to a phase, not just the bracketed time.
                # stop events carry started/ended; auto-close events carry only seconds+ts.
                en = e.get("ended")
                if en is None:
                    en = ts_epoch(e.get("ts", ""))
                st = e.get("started")
                if st is None and en is not None:
                    st = en - secs
                if st is not None and en is not None:
                    intervals.append((int(st), int(en), e.get("phase", "unknown")))
            elif e.get("event") == "usage":
                b["tokens_in"]    += int(e.get("tokens_in", 0) or 0)
                b["tokens_out"]   += int(e.get("tokens_out", 0) or 0)
                b["cache_read"]   += int(e.get("cache_read", 0) or 0)
                b["cache_write"]  += int(e.get("cache_write", 0) or 0)
                b["tokens_total"] += int(e.get("tokens_total", 0) or 0)
                b["requests"]     += int(e.get("requests", 0) or 0)
                b["credits"]      += float(e.get("credits", 0) or 0)
                b["cost_usd"]     += float(e.get("cost_usd", 0) or 0)

# Per-phase ELAPSED: attribute the timeline between phase boundaries to the phase that was active.
# Sort the reconstructed intervals by start; each phase gets the span from its own start up to the
# NEXT phase's start (so the idle gap that follows a phase -- approval, latency, transition -- is
# charged to it), and the final phase gets its own start->end. This telescopes so the per-phase
# elapsed sums to the whole run span, while `wall_clock_seconds` stays the bracketed "active" time.
intervals.sort(key=lambda iv: iv[0])
for i, (st, en, ph) in enumerate(intervals):
    nxt = intervals[i + 1][0] if i + 1 < len(intervals) else en
    span = nxt - st if i + 1 < len(intervals) else en - st
    bucket(ph)["elapsed_wall_seconds"] += max(0, span)

totals = blank()
for b in per.values():
    for k in totals:
        totals[k] += b[k]

# True elapsed wall-clock = span from the first to the last recorded event. Unlike the per-phase
# `wall_clock_seconds` (the sum of bracketed start/stop intervals -- "active" reasoning/tool time),
# this captures the WHOLE run including everything the agent never bracketed: human approval/review,
# model latency, and idle time between turns. On engines that don't expose token usage back to the
# agent (VS Code Copilot, Trae) the active sum badly undercounts real time, so this is the faithful
# duration signal. It needs no cooperation from the agent -- any start/stop/usage/import event anchors it.
#
# Heartbeat cross-check: gen-progress.sh appends a timestamped snapshot to progress.jsonl after EVERY
# state change -- far more reliably than the agent brackets phases. We read those timestamps to (a)
# report an independent `observed_span` / `observed_work` (idle-capped) that does not depend on
# start/stop, and (b) ANCHOR the elapsed span itself, so `elapsed_wall_seconds` is faithful even when
# no start/stop was ever called.
beat_epochs = []
if os.path.exists(progress_path):
    with open(progress_path) as pf:
        for ln in pf:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            d = parse_ts(r.get("ts", ""))
            if d is not None:
                beat_epochs.append(int(d.timestamp()))
beat_epochs.sort()
IDLE_CAP = int(os.environ.get("SYCL_PROGRESS_IDLE_CAP", "1800"))
observed_span = observed_work = 0
if beat_epochs:
    observed_span = beat_epochs[-1] - beat_epochs[0]
    for a, b in zip(beat_epochs, beat_epochs[1:]):
        observed_work += min(max(b - a, 0), IDLE_CAP)

elapsed = 0
started_at = ended_at = None
epochs = []
if first_ts is not None:
    epochs.append(int(first_ts.timestamp()))
if last_ts is not None:
    epochs.append(int(last_ts.timestamp()))
for st, en, _ph in intervals:
    epochs.append(st); epochs.append(en)
epochs.extend(beat_epochs)   # heartbeats anchor elapsed too, so it survives missing start/stop
if epochs:
    lo, hi = min(epochs), max(epochs)
    elapsed = hi - lo
    started_at = datetime.datetime.fromtimestamp(lo, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ended_at   = datetime.datetime.fromtimestamp(hi, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
session = {
    "elapsed_wall_seconds": elapsed,
    "active_wall_seconds": totals["wall_clock_seconds"],
    "observed_span_seconds": observed_span,
    "observed_work_seconds": observed_work,
    "observed_snapshots": len(beat_epochs),
    "started_at": started_at,
    "ended_at": ended_at,
}

def tidy(d):
    d = dict(d)
    d["credits"] = round(float(d.get("credits", 0)), 4)
    d["cost_usd"] = round(float(d.get("cost_usd", 0)), 4)
    return d

out = {
    "schema": "sycl-agent/metrics",
    "unit": {"time": "seconds",
             "cost": "USD (if reported) and/or tokens (uncached + cached) and/or GitHub AI credits / premium requests"},
    "session": session,
    "totals": tidy(totals),
    "phases": {p: tidy(b) for p, b in per.items()},
    "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(rollup_path, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY
}

cmd_import() {
  # Ingest cost/usage recorded OUT-OF-BAND by the IDE/engine itself, for engines that do not expose
  # token usage back to the agent (e.g. VS Code Copilot, Trae). Export the engine's usage/telemetry,
  # normalize it to this simple JSON shape, and fold it in -- no self-reporting by the model required.
  #
  #   metrics.sh import <usage.json>
  #
  # The file is either an array of usage rows, or an object with a "usage": [...] array. Each row:
  #   { "phase": "migrate", "tokens_in": N, "tokens_out": N, "cache_read": N, "cache_write": N,
  #     "requests": N, "credits": X, "cost_usd": X, "model": "...", "kernel": "...", "ts": "ISO8601",
  #     "uid": "...", "source": "...", "note": "..." }
  #                            -- every field optional (phase defaults to "import", ts to now).
  # `uid` is a stable id of the source record (e.g. the engine's own message id). It is written to
  # the event log and rows whose uid is already there are SKIPPED, so importing the same export
  # twice cannot double-count. collect-usage.sh sets it automatically.
  local file="${1:?usage: metrics.sh import <usage.json>}"
  [[ -f "$file" ]] || { echo "metrics.sh import: no such file: $file" >&2; return 2; }
  python3 - "$file" "$EVENTS" <<'PY'
import json, sys, datetime
src, events = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(src))
except Exception as e:
    sys.stderr.write(f"metrics.sh import: cannot parse {src}: {e}\n"); sys.exit(2)
rows = data.get("usage", []) if isinstance(data, dict) else data
if isinstance(rows, dict):
    rows = [rows]
if not isinstance(rows, list):
    sys.stderr.write("metrics.sh import: expected an array of usage rows or {\"usage\": [...]}\n")
    sys.exit(2)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def i(r, k): 
    try: return int(r.get(k, 0) or 0)
    except Exception: return 0
def fl(r, k):
    try: return float(r.get(k, 0) or 0)
    except Exception: return 0.0
# uids already in the log -- re-importing the same export must not double-count
seen = set()
try:
    for line in open(events):
        try: u = json.loads(line).get("uid")
        except Exception: u = None
        if u: seen.add(u)
except FileNotFoundError:
    pass
n = dup = 0
with open(events, "a") as f:
    for r in rows:
        if not isinstance(r, dict):
            continue
        uid = r.get("uid") or ""
        if uid and uid in seen:
            dup += 1
            continue
        if uid:
            seen.add(uid)
        ti, to = i(r, "tokens_in"), i(r, "tokens_out")
        crd, cwr = i(r, "cache_read"), i(r, "cache_write")
        ev = {
            "ts": r.get("ts") or now, "event": "usage", "phase": r.get("phase") or "import",
            "imported": True,
            "tokens_in": ti, "tokens_out": to, "cache_read": crd, "cache_write": cwr,
            "tokens_total": ti + to + crd + cwr,
            "requests": i(r, "requests"), "credits": fl(r, "credits"), "cost_usd": fl(r, "cost_usd"),
            "model": r.get("model", ""), "kernel": r.get("kernel", ""),
            "uid": uid, "source": r.get("source", ""),
            "note": r.get("note", "imported (out-of-band)"),
        }
        f.write(json.dumps(ev) + "\n")
        n += 1
print(f"imported {n} usage row(s) from {src}" + (f" ({dup} duplicate uid(s) skipped)" if dup else ""))
PY
  cmd_rollup
}

cmd_summary() {
  [[ -f "$ROLLUP" ]] || cmd_rollup
  python3 - "$ROLLUP" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
t = m.get("totals", {})
s = m.get("session", {})

def hms(s):
    s = int(s); h = s // 3600; mnt = (s % 3600) // 60; sec = s % 60
    if h:   return f"{h}h{mnt:02d}m{sec:02d}s"
    if mnt: return f"{mnt}m{sec:02d}s"
    return f"{sec}s"

print("Agent efficiency & cost")
print(f"  elapsed time  : {hms(s.get('elapsed_wall_seconds', 0))}   (wall-clock span of the whole run)")
print(f"  active time   : {hms(s.get('active_wall_seconds', t.get('wall_clock_seconds', 0)))}   (bracketed reasoning + tool time)")
if s.get('observed_snapshots'):
    obs_span = s.get('observed_span_seconds', 0) or 0
    obs_work = s.get('observed_work_seconds', 0) or 0
    print(f"  observed span : {hms(obs_span)}   (gen-progress heartbeat, {s.get('observed_snapshots')} snapshots — independent of start/stop)")
    print(f"  observed work : {hms(obs_work)}   (idle-capped)")
    act = s.get('active_wall_seconds', t.get('wall_clock_seconds', 0)) or 0
    if obs_work and act < 0.5 * obs_work:
        print(f"  ! WARNING     : bracketed active ({hms(act)}) is far below observed work ({hms(obs_work)}) — phases under-bracketed; trust elapsed/observed")
print(f"  total cost    : ${t.get('cost_usd', 0)}")
print(f"  total tokens  : {t.get('tokens_total', 0)} "
      f"(in {t.get('tokens_in', 0)} / out {t.get('tokens_out', 0)} / "
      f"cache r {t.get('cache_read', 0)} / cache w {t.get('cache_write', 0)})")
print(f"  premium reqs  : {t.get('requests', 0)}")
print(f"  AI credits    : {t.get('credits', 0)}")
print()
hdr = f"  {'phase':<12} {'elapsed':>10} {'active':>10} {'tokens':>12} {'reqs':>6} {'credits':>9} {'USD':>9}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for p, b in m.get("phases", {}).items():
    if not any(b.values()):
        continue
    print(f"  {p:<12} {hms(b.get('elapsed_wall_seconds', 0)):>10} "
          f"{hms(b.get('wall_clock_seconds', 0)):>10} "
          f"{b.get('tokens_total', 0):>12} {b.get('requests', 0):>6} "
          f"{b.get('credits', 0):>9} {b.get('cost_usd', 0):>9}")
PY
}

cmd="${1:-summary}"; shift || true
case "$cmd" in
  start)   cmd_start "$@" ;;
  stop)    cmd_stop  "$@" ;;
  usage)   cmd_usage "$@" ;;
  import)  cmd_import "$@" ;;
  rollup)  cmd_rollup ;;
  summary) cmd_summary ;;
  -h|--help) sed -n '2,45p' "$0" ;;
  *) echo "metrics.sh: unknown command '$cmd' (start|stop|usage|import|rollup|summary)" >&2; exit 2 ;;
esac
