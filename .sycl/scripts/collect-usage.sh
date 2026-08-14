#!/usr/bin/env bash
# Engine usage collector — harvest REAL token/cost usage from the coding engine's own session
# store, attribute it to workflow phases, and fold it into the metrics rollup.
#
# Why this exists: most engines do NOT hand their token accounting back to the model, so the agent
# cannot self-report it (`metrics.sh usage` stays 0) — yet cost is exactly what a migration has to be
# judged on. Every engine does, however, persist its own session log on disk. This script reads that
# log, normalizes it to the `metrics.sh import` shape, and imports it — no self-reporting, no guessing.
#
# Usage:
#   collect-usage.sh detect                       list engine session stores that hold data for this project
#   collect-usage.sh collect [flags]              extract usage rows -> usage.json (and optionally import)
#   collect-usage.sh stats save|diff [label]      opencode cross-check: snapshot `opencode stats`, then diff
#   collect-usage.sh bench                        print the sycl-agent-bench `usage` block from the rollup
#
#   collect flags:
#     --engine auto|claude|opencode|copilot|trae|all   which session store to read (default: auto = all that match)
#     --project DIR                               project root the session must belong to (default: parent of .sycl)
#     --since ISO8601|EPOCH                       ignore records older than this (default: first metrics/progress event)
#     --out FILE                                  where to write the normalized rows (default: .sycl/logs/usage.import.json)
#     --import                                    fold the rows into metrics.jsonl via `metrics.sh import`
#     --estimate-tokens                           VS Code Copilot chat only: ESTIMATE tokens from transcript size
#     --all-time                                  do not apply the default --since floor
#
# Engines and what they expose:
#   claude    ~/.claude/**/*.jsonl (projects/<slug>/, incl. subagents/) — EXACT per-request
#             input/output/cache-read/cache-write tokens + model, and costUSD when present.
#   opencode  ~/.local/share/opencode/opencode.db (sqlite) — EXACT per-message input/output/reasoning/
#             cache tokens + USD cost + model, sessions filtered by their working directory.
#             Cross-check the total with `collect-usage.sh stats save/diff` (wraps `opencode stats`).
#   copilot   ~/.copilot/session-store.db (sqlite, `assistant_usage_events`) — EXACT per-request
#             input/output/cache-read/cache-write/reasoning tokens, model, premium-request multiplier
#             and API duration, with sessions filtered by their `cwd`. Falls back to the
#             `session.shutdown` totals in ~/.copilot/session-state/<id>/events.jsonl (still exact,
#             but per session instead of per request), and finally to VS Code
#             workspaceStorage/*/GitHub.copilot-chat/transcripts/*.jsonl — the VS Code chat
#             transcript logs NO tokens, so it only yields premium requests + round-trips (or
#             clearly-marked estimates with --estimate-tokens).
#   trae      ~/.config/Trae[ CN]/logs/<stamp>/window*/renderer*.log — Trae ENCRYPTS its conversation
#             store (ModularData/ai-agent/database.db is not readable sqlite) and logs no token
#             counts, so only the chat markers are recorded: user prompts (`chatStream
#             handleUserMessage`) and model round-trips (`event=token_usage`). Tokens stay 0 and
#             cannot be estimated — the message text never reaches the log.
#
# Phase attribution: each usage record carries a timestamp, so it is bucketed into the workflow phase
# that was open at that moment — reconstructed from the bracketed intervals in .sycl/logs/metrics.jsonl
# and, as a denser fallback, gen-progress's heartbeat trail .sycl/logs/progress.jsonl (which records
# the phase after every state change). Records outside any known interval clamp to the nearest phase.
#
# Idempotent: every row carries a stable `uid` from the engine's own record id. Rows whose uid is
# already present in .sycl/logs/metrics.jsonl are skipped, so this can be run after every phase (or
# repeatedly at the end) without double-counting.
#
# Overrides: SYCL_CLAUDE_HOME, SYCL_OPENCODE_DB, SYCL_COPILOT_HOME, SYCL_COPILOT_DB,
#            SYCL_VSCODE_USER_DIRS (colon-separated), SYCL_TRAE_DIRS (colon-separated),
#            SYCL_TRAE_LOG_MAX (skip logs larger than this, default 200 MB), SYCL_COPILOT_CTX_CAP
#            (context cap used by --estimate-tokens, default 200000).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYCL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"     # .sycl/
STATE="$SYCL_DIR/state"
LOG_DIR="$SYCL_DIR/logs"
EVENTS="$LOG_DIR/metrics.jsonl"
PROGRESS="$LOG_DIR/progress.jsonl"
ROLLUP="$STATE/metrics.json"
PROJECT_DEFAULT="$(cd "$SYCL_DIR/.." && pwd)"

mkdir -p "$LOG_DIR" "$STATE"

PY_COLLECTOR="$(cat <<'PY'
import datetime, glob, json, os, re, sqlite3, sys, urllib.parse

mode        = sys.argv[1]
project     = os.path.realpath(sys.argv[2])
engine      = sys.argv[3]
since_arg   = sys.argv[4]
out_path    = sys.argv[5]
events_path = sys.argv[6]
prog_path   = sys.argv[7]
estimate    = sys.argv[8] == "1"
all_time    = sys.argv[9] == "1"

# ---------------------------------------------------------------- helpers
def iso(ep):
    return datetime.datetime.fromtimestamp(int(ep), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_ts(s):
    """ISO-8601 (with or without fractional seconds / Z) -> epoch seconds."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{9,13}", s):                     # raw epoch (s or ms)
        v = int(s)
        return v // 1000 if v > 10_000_000_000 else v
    s = s.replace("Z", "+00:00")
    try:
        d = datetime.datetime.fromisoformat(s)
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return int(d.timestamp())

def under(path, root):
    if not path:
        return False
    try:
        path = os.path.realpath(path)
    except Exception:
        return False
    return path == root or path.startswith(root + os.sep)

# ---------------------------------------------------------------- phase timeline
def phase_timeline():
    """Reconstruct (start, end, phase) intervals from the bracketed metrics events, plus a denser
    list of (epoch, phase) marks from the gen-progress heartbeat trail."""
    intervals, marks = [], []
    if os.path.exists(events_path):
        for line in open(events_path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event") == "phase" and d.get("started") and d.get("ended"):
                intervals.append((int(d["started"]), int(d["ended"]), d.get("phase") or "import"))
    if os.path.exists(prog_path):
        for line in open(prog_path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            ep, ph = parse_ts(d.get("ts")), d.get("phase")
            if ep and ph:
                marks.append((ep, ph))
    intervals.sort()
    marks.sort()
    return intervals, marks

def phase_for(ep, intervals, marks):
    for s, e, p in intervals:
        if s <= ep <= e:
            return p
    prev = None
    for m_ep, m_ph in marks:            # last heartbeat at or before the record
        if m_ep <= ep:
            prev = m_ph
        else:
            break
    if prev:
        return prev
    if marks:
        return marks[0][1]              # record predates the first heartbeat
    if intervals:
        return min(intervals, key=lambda iv: abs(iv[0] - ep))[2]
    return "import"

def imported_uids():
    seen = set()
    if os.path.exists(events_path):
        for line in open(events_path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("uid"):
                seen.add(d["uid"])
    return seen

def since_floor():
    """Default floor: the first event of this run, so pre-existing sessions are not swept in."""
    if all_time:
        return None
    if since_arg:
        return parse_ts(since_arg)
    firsts = []
    for p in (events_path, prog_path):
        if os.path.exists(p):
            for line in open(p, errors="replace"):
                try:
                    ep = parse_ts(json.loads(line).get("ts"))
                except Exception:
                    ep = None
                if ep:
                    firsts.append(ep)
                    break
    return min(firsts) if firsts else None

# ---------------------------------------------------------------- claude code
def claude_slug(path):
    return re.sub(r"[^A-Za-z0-9]", "-", path)

def collect_claude():
    home = os.environ.get("SYCL_CLAUDE_HOME") or os.path.expanduser("~/.claude")
    if not os.path.isdir(home):
        return [], []
    slug = claude_slug(project)
    rows, files = [], []
    # usage lives under projects/<slug>/, but scan every jsonl under ~/.claude so a layout change
    # (or a session recorded elsewhere) is still picked up — records are filtered by slug or cwd.
    for f in glob.glob(os.path.join(home, "**", "*.jsonl"), recursive=True):
        parts = os.path.relpath(f, home).split(os.sep)
        slug_match = any(p == slug or p.startswith(slug + "-") for p in parts)
        used = False
        for line in open(f, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = (d.get("message") or {}).get("usage")
            if not u:
                continue
            cwd = d.get("cwd")
            if not (slug_match or under(cwd, project)):
                continue
            ep = parse_ts(d.get("timestamp"))
            if ep is None:
                continue
            used = True
            rows.append({
                "ep": ep,
                "tokens_in":   int(u.get("input_tokens") or 0),
                "tokens_out":  int(u.get("output_tokens") or 0),
                "cache_read":  int(u.get("cache_read_input_tokens") or 0),
                "cache_write": int(u.get("cache_creation_input_tokens") or 0),
                "cost_usd":    float(d.get("costUSD") or 0),
                "requests":    0,
                "model":       (d.get("message") or {}).get("model") or "",
                "uid":         "claude:" + (d.get("uuid") or f"{d.get('sessionId','?')}:{len(rows)}"),
                "source":      "claude-code session log",
                "note":        "claude-code session log (exact)",
            })
        if used:
            files.append(f)
    return rows, files

# ---------------------------------------------------------------- opencode
def collect_opencode():
    db = os.environ.get("SYCL_OPENCODE_DB") or os.path.expanduser("~/.local/share/opencode/opencode.db")
    rows, files = [], []
    if os.path.exists(db):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            sessions = {}
            for sid, directory in con.execute("SELECT id, directory FROM session"):
                if under(directory, project):
                    sessions[sid] = directory
            if sessions:
                files.append(db)
                q = "SELECT id, session_id, time_created, data FROM message"
                for mid, sid, created, data in con.execute(q):
                    if sid not in sessions:
                        continue
                    try:
                        d = json.loads(data)
                    except Exception:
                        continue
                    row = _opencode_row(mid, d, created)
                    if row:
                        rows.append(row)
            con.close()
        except sqlite3.Error as e:
            print(f"collect-usage: opencode db unreadable ({e})", file=sys.stderr)
    # older opencode keeps one JSON file per message
    store = os.path.expanduser("~/.local/share/opencode/storage/message")
    if not rows and os.path.isdir(store):
        for f in glob.glob(os.path.join(store, "*", "*.json")):
            try:
                d = json.load(open(f, errors="replace"))
            except Exception:
                continue
            if not under(d.get("path", {}).get("cwd") if isinstance(d.get("path"), dict) else None, project):
                continue
            row = _opencode_row(os.path.splitext(os.path.basename(f))[0], d, None)
            if row:
                rows.append(row)
                files.append(f)
    return rows, files

def _opencode_row(mid, d, created_ms):
    if d.get("role") != "assistant":
        return None
    tok = d.get("tokens") or {}
    cache = tok.get("cache") or {}
    ep = parse_ts(str((d.get("time") or {}).get("created") or created_ms or ""))
    if ep is None:
        return None
    return {
        "ep": ep,
        "tokens_in":   int(tok.get("input") or 0),
        # reasoning tokens are billed as output
        "tokens_out":  int(tok.get("output") or 0) + int(tok.get("reasoning") or 0),
        "cache_read":  int(cache.get("read") or 0),
        "cache_write": int(cache.get("write") or 0),
        "cost_usd":    float(d.get("cost") or 0),
        "requests":    0,
        "model":       d.get("modelID") or "",
        "uid":         f"opencode:{mid}",
        "source":      "opencode session store",
        "note":        "opencode session store (exact)",
    }

# ---------------------------------------------------------------- github copilot
def collect_copilot():
    """Copilot keeps three stores, in decreasing order of fidelity:
       1. ~/.copilot/session-store.db  -> exact tokens per model request (CLI / agent sessions)
       2. ~/.copilot/session-state/<id>/events.jsonl session.shutdown -> exact tokens per session
       3. VS Code chat transcripts     -> no tokens at all, only premium requests + round-trips"""
    home = os.environ.get("SYCL_COPILOT_HOME") or os.path.expanduser("~/.copilot")
    rows, files = [], []
    db_rows, db_files, covered = _copilot_db(home)
    rows.extend(db_rows); files.extend(db_files)
    st_rows, st_files = _copilot_session_state(home, covered)
    rows.extend(st_rows); files.extend(st_files)
    if rows:
        return rows, files            # exact stores answered — no need for the token-less transcripts
    vs_rows, vs_files = _copilot_vscode()
    rows.extend(vs_rows); files.extend(vs_files)
    return rows, files

def _copilot_db(home):
    """assistant_usage_events: one row per model request, with sessions scoped by their cwd."""
    db = os.environ.get("SYCL_COPILOT_DB") or os.path.join(home, "session-store.db")
    rows, files, covered = [], [], set()
    if not os.path.exists(db):
        return rows, files, covered
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"sessions", "assistant_usage_events"} <= tables:
            con.close()
            return rows, files, covered
        mine = {sid for sid, cwd in con.execute("SELECT id, cwd FROM sessions") if under(cwd, project)}
        if mine:
            q = ("SELECT id, session_id, model, input_tokens, output_tokens, cache_read_tokens, "
                 "cache_write_tokens, request_multiplier, created_at FROM assistant_usage_events")
            for rid, sid, model, tin, tout, cr, cw, mult, created in con.execute(q):
                if sid not in mine:
                    continue
                ep = parse_ts(created)
                if ep is None:
                    continue
                cr, cw = int(cr or 0), int(cw or 0)
                covered.add(sid)
                rows.append({
                    "ep": ep,
                    # input_tokens is the whole prompt, cached part included
                    "tokens_in":   max(int(tin or 0) - cr - cw, 0),
                    "tokens_out":  int(tout or 0),      # reasoning tokens are already counted here
                    "cache_read":  cr,
                    "cache_write": cw,
                    "cost_usd":    0.0,
                    "requests":    1,
                    "credits":     float(mult or 0),    # premium-request cost of this call
                    "model":       model or "",
                    "uid":         f"copilot:req:{sid}:{rid}",
                    "source":      "copilot session-store.db",
                    "note":        "copilot assistant_usage_events (exact, per request)",
                })
        con.close()
    except sqlite3.Error as e:
        print(f"collect-usage: copilot session-store.db unreadable ({e})", file=sys.stderr)
    if rows:
        files.append(db)
    return rows, files, covered

def _copilot_ws_match(ws_yaml):
    """workspace.yaml is a flat key: value file — cwd/git_root tell us which project the session ran in."""
    try:
        text = open(ws_yaml, errors="replace").read()
    except Exception:
        return False
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k.strip() in ("cwd", "git_root") and under(v.strip().strip('"\''), project):
            return True
    return False

def _copilot_session_state(home, covered):
    root = os.path.join(home, "session-state")
    rows, files = [], []
    if not os.path.isdir(root):
        return rows, files
    for ws_yaml in glob.glob(os.path.join(root, "*", "workspace.yaml")):
        sdir = os.path.dirname(ws_yaml)
        sid = os.path.basename(sdir)
        if sid in covered or not _copilot_ws_match(ws_yaml):
            continue                                   # already covered by the exact per-request store
        events = os.path.join(sdir, "events.jsonl")
        if not os.path.exists(events):
            continue
        got = _copilot_shutdowns(events, sid)
        if got:
            rows.extend(got)
            files.append(events)
    return rows, files

def _copilot_shutdown_metrics(m):
    u = m.get("usage") or {}
    td = m.get("tokenDetails") or {}
    def tdv(k):
        return int(((td.get(k) or {}).get("tokenCount")) or 0)
    cr = int(u.get("cacheReadTokens") or 0) or tdv("cache_read")
    cw = int(u.get("cacheWriteTokens") or 0) or tdv("cache_write")
    inp = int(u.get("inputTokens") or 0)
    req = m.get("requests") or {}
    return {
        # inputTokens counts the cached part too; tokenDetails.input is already the fresh part
        "tokens_in":   max(inp - cr - cw, 0) if inp else tdv("input"),
        "tokens_out":  int(u.get("outputTokens") or 0) or tdv("output"),
        "cache_read":  cr,
        "cache_write": cw,
        "requests":    int(req.get("count") or 0),
        "credits":     float(req.get("cost") or 0),
    }

def _copilot_shutdowns(path, sid):
    """session.shutdown carries the exact per-model totals of ONE cli process, so a session that was
    resumed writes one event per run and they add up (verified against session-store.db)."""
    rows = []
    for line in open(path, errors="replace"):
        if '"session.shutdown"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        ep = parse_ts(d.get("timestamp"))
        if ep is None:
            continue
        data = d.get("data") or {}
        models = data.get("modelMetrics") or {}
        if not models and data.get("tokenDetails"):      # older shutdown events: session-wide only
            models = {data.get("currentModel") or "": {
                "tokenDetails": data["tokenDetails"],
                "requests": {"cost": data.get("totalPremiumRequests") or 0}}}
        for model, m in models.items():
            v = _copilot_shutdown_metrics(m)
            if not any(v[k] for k in ("tokens_in", "tokens_out", "cache_read", "cache_write")):
                continue
            rows.append({
                "ep": ep, "cost_usd": 0.0, "model": model,
                "uid": f"copilot:session:{d.get('id') or sid}:{model}",
                "source": "copilot session-state events.jsonl",
                "note": f"copilot session.shutdown totals for session {sid[:8]} "
                        f"(exact, whole run booked at its end)",
                **v,
            })
    return rows

def vscode_user_dirs():
    env = os.environ.get("SYCL_VSCODE_USER_DIRS")
    if env:
        return [p for p in env.split(":") if p]
    home = os.path.expanduser("~")
    return [
        os.path.join(home, ".config/Code/User"),
        os.path.join(home, ".config/Code - Insiders/User"),
        os.path.join(home, "Library/Application Support/Code/User"),
        os.path.join(home, ".vscode-server/data/User"),
        os.path.join(home, "AppData/Roaming/Code/User"),
    ]

def _workspace_paths(ws_json):
    """Folders opened in that VS Code window: workspace.json points either at a folder or at a
    .code-workspace file, whose "path" entries are resolved relative to it."""
    try:
        meta = json.load(open(ws_json, errors="replace"))
    except Exception:
        return []
    out = []
    for key in ("folder", "workspace"):
        uri = meta.get(key)
        if not (isinstance(uri, str) and uri.startswith("file://")):
            continue
        path = urllib.parse.unquote(uri[len("file://"):])
        out.append(path)
        if path.endswith(".code-workspace") and os.path.exists(path):
            try:
                raw = open(path, errors="replace").read()
            except Exception:
                continue
            base = os.path.dirname(path)
            out += [p if os.path.isabs(p) else os.path.join(base, p)
                    for p in re.findall(r'"path"\s*:\s*"([^"]+)"', raw)]
    return out

def _copilot_vscode():
    """VS Code chat sessions never reach ~/.copilot and log no tokens — requests only."""
    cap = int(os.environ.get("SYCL_COPILOT_CTX_CAP") or 200000)
    rows, files = [], []
    for user_dir in vscode_user_dirs():
        for ws in glob.glob(os.path.join(user_dir, "workspaceStorage", "*")):
            tdir = os.path.join(ws, "GitHub.copilot-chat", "transcripts")
            if not os.path.isdir(tdir):
                continue
            # Scope by the window's own folder list — a transcript that merely mentions the project
            # path (because someone talked about it) is NOT this project's usage.
            if not any(under(p, project) or under(project, p)
                       for p in _workspace_paths(os.path.join(ws, "workspace.json"))):
                continue
            for t in glob.glob(os.path.join(tdir, "*.jsonl")):
                got = _copilot_transcript(t, cap)
                if got:
                    rows.extend(got)
                    files.append(t)
    return rows, files

def _copilot_transcript(path, cap):
    try:
        raw = open(path, errors="replace").read()
    except Exception:
        return []
    rows, ctx_chars, turn_chars, pending = [], 0, 0, None
    sid = os.path.splitext(os.path.basename(path))[0]
    for line in raw.splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        typ = d.get("type")
        size = len(line)
        if typ == "user.message":
            if pending:
                rows.append(_copilot_row(pending, turn_chars, ctx_chars, cap, sid))
            ep = parse_ts(d.get("timestamp"))
            pending = {"ep": ep, "id": d.get("id"), "round_trips": 0} if ep else None
            turn_chars = 0
        elif typ == "assistant.turn_start" and pending:
            pending["round_trips"] += 1
        ctx_chars = min(ctx_chars + size, cap * 4)
        turn_chars += size
    if pending:
        rows.append(_copilot_row(pending, turn_chars, ctx_chars, cap, sid))
    return [r for r in rows if r]

def _copilot_row(pending, turn_chars, ctx_chars, cap, sid):
    row = {
        "ep": pending["ep"],
        "tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0,
        "cost_usd": 0.0,
        "requests": 1,                       # one premium request per user prompt
        "model": "",
        "uid": f"copilot:vscode:{pending['id'] or sid}",
        "source": "vscode copilot chat transcript",
        "note": f"vscode copilot chat: 1 premium request, {pending['round_trips']} model round-trips "
                f"(the vscode chat transcript logs no token counts)",
    }
    if estimate:
        rt = max(pending["round_trips"], 1)
        # The whole context is resent on every round-trip; ~4 chars/token, capped at the window.
        row["tokens_in"] = int(min(ctx_chars / 4, cap) * rt)
        row["tokens_out"] = int(turn_chars / 4)
        row["note"] = (f"vscode copilot chat: 1 premium request, {pending['round_trips']} model "
                       f"round-trips; tokens ESTIMATED from transcript size (~4 chars/token, "
                       f"context cap {cap}) — the vscode chat transcript logs no token counts")
    return row

# ---------------------------------------------------------------- trae
def trae_roots():
    env = os.environ.get("SYCL_TRAE_DIRS")
    if env:
        return [p for p in env.split(":") if p]
    home = os.path.expanduser("~")
    out = []
    for name in ("Trae", "Trae CN"):
        out += [os.path.join(home, ".config", name),
                os.path.join(home, "Library/Application Support", name),
                os.path.join(home, "AppData/Roaming", name)]
    out.append(os.path.join(home, ".trae-server/data"))       # remote (SSH) sessions
    return out

def collect_trae():
    # Trae encrypts its conversation store (ModularData/ai-agent/database.db is not a readable
    # sqlite file) and logs no token counts anywhere, so the only honest signals are the chat
    # markers in the window renderer logs: one `chatStream handleUserMessage` per user prompt and
    # one `event=token_usage` per model round-trip that reported usage.
    max_bytes = int(os.environ.get("SYCL_TRAE_LOG_MAX") or 200 * 1024 * 1024)
    rows, files = [], []
    for root in trae_roots():
        for wdir in glob.glob(os.path.join(root, "logs", "*", "window*")):
            logs = sorted(glob.glob(os.path.join(wdir, "renderer*.log")))
            if not logs:
                continue
            probe = logs + [os.path.join(wdir, "exthost", "exthost.log")]
            wrows, hit = [], False
            for f in probe:
                if not os.path.isfile(f) or os.path.getsize(f) > max_bytes:
                    continue
                got, f_hit = _trae_log(f, wdir)
                hit = hit or f_hit
                if f in logs:
                    wrows.extend(got)
            if hit and wrows:
                rows.extend(wrows)
                files.append(wdir)
    return rows, files

def _trae_log(path, wdir):
    tag = os.sep.join(path.split(os.sep)[-3:-1])          # <launch-stamp>/<window>
    out, hit, pending = [], False, None
    try:
        fh = open(path, errors="replace")
    except Exception:
        return out, hit
    with fh:
        for line in fh:
            if not hit and project in line:
                hit = True
            user = "chatStream handleUserMessage" in line
            turn = "event=token_usage" in line
            if not (user or turn):
                continue
            stamp = line.split(" ", 1)[0]
            ep = parse_ts(stamp)
            if ep is None:
                continue
            if user:
                if pending:
                    out.append(_trae_row(pending, tag))
                pending = {"ep": ep, "stamp": stamp, "requests": 1, "round_trips": 0}
            else:
                if pending is None:                        # prompt rotated out of this log chunk
                    pending = {"ep": ep, "stamp": stamp, "requests": 0, "round_trips": 0}
                pending["round_trips"] += 1
    if pending:
        out.append(_trae_row(pending, tag))
    return out, hit

def _trae_row(pending, tag):
    what = "1 user prompt" if pending["requests"] else "prompt not in this log chunk"
    return {
        "ep": pending["ep"],
        "tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0,
        "cost_usd": 0.0,
        "requests": pending["requests"],
        "model": "",
        # the log is rotated by rename, so the file name is not stable — the ms timestamp is
        "uid": f"trae:{tag}:{pending['stamp']}",
        "source": "trae renderer log",
        "note": f"trae log: {what}, {pending['round_trips']} model round-trips "
                f"(trae encrypts its conversation store and logs no token counts)",
    }

# ---------------------------------------------------------------- drive
ENGINES = {"claude": collect_claude, "opencode": collect_opencode,
           "copilot": collect_copilot, "trae": collect_trae}
wanted = list(ENGINES) if engine in ("auto", "all") else [engine]

found, sources = {}, {}
for name in wanted:
    try:
        rows, files = ENGINES[name]()
    except Exception as e:                                  # never let one engine break the rest
        print(f"collect-usage: {name} collector failed: {e}", file=sys.stderr)
        rows, files = [], []
    if rows:
        found[name] = rows
        sources[name] = files

if mode == "detect":
    if not found:
        print("collect-usage: no engine session store holds usage for " + project)
        print("  looked in: ~/.claude/**/*.jsonl, opencode.db, ~/.copilot/session-store.db + "
              "session-state, VS Code workspaceStorage transcripts, Trae window logs")
        sys.exit(0)
    for name, rows in found.items():
        eps = [r["ep"] for r in rows]
        tot = sum(r["tokens_in"] + r["tokens_out"] + r["cache_read"] + r["cache_write"] for r in rows)
        print(f"{name:<9} {len(rows):>6} records  {iso(min(eps))} .. {iso(max(eps))}  "
              f"tokens={tot}  requests={sum(r['requests'] for r in rows)}")
        for f in sources[name][:3]:
            print(f"          {f}")
        if len(sources[name]) > 3:
            print(f"          ... and {len(sources[name]) - 3} more")
    sys.exit(0)

floor = since_floor()
intervals, marks = phase_timeline()
seen = imported_uids()
out_rows, skipped, stale = [], 0, 0
for name, rows in found.items():
    for r in rows:
        if floor is not None and r["ep"] < floor:
            stale += 1
            continue
        if r["uid"] in seen:
            skipped += 1
            continue
        seen.add(r["uid"])
        out_rows.append({
            "ts": iso(r["ep"]),
            "phase": phase_for(r["ep"], intervals, marks),
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "cache_read": r["cache_read"], "cache_write": r["cache_write"],
            "requests": r["requests"], "credits": r.get("credits", 0), "cost_usd": r["cost_usd"],
            "model": r["model"], "uid": r["uid"], "source": r["source"], "note": r["note"],
        })

out_rows.sort(key=lambda r: r["ts"])
with open(out_path, "w") as f:
    json.dump({"schema": "sycl-agent/usage-import", "project": project,
               "collected_at": iso(datetime.datetime.now(datetime.timezone.utc).timestamp()),
               "usage": out_rows}, f, indent=2)
    f.write("\n")

tot_in  = sum(r["tokens_in"] for r in out_rows)
tot_out = sum(r["tokens_out"] for r in out_rows)
tot_cr  = sum(r["cache_read"] for r in out_rows)
tot_cw  = sum(r["cache_write"] for r in out_rows)
print(f"collect-usage: {len(out_rows)} new row(s) from {', '.join(found) or 'no engine'} -> {out_path}")
if skipped:
    print(f"  {skipped} already imported (uid match) — skipped")
if stale:
    print(f"  {stale} older than the run start — skipped (use --all-time to include)")
if out_rows:
    print(f"  tokens in {tot_in} / out {tot_out} / cache r {tot_cr} / cache w {tot_cw}   "
          f"requests {sum(r['requests'] for r in out_rows)}   "
          f"cost ${round(sum(r['cost_usd'] for r in out_rows), 4)}")
    per = {}
    for r in out_rows:
        per[r["phase"]] = per.get(r["phase"], 0) + r["tokens_in"] + r["tokens_out"] + r["cache_read"] + r["cache_write"]
    print("  per phase: " + ", ".join(f"{k}={v}" for k, v in per.items()))
PY
)"

PY_BENCH="$(cat <<'PY'
import json, os, sys
rollup, events = sys.argv[1], sys.argv[2]
if not os.path.exists(rollup):
    sys.stderr.write("collect-usage bench: no metrics rollup yet — run metrics.sh rollup first\n")
    sys.exit(2)
m = json.load(open(rollup))
t, s = m.get("totals", {}), m.get("session", {})
iterations = int(t.get("requests") or 0)
if not iterations and os.path.exists(events):
    iterations = sum(1 for line in open(events, errors="replace")
                     if '"event":"usage"' in line or '"event": "usage"' in line)
wall = s.get("elapsed_wall_seconds") or s.get("observed_span_seconds") or t.get("wall_clock_seconds") or 0
print(json.dumps({"usage": {
    "wall_time_seconds": int(wall),
    # cached input is still input from the benchmark's point of view
    "input_tokens": int(t.get("tokens_in", 0)) + int(t.get("cache_read", 0)) + int(t.get("cache_write", 0)),
    "output_tokens": int(t.get("tokens_out", 0)),
    "total_tokens": int(t.get("tokens_total", 0)),
    "iterations": iterations,
}}, indent=2))
PY
)"

PY_STATS="$(cat <<'PY'
# opencode cross-check: `opencode stats` prints the engine's own lifetime totals. Snapshot it before
# the run and diff it afterwards — the delta must line up with what the collector imported.
import datetime, json, os, re, sys

mode, before_path, after_path, raw_path, events = sys.argv[1:6]
KEYS = ("Sessions", "Messages", "Days", "Total Cost", "Input", "Output", "Cache Read", "Cache Write")
MULT = {"K": 1e3, "M": 1e6, "B": 1e9}
ROW = re.compile(r"^([A-Za-z][A-Za-z /]*?)\s{2,}\$?([\d.,]+)\s*([KMB])?$")

def parse_stats(path):
    out = {}
    for line in open(path, errors="replace"):
        line = line.strip().strip("│|").strip()
        m = ROW.match(line)
        if not m:
            continue
        label, num, suf = m.group(1).strip(), m.group(2).replace(",", ""), m.group(3)
        if label not in KEYS:
            continue
        try:
            v = float(num) * MULT.get(suf or "", 1)
        except ValueError:
            continue
        out[label] = round(v, 2) if label == "Total Cost" else int(v)
    return out

def snapshot(path):
    return json.load(open(path))["stats"] if path.endswith(".json") else parse_stats(path)

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
stats = parse_stats(raw_path)
if not stats:
    sys.stderr.write("collect-usage stats: could not parse `opencode stats` output\n")
    sys.exit(2)

if mode == "save":
    with open(after_path, "w") as f:
        json.dump({"schema": "sycl-agent/opencode-stats", "captured_at": now, "stats": stats}, f, indent=2)
        f.write("\n")
    print(f"collect-usage: opencode stats snapshot -> {after_path}")
    for k in KEYS:
        if k in stats:
            print(f"  {k:<12} {stats[k]}")
    sys.exit(0)

if not os.path.exists(before_path):
    sys.stderr.write(f"collect-usage stats: no baseline at {before_path} — run `stats save` first\n")
    sys.exit(2)
before = snapshot(before_path)
after = snapshot(after_path) if after_path and os.path.exists(after_path) else stats

print(f"opencode stats delta ({os.path.basename(before_path)} -> "
      f"{os.path.basename(after_path) if after_path and os.path.exists(after_path) else 'now'})")
delta = {}
for k in KEYS:
    if k not in before and k not in after:
        continue
    delta[k] = round(after.get(k, 0) - before.get(k, 0), 2)
    print(f"  {k:<12} {before.get(k, 0):>14} -> {after.get(k, 0):>14}   delta {delta[k]:>14}")

engine_tokens = sum(int(delta.get(k, 0)) for k in ("Input", "Output", "Cache Read", "Cache Write"))
print(f"  {'TOKENS':<12} {'':>14}    {'':>14}   delta {engine_tokens:>14}")
mine = 0
if os.path.exists(events):
    for line in open(events, errors="replace"):
        if "opencode" not in line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "usage" and "opencode" in (e.get("source") or ""):
            mine += int(e.get("tokens_total") or 0)
print(f"  imported from opencode into metrics.jsonl: {mine}")
if engine_tokens and mine:
    drift = (mine - engine_tokens) / engine_tokens * 100
    print(f"  drift vs engine: {drift:+.1f}%  "
          f"({'ok' if abs(drift) <= 5 else 'CHECK — the collector may be missing sessions'})")
print("  note: `opencode stats` rounds to 3 significant digits and counts ALL projects, so treat "
      "this as a sanity check, not an audit.")
PY
)"

cmd_collect() {
  local engine="auto" project="$PROJECT_DEFAULT" since="" out="$LOG_DIR/usage.import.json"
  local do_import=0 estimate=0 all_time=0 mode="collect"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --engine)          engine="${2:?}";  shift 2 ;;
      --project)         project="${2:?}"; shift 2 ;;
      --since)           since="${2:?}";   shift 2 ;;
      --out)             out="${2:?}";     shift 2 ;;
      --import)          do_import=1;      shift ;;
      --estimate-tokens) estimate=1;       shift ;;
      --all-time)        all_time=1;       shift ;;
      --detect)          mode="detect";    shift ;;
      *) echo "collect-usage.sh: unknown flag '$1'" >&2; return 2 ;;
    esac
  done
  case "$engine" in
    auto|all|claude|opencode|copilot|trae) ;;
    *) echo "collect-usage.sh: unknown engine '$engine' (auto|all|claude|opencode|copilot|trae)" >&2; return 2 ;;
  esac
  python3 -c "$PY_COLLECTOR" "$mode" "$project" "$engine" "$since" "$out" \
          "$EVENTS" "$PROGRESS" "$estimate" "$all_time" || return $?
  if [[ "$mode" == "collect" && $do_import -eq 1 ]]; then
    "$SCRIPT_DIR/metrics.sh" import "$out"
  fi
}

cmd_stats() {
  local sub="${1:-diff}"; shift || true
  local before="${1:-baseline}" after="${2:-}"
  if ! command -v opencode >/dev/null 2>&1; then
    echo "collect-usage.sh stats: 'opencode' is not on PATH (this cross-check is opencode-only)" >&2
    return 2
  fi
  local raw; raw="$(mktemp)"
  trap 'rm -f "$raw"' RETURN
  if ! opencode stats >"$raw" 2>&1; then
    echo "collect-usage.sh stats: 'opencode stats' failed:" >&2
    cat "$raw" >&2
    return 1
  fi
  case "$sub" in
    save) python3 -c "$PY_STATS" save "" "$LOG_DIR/usage.stats.${before}.json" "$raw" "$EVENTS" ;;
    diff) python3 -c "$PY_STATS" diff "$LOG_DIR/usage.stats.${before}.json" \
                 "${after:+$LOG_DIR/usage.stats.${after}.json}" "$raw" "$EVENTS" ;;
    *) echo "collect-usage.sh stats: unknown sub-command '$sub' (save|diff)" >&2; return 2 ;;
  esac
}

cmd="${1:-detect}"; shift || true
case "$cmd" in
  detect)  cmd_collect --detect "$@" ;;
  collect) cmd_collect "$@" ;;
  stats)   cmd_stats "$@" ;;
  bench)   python3 -c "$PY_BENCH" "$ROLLUP" "$EVENTS" ;;
  -h|--help) sed -n '2,57p' "$0" ;;
  *) echo "collect-usage.sh: unknown command '$cmd' (detect|collect|stats|bench)" >&2; exit 2 ;;
esac
