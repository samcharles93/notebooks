"""opencode session parser.

Reads `~/.local/share/opencode/opencode.db` (SQLite) and emits `Session`
records. The DB schema (verified against the live file) is:

    session(id TEXT PK, project_id, parent_id, slug, directory, title,
            version, share_url, summary_*, revert, permission, time_*,
            workspace_id)
    message(id TEXT PK, session_id, time_created, time_updated, data JSON)
    part(id TEXT PK, message_id, session_id, time_created, time_updated,
         data JSON)

`message.data` carries `role`, `modelID`, `path.cwd`, `agent`, `mode`,
`tokens`, `finish`. `part.data.type` is the dispatch key; the values we
care about are:

    step-start    -- boundary marker, ignored
    reasoning     -- chain-of-thought text (kept as `Turn.reasoning`)
    text          -- assistant or user content
    tool          -- tool invocation with input/output
    step-finish   -- token accounting + snapshot ref, ignored for turns
    patch         -- diff payload (treated as text content)
    compaction    -- context-window compaction artefact, skipped
    file          -- attached file reference, skipped (handled by tool parts)

Reasoning arrives as its own `reasoning` part before the matching `text`
part of the same assistant message; we merge them into one `Turn` keyed
by `message_id` so `Turn.reasoning` accompanies the assistant answer.

Tool parts carry both `input` (arguments) and `output` on a single
record, so one opencode tool part becomes one `ToolCall` plus one
`ToolResult`; opencode does not provide a correlation id, so we
synthesise `call_id = part.id` to keep the two halves linked downstream.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from .types import Session, ToolCall, ToolResult, Turn


DEFAULT_DB_PATH = Path.home() / ".local/share/opencode/opencode.db"


def _db_path() -> Path:
    env = os.environ.get("MANTLE_FT_OPENCODE_DB")
    return Path(env) if env else DEFAULT_DB_PATH


def _json_loads(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) else {}


def _cwd_from_message(msg_data: dict[str, Any], session_directory: str) -> str:
    # `path.cwd` is the authoritative per-message cwd; `session.directory`
    # is the fallback for older rows that pre-date the `path` field.
    path = msg_data.get("path") or {}
    if isinstance(path, dict):
        cwd = path.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return session_directory or ""


def _role_of(msg_data: dict[str, Any]) -> str:
    role = msg_data.get("role")
    if isinstance(role, str) and role in ("user", "assistant", "system", "tool"):
        return role
    # Some older rows omit role; the presence of `modelID` implies assistant.
    return "assistant" if msg_data.get("modelID") else "user"


def _tool_call_from_part(part_data: dict[str, Any], part_id: str) -> Optional[ToolCall]:
    name = part_data.get("tool") or part_data.get("name")
    if not isinstance(name, str):
        return None
    state = part_data.get("state") or {}
    args = state.get("input") if isinstance(state, dict) else None
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=name, arguments=args, call_id=part_id)


def _tool_result_from_part(part_data: dict[str, Any], part_id: str) -> Optional[ToolResult]:
    state = part_data.get("state") or {}
    if not isinstance(state, dict):
        return None
    output = state.get("output")
    if output is None:
        return None
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    err = state.get("error") if isinstance(state.get("error"), str) else None
    return ToolResult(call_id=part_id, output=output, error=err)


def _text_of_part(part_data: dict[str, Any]) -> str:
    t = part_data.get("type")
    if t == "text":
        v = part_data.get("text")
        return v if isinstance(v, str) else ""
    if t == "patch":
        # `patch` parts carry a unified-diff-style payload in `hash`/`files`;
        # serialise minimally so format.py sees the intent without dumping
        # the whole snapshot.
        files = part_data.get("files")
        if isinstance(files, list) and files:
            return "[patch]\n" + "\n".join(str(f) for f in files)
        return "[patch]"
    return ""


def _build_turn(
    message_id: str,
    msg_data: dict[str, Any],
    parts: list[dict[str, Any]],
    part_ids: list[str],
) -> Optional[Turn]:
    role = _role_of(msg_data)
    if role not in ("user", "assistant", "system", "tool"):
        return None

    reasoning_buf: list[str] = []
    text_buf: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []

    for pid, pd in zip(part_ids, parts):
        t = pd.get("type")
        if t == "reasoning":
            v = pd.get("text")
            if isinstance(v, str) and v.strip():
                reasoning_buf.append(v)
        elif t in ("text", "patch"):
            s = _text_of_part(pd)
            if s:
                text_buf.append(s)
        elif t == "tool":
            tc = _tool_call_from_part(pd, pid)
            if tc is not None:
                tool_calls.append(tc)
            tr = _tool_result_from_part(pd, pid)
            if tr is not None:
                tool_results.append(tr)
        # step-start / step-finish / compaction / file are intentionally ignored.

    content = "\n\n".join(text_buf).strip()
    reasoning = "\n\n".join(reasoning_buf).strip() or None
    if not content and not reasoning and not tool_calls:
        return None

    ts = msg_data.get("time")
    timestamp_ms: Optional[int] = None
    if isinstance(ts, dict):
        v = ts.get("created")
        if isinstance(v, (int, float)):
            timestamp_ms = int(v)

    return Turn(
        role=role,  # type: ignore[arg-type]
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        tool_results=tool_results,
        timestamp_ms=timestamp_ms,
    )


def _iter_session_rows(conn: sqlite3.Connection):
    cur = conn.execute(
        "SELECT id, directory, time_created FROM session ORDER BY time_created ASC"
    )
    for row in cur:
        yield row


def _load_session(conn: sqlite3.Connection, session_id: str, directory: str, created_at_ms: int) -> Optional[Session]:
    msg_cur = conn.execute(
        "SELECT id, time_created, data FROM message "
        "WHERE session_id = ? ORDER BY time_created ASC",
        (session_id,),
    )
    msg_rows = msg_cur.fetchall()
    if not msg_rows:
        return None

    part_cur = conn.execute(
        "SELECT id, message_id, data FROM part "
        "WHERE session_id = ? ORDER BY time_created ASC",
        (session_id,),
    )
    parts_by_msg: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for pid, mid, pdata in part_cur:
        parts_by_msg.setdefault(mid, []).append((pid, _json_loads(pdata)))

    turns: list[Turn] = []
    cwd = ""
    for mid, mtime, mdata in msg_rows:
        md = _json_loads(mdata)
        if not cwd:
            cwd = _cwd_from_message(md, directory)
        entries = parts_by_msg.get(mid, [])
        part_ids = [e[0] for e in entries]
        part_data = [e[1] for e in entries]
        turn = _build_turn(mid, md, part_data, part_ids)
        if turn is not None:
            turns.append(turn)

    if not turns:
        return None

    return Session(
        source="opencode",
        session_id=session_id,
        cwd=cwd or directory or "",
        created_at_ms=int(created_at_ms) if isinstance(created_at_ms, (int, float)) else 0,
        turns=turns,
        meta={"directory": directory},
    )


def parse_all(db_path: Optional[Path] = None) -> list[Session]:
    """Parse every session in the opencode DB (read-only)."""

    path = db_path or _db_path()
    if not path.exists():
        return []

    # Read-only URI connection so a concurrent opencode process cannot
    # block us and we cannot corrupt its DB.
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        sessions: list[Session] = []
        for sid, directory, created in _iter_session_rows(conn):
            s = _load_session(conn, sid, directory or "", created or 0)
            if s is not None:
                sessions.append(s)
        return sessions
    finally:
        conn.close()
