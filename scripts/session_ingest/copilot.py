"""GitHub Copilot CLI session parser.

Reads `~/.copilot/session-state/<session_id>/events.jsonl`. Each directory
is one session; the JSONL is an event stream with envelope
`{id, parentId, timestamp, type, data}`.

Event types consumed:

    session.start              data.context.cwd, data.sessionId
    session.resume             data.context.cwd (overrides if later)
    user.message               data.content (real user input only)
    assistant.turn_start       turn boundary (unused; informational)
    assistant.message          data.content, data.reasoningText,
                               data.toolRequests[]
    assistant.turn_end         turn boundary (unused; informational)
    tool.execution_start       data.toolCallId, data.toolName, data.arguments
    tool.execution_complete    data.toolCallId, data.success, data.result
    abort                      control signal; turn is dropped if it lands
                               between user.message and assistant.message
    skill.invoked              data.content is the skill prompt body — kept
                               as assistant-visible content so the model
                               learns skill-triggered behaviour.

All other event types (`hook.*`, `system.notification`, `subagent.*`,
`session.compaction_*`, `session.mode_changed`, `session.plan_changed`,
`session.model_change`, `session.info`, `session.error`,
`session.shutdown`, `session.task_complete`, `tool.user_requested`) are
intentionally skipped: they are either telemetry, UI state, or
control-plane signals that do not belong in SFT.

`tool.execution_complete.success=false` maps to `ToolResult.error`.
`tool.execution_complete.result` is coerced to string via json.dumps when
non-string.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .types import Session, ToolCall, ToolResult, Turn


DEFAULT_COPILOT_SESSIONS = Path.home() / ".copilot/session-state"


def _sessions_root() -> Path:
    env = os.environ.get("MANTLE_FT_COPILOT_SESSIONS")
    return Path(env) if env else DEFAULT_COPILOT_SESSIONS


def _parse_ts(v: Any) -> Optional[int]:
    if not isinstance(v, str) or not v:
        return None
    s = v.replace("Z", "+00:00")
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


def _coerce_output(v: Any) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(v)


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _parse_file(events_path: Path, session_dir: Path) -> Optional[Session]:
    session_id = session_dir.name
    cwd = ""
    created_at_ms = 0
    turns: list[Turn] = []
    tool_calls_by_id: dict[str, tuple[Turn, ToolCall]] = {}
    last_assistant: Optional[Turn] = None
    aborted = False

    for ev in _iter_events(events_path):
        t = ev.get("type")
        data = ev.get("data") or {}
        ts_ms = _parse_ts(ev.get("timestamp"))

        if t in ("session.start", "session.resume"):
            ctx = data.get("context") or {}
            c = ctx.get("cwd")
            if isinstance(c, str) and c and not cwd:
                cwd = c
            sid = data.get("sessionId")
            if isinstance(sid, str) and sid:
                session_id = sid
            if ts_ms is not None and created_at_ms == 0:
                created_at_ms = ts_ms
            continue

        if t == "user.message":
            content = data.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            turns.append(
                Turn(role="user", content=content, timestamp_ms=ts_ms)
            )
            last_assistant = None
            aborted = False
            continue

        if t == "skill.invoked":
            content = data.get("content")
            name = data.get("name")
            if not isinstance(content, str) or not content.strip():
                continue
            header = f"[skill: {name}]\n" if isinstance(name, str) and name else ""
            turns.append(
                Turn(
                    role="user",
                    content=header + content,
                    timestamp_ms=ts_ms,
                )
            )
            last_assistant = None
            aborted = False
            continue

        if t == "abort":
            aborted = True
            continue

        if t == "assistant.message":
            if aborted:
                aborted = False
                continue
            content = data.get("content")
            if not isinstance(content, str):
                content = ""
            reasoning_text = data.get("reasoningText")
            reasoning = (
                reasoning_text
                if isinstance(reasoning_text, str) and reasoning_text.strip()
                else None
            )
            turn = Turn(
                role="assistant",
                content=content,
                reasoning=reasoning,
                timestamp_ms=ts_ms,
            )
            turns.append(turn)
            last_assistant = turn
            continue

        if t == "tool.execution_start":
            call_id = data.get("toolCallId")
            name = data.get("toolName")
            args = data.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str):
                continue
            if not isinstance(args, dict):
                args = {"_raw": args} if args is not None else {}
            if last_assistant is None:
                last_assistant = Turn(
                    role="assistant",
                    content="",
                    timestamp_ms=ts_ms,
                )
                turns.append(last_assistant)
            tc = ToolCall(name=name, arguments=args, call_id=call_id)
            last_assistant.tool_calls.append(tc)
            tool_calls_by_id[call_id] = (last_assistant, tc)
            continue

        if t == "tool.execution_complete":
            call_id = data.get("toolCallId")
            if not isinstance(call_id, str):
                continue
            pair = tool_calls_by_id.get(call_id)
            if pair is None:
                continue
            turn, _ = pair
            output = _coerce_output(data.get("result"))
            success = data.get("success")
            err = None if success is not False else "tool reported success=false"
            turn.tool_results.append(
                ToolResult(call_id=call_id, output=output, error=err)
            )
            continue

    if not turns:
        return None

    return Session(
        source="copilot",
        session_id=session_id,
        cwd=cwd,
        created_at_ms=created_at_ms,
        turns=turns,
        meta={"path": str(events_path)},
    )


def parse_all(sessions_root: Optional[Path] = None) -> list[Session]:
    """Parse every Copilot CLI session directory under the sessions root."""

    root = sessions_root or _sessions_root()
    if not root.exists():
        return []

    sessions: list[Session] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        events = d / "events.jsonl"
        if not events.exists():
            continue
        try:
            s = _parse_file(events, d)
        except OSError:
            continue
        if s is not None:
            sessions.append(s)
    return sessions
