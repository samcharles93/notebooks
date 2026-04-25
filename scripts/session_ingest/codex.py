"""Codex CLI session parser.

Reads `~/.codex/sessions/**/rollout-*.jsonl`. Each file is one session.

Event envelope has `type ∈ {session_meta, turn_context, event_msg,
response_item, compacted}`. Only `session_meta`, `turn_context`, and
`response_item` are used; UI-level `event_msg` entries are ignored and
`compacted` summaries are dropped to avoid double-counting.

`session_meta.payload` carries `id`, `cwd`, `timestamp`, and an opaque
`base_instructions` block (system-prompt text that MUST NOT enter the
SFT corpus).

`response_item.payload.type ∈ {message, reasoning, function_call,
function_call_output, custom_tool_call, custom_tool_call_output,
web_search_call}`:

    message                    role ∈ {developer, user, assistant};
                               content[] of {input_text|output_text|input_image}.
                               `developer` role is dropped (system prompt).
    reasoning                  `summary[]` items {type: summary_text, text}.
                               `content` is always null; `encrypted_content`
                               is opaque and intentionally skipped.
    function_call              {call_id, name, arguments:str(JSON)}.
    custom_tool_call           {call_id, name, input:str} — `input` is
                               not guaranteed to be JSON.
    *_output                   {call_id, output:str}.
    web_search_call            no useful content; skipped.

Reasoning summaries that immediately precede an assistant message are
merged into that turn's `reasoning`. Tool calls are paired with their
`_output` by `call_id` and attached to the issuing assistant turn.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .types import Session, ToolCall, ToolResult, Turn


DEFAULT_CODEX_SESSIONS = Path.home() / ".codex/sessions"

# Codex CLI injects several scaffolding messages as `user` turns. None of
# them are real user input and every one of them would poison SFT: the
# AGENTS.md block leaks repo instructions, environment_context leaks paths,
# IDE context leaks editor state, turn_aborted is a control signal, and
# the apply_patch warning is CLI tooling noise.
_INJECTED_USER_PREFIXES = (
    "# AGENTS.md instructions",
    "<environment_context>",
    "# Context from my IDE setup:",
    "<turn_aborted>",
    "Warning: apply_patch was requested",
)


def _sessions_root() -> Path:
    env = os.environ.get("MANTLE_FT_CODEX_SESSIONS")
    return Path(env) if env else DEFAULT_CODEX_SESSIONS


def _parse_ts(v: Any) -> Optional[int]:
    if not isinstance(v, str) or not v:
        return None
    s = v.replace("Z", "+00:00")
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


def _message_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in ("input_text", "output_text"):
            v = item.get("text")
            if isinstance(v, str) and v:
                parts.append(v)
        elif t == "input_image":
            parts.append("[image attachment]")
    return "\n\n".join(parts).strip()


def _reasoning_text(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if not isinstance(summary, list):
        return ""
    parts: list[str] = []
    for item in summary:
        if isinstance(item, dict) and item.get("type") == "summary_text":
            v = item.get("text")
            if isinstance(v, str) and v.strip():
                parts.append(v)
    return "\n\n".join(parts).strip()


def _parse_tool_arguments(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"_raw": v}
    except json.JSONDecodeError:
        return {"_raw": s}


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


def _parse_file(path: Path) -> Optional[Session]:
    session_id = ""
    cwd = ""
    created_at_ms = 0
    turns: list[Turn] = []

    pending_reasoning: list[str] = []
    tool_calls_by_id: dict[str, tuple[Turn, ToolCall]] = {}
    last_assistant: Optional[Turn] = None

    for ev in _iter_events(path):
        t = ev.get("type")

        if t == "session_meta":
            p = ev.get("payload") or {}
            pid = p.get("id")
            if isinstance(pid, str) and pid:
                session_id = pid
            pcwd = p.get("cwd")
            if isinstance(pcwd, str) and pcwd:
                cwd = pcwd
            ts = _parse_ts(p.get("timestamp") or ev.get("timestamp"))
            if ts is not None and created_at_ms == 0:
                created_at_ms = ts
            continue

        if t == "turn_context":
            p = ev.get("payload") or {}
            pcwd = p.get("cwd")
            if isinstance(pcwd, str) and pcwd and not cwd:
                cwd = pcwd
            continue

        if t != "response_item":
            continue

        payload = ev.get("payload") or {}
        ptype = payload.get("type")
        ts_ms = _parse_ts(ev.get("timestamp"))

        if ptype == "reasoning":
            rt = _reasoning_text(payload)
            if rt:
                pending_reasoning.append(rt)
            continue

        if ptype == "message":
            role = payload.get("role")
            text = _message_text(payload.get("content"))
            if role == "developer":
                pending_reasoning.clear()
                continue
            if role not in ("user", "assistant"):
                continue
            if not text and role == "user":
                continue
            if role == "user" and any(
                text.startswith(p) for p in _INJECTED_USER_PREFIXES
            ):
                continue
            if role == "assistant":
                reasoning = (
                    "\n\n".join(pending_reasoning).strip()
                    if pending_reasoning
                    else None
                )
                pending_reasoning.clear()
                turn = Turn(
                    role="assistant",
                    content=text,
                    reasoning=reasoning,
                    timestamp_ms=ts_ms,
                )
                turns.append(turn)
                last_assistant = turn
            else:
                pending_reasoning.clear()
                turns.append(
                    Turn(role="user", content=text, timestamp_ms=ts_ms)
                )
                last_assistant = None
            continue

        if ptype in ("function_call", "custom_tool_call"):
            call_id = payload.get("call_id")
            name = payload.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                continue
            if ptype == "function_call":
                args = _parse_tool_arguments(payload.get("arguments"))
            else:
                args = _parse_tool_arguments(payload.get("input"))
            if last_assistant is None:
                reasoning = (
                    "\n\n".join(pending_reasoning).strip()
                    if pending_reasoning
                    else None
                )
                pending_reasoning.clear()
                last_assistant = Turn(
                    role="assistant",
                    content="",
                    reasoning=reasoning,
                    timestamp_ms=ts_ms,
                )
                turns.append(last_assistant)
            tc = ToolCall(name=name, arguments=args, call_id=call_id)
            last_assistant.tool_calls.append(tc)
            tool_calls_by_id[call_id] = (last_assistant, tc)
            continue

        if ptype in ("function_call_output", "custom_tool_call_output"):
            call_id = payload.get("call_id")
            output = payload.get("output")
            if not isinstance(call_id, str):
                continue
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            pair = tool_calls_by_id.get(call_id)
            if pair is None:
                continue
            turn, _ = pair
            turn.tool_results.append(ToolResult(call_id=call_id, output=output))
            continue

        # web_search_call and any unknown payload types are intentionally skipped.

    if not turns:
        return None
    if not session_id:
        session_id = path.stem

    return Session(
        source="codex",
        session_id=session_id,
        cwd=cwd,
        created_at_ms=created_at_ms,
        turns=turns,
        meta={"path": str(path)},
    )


def parse_all(sessions_root: Optional[Path] = None) -> list[Session]:
    """Parse every Codex CLI rollout JSONL under the sessions root."""

    root = sessions_root or _sessions_root()
    if not root.exists():
        return []

    sessions: list[Session] = []
    for f in sorted(root.rglob("rollout-*.jsonl")):
        try:
            s = _parse_file(f)
        except OSError:
            continue
        if s is not None:
            sessions.append(s)
    return sessions
