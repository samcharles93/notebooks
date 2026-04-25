"""Claude Code session parser.

Reads JSONL files under `~/.claude/projects/<slugified-cwd>/<uuid>.jsonl`
and emits `Session` records. One file == one session; the filename stem
is the session UUID.

Each line is a JSON event. `event.type` values observed in the live
projects directory:

    user                    -- one user message (content str | content[])
    assistant               -- one assistant message (content[])
    system                  -- system/subtype messages (ignored)
    attachment              -- file attachment metadata (ignored)
    file-history-snapshot   -- editor state (ignored)
    permission-mode         -- UI state (ignored)
    queue-operation         -- UI state (ignored)
    last-prompt             -- UI state (ignored)

For `user`, `message.content` is either a string OR a list whose items
have `type ∈ {text, tool_result, image}`. `tool_result.content` is
either a string or a list of `{type: text, text: ...}` blocks; the
optional `is_error: true` flag maps to `ToolResult.error`.

For `assistant`, `message.content[]` items have
`type ∈ {text, thinking, tool_use}`:

    text      -- assistant prose (accumulated into Turn.content)
    thinking  -- chain-of-thought (accumulated into Turn.reasoning)
    tool_use  -- tool invocation (id/name/input), one ToolCall each

Tool results arrive on subsequent `user` events with matching
`tool_use_id`; they are attached to the preceding assistant turn that
issued the call so format.py can emit a coherent tool-call trajectory.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .types import Session, ToolCall, ToolResult, Turn


DEFAULT_CLAUDE_PROJECTS = Path.home() / ".claude/projects"


# Claude Code injects CLI scaffolding as `user` events. They are not real
# user input and every one of them would poison SFT: local-command-* leak
# shell state, command-{name,message,args} leak slash-command plumbing,
# bash-{input,stdout,stderr} leak tool I/O (redundant with tool_result),
# system-reminder leaks CLI control state, user-prompt-submit-hook leaks
# hook plumbing, Caveat:/[Request interrupted are abort/meta signals.
_INJECTED_USER_PREFIXES = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<system-reminder>",
    "<user-prompt-submit-hook>",
    "Caveat: The messages below",
    "[Request interrupted",
)


def _is_injected_user(text: str) -> bool:
    if not text:
        return True
    s = text.lstrip()
    return any(s.startswith(p) for p in _INJECTED_USER_PREFIXES)


def _projects_root() -> Path:
    env = os.environ.get("MANTLE_FT_CLAUDE_PROJECTS")
    return Path(env) if env else DEFAULT_CLAUDE_PROJECTS


def _parse_ts(v: Any) -> Optional[int]:
    if not isinstance(v, str) or not v:
        return None
    # Claude timestamps are ISO 8601 with trailing Z.
    s = v.replace("Z", "+00:00")
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


def _flatten_tool_result_content(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out: list[str] = []
        for item in c:
            if isinstance(item, dict):
                t = item.get("type")
                if t == "text":
                    v = item.get("text")
                    if isinstance(v, str):
                        out.append(v)
                else:
                    # Preserve unknown blocks as JSON so downstream filters
                    # can decide what to do rather than silently dropping.
                    out.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(out)
    if c is None:
        return ""
    return json.dumps(c, ensure_ascii=False)


def _user_turn_parts(
    content: Any,
) -> tuple[str, list[tuple[str, str, Optional[str]]]]:
    """Return (text_content, [(tool_use_id, output, error)])."""

    if isinstance(content, str):
        return content, []

    text_buf: list[str] = []
    tool_results: list[tuple[str, str, Optional[str]]] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                v = item.get("text")
                if isinstance(v, str) and v:
                    text_buf.append(v)
            elif t == "tool_result":
                tuid = item.get("tool_use_id")
                if not isinstance(tuid, str) or not tuid:
                    continue
                out = _flatten_tool_result_content(item.get("content"))
                err = "tool reported is_error" if item.get("is_error") else None
                tool_results.append((tuid, out, err))
            elif t == "image":
                # Images are not useful for text SFT; mark their presence so
                # assistants can see the user sent one without leaking bytes.
                text_buf.append("[image attachment]")
    return "\n\n".join(text_buf).strip(), tool_results


def _assistant_turn_parts(
    content: Any,
) -> tuple[str, Optional[str], list[ToolCall]]:
    """Return (text_content, reasoning, tool_calls)."""

    text_buf: list[str] = []
    think_buf: list[str] = []
    calls: list[ToolCall] = []
    if not isinstance(content, list):
        return "", None, []

    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            v = item.get("text")
            if isinstance(v, str) and v:
                text_buf.append(v)
        elif t == "thinking":
            v = item.get("thinking")
            if isinstance(v, str) and v.strip():
                think_buf.append(v)
        elif t == "tool_use":
            cid = item.get("id")
            name = item.get("name")
            if not isinstance(cid, str) or not isinstance(name, str):
                continue
            inp = item.get("input")
            if not isinstance(inp, dict):
                inp = {}
            calls.append(ToolCall(name=name, arguments=inp, call_id=cid))

    return (
        "\n\n".join(text_buf).strip(),
        ("\n\n".join(think_buf).strip() or None),
        calls,
    )


def _cwd_from_event(event: dict[str, Any]) -> str:
    v = event.get("cwd")
    return v if isinstance(v, str) else ""


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
    turns: list[Turn] = []
    cwd = ""
    created_at_ms = 0
    session_id = path.stem
    last_assistant: Optional[Turn] = None

    for ev in _iter_events(path):
        t = ev.get("type")
        if not cwd:
            c = _cwd_from_event(ev)
            if c:
                cwd = c
        if created_at_ms == 0:
            ts = _parse_ts(ev.get("timestamp"))
            if ts is not None:
                created_at_ms = ts

        if t == "user":
            msg = ev.get("message") or {}
            content = msg.get("content")
            text, tool_results = _user_turn_parts(content)

            # Attach tool results to the preceding assistant turn that
            # issued the matching tool_use_id; anything unattached is
            # dropped (no paired call).
            if tool_results and last_assistant is not None:
                known_ids = {tc.call_id for tc in last_assistant.tool_calls}
                for tuid, out, err in tool_results:
                    if tuid in known_ids:
                        last_assistant.tool_results.append(
                            ToolResult(call_id=tuid, output=out, error=err)
                        )

            if text and not _is_injected_user(text):
                turns.append(
                    Turn(
                        role="user",
                        content=text,
                        timestamp_ms=_parse_ts(ev.get("timestamp")),
                    )
                )
                last_assistant = None

        elif t == "assistant":
            msg = ev.get("message") or {}
            content = msg.get("content")
            text, reasoning, calls = _assistant_turn_parts(content)
            if not text and not reasoning and not calls:
                continue
            turn = Turn(
                role="assistant",
                content=text,
                reasoning=reasoning,
                tool_calls=calls,
                timestamp_ms=_parse_ts(ev.get("timestamp")),
            )
            turns.append(turn)
            last_assistant = turn

        # All other event types are intentionally ignored.

    if not turns:
        return None

    return Session(
        source="claude",
        session_id=session_id,
        cwd=cwd,
        created_at_ms=created_at_ms,
        turns=turns,
        meta={"path": str(path)},
    )


def parse_all(projects_root: Optional[Path] = None) -> list[Session]:
    """Parse every Claude Code session JSONL under the projects root."""

    root = projects_root or _projects_root()
    if not root.exists():
        return []

    sessions: list[Session] = []
    for f in sorted(root.glob("*/*.jsonl")):
        try:
            s = _parse_file(f)
        except OSError:
            continue
        if s is not None:
            sessions.append(s)
    return sessions
