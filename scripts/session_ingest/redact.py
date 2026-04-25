"""Deterministic regex redaction for session text.

Ordering matters: token patterns run before the path scrub so that
Bearer-/sk- tokens with `/home/sam/` suffixes are caught as tokens first.
All patterns operate on arbitrary text; callers apply `scrub_text` to every
string field of a `Session` before the filter/format stages.

Policy (see notes/decisions.md pivot entry):
    - absolute user paths -> `/home/USER/` or `/Users/USER`
    - email addresses     -> `<email>`
    - API tokens          -> `<token>`
    - Bearer headers      -> `Bearer <token>`
    - AWS access keys     -> `<token>`
    - VAR=value assignments are intentionally preserved (signal > risk).
"""

from __future__ import annotations

import re
from typing import Any

from .types import Session, Turn, ToolCall, ToolResult


# Order-sensitive. Each entry is (pattern, replacement).
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # API tokens (run before path/email so trailing paths in tokens are caught).
    (re.compile(r"(?:sk-|ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9]{20,}"), "<token>"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"), "Bearer <token>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<token>"),
    # Email.
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "<email>"),
    # User home paths. `\b` would match `/home/sama` so use an explicit
    # follow-up char class: path separator, whitespace, or string end.
    (re.compile(r"/home/sam(?=[/\s\"']|$)"), "/home/USER"),
    (re.compile(r"/Users/[^/\s\"']+"), "/Users/USER"),
]


def scrub_text(text: str) -> str:
    """Apply all redaction patterns in order."""

    if not text:
        return text
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def _scrub_args(args: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub all string values in a JSON-like dict."""

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return scrub_text(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v

    return walk(args)  # type: ignore[no-any-return]


def scrub_turn(turn: Turn) -> Turn:
    """Return a new Turn with every text field redacted."""

    return Turn(
        role=turn.role,
        content=scrub_text(turn.content),
        reasoning=scrub_text(turn.reasoning) if turn.reasoning else None,
        tool_calls=[
            ToolCall(name=tc.name, arguments=_scrub_args(tc.arguments), call_id=tc.call_id)
            for tc in turn.tool_calls
        ],
        tool_results=[
            ToolResult(
                call_id=tr.call_id,
                output=scrub_text(tr.output),
                error=scrub_text(tr.error) if tr.error else None,
            )
            for tr in turn.tool_results
        ],
        timestamp_ms=turn.timestamp_ms,
    )


def scrub_session(session: Session) -> Session:
    """Return a new Session with every text field redacted."""

    return Session(
        source=session.source,
        session_id=session.session_id,
        cwd=scrub_text(session.cwd),
        created_at_ms=session.created_at_ms,
        turns=[scrub_turn(t) for t in session.turns],
        meta=session.meta,
    )
