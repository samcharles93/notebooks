from __future__ import annotations

import json
from typing import Any, Optional

from .types import SFTRecord, Session, ToolCall, ToolResult, Turn


_REASONING_SOURCE_BY_INGEST_SOURCE: dict[str, str] = {
    "claude": "claude_thinking",
    "codex": "codex_reasoning",
    "opencode": "opencode_reasoning_part",
    "copilot": "copilot_reasoning_text",
}


def _tool_call_message(tc: ToolCall) -> dict[str, Any]:
    try:
        args_str = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        args_str = json.dumps({"_raw": str(tc.arguments)}, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": args_str},
            }
        ],
    }


def _tool_result_message(tr: ToolResult) -> dict[str, Any]:
    content = tr.output if tr.output else ""
    if tr.error:
        content = f"[error: {tr.error}]\n{content}".rstrip()
    return {
        "role": "tool",
        "tool_call_id": tr.call_id,
        "content": content,
    }


def _has_any_reasoning(session: Session) -> bool:
    return any(
        t.role == "assistant" and t.reasoning and t.reasoning.strip()
        for t in session.turns
    )


def _assistant_messages(turn: Turn, include_thinking: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    text = turn.content or ""
    reasoning = (turn.reasoning or "").strip() if include_thinking else ""

    if text.strip() or reasoning:
        content = f"<think>\n{reasoning}\n</think>\n\n{text}" if reasoning else text
        out.append({"role": "assistant", "content": content})

    for tc in turn.tool_calls:
        out.append(_tool_call_message(tc))

    return out


def _tool_result_messages(turn: Turn) -> list[dict[str, Any]]:
    return [_tool_result_message(tr) for tr in turn.tool_results]


def _user_message(turn: Turn) -> Optional[dict[str, Any]]:
    text = (turn.content or "").strip()
    if not text:
        return None
    return {"role": "user", "content": turn.content}


def _build_messages(session: Session, include_thinking: bool) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in session.turns:
        if turn.role == "user":
            m = _user_message(turn)
            if m is not None:
                messages.append(m)
            continue

        if turn.role == "assistant":
            messages.extend(_assistant_messages(turn, include_thinking))
            messages.extend(_tool_result_messages(turn))
            continue

    return messages


def _has_user_and_assistant(messages: list[dict[str, Any]]) -> bool:
    roles = {m["role"] for m in messages}
    return "user" in roles and "assistant" in roles


def to_sft_record(session: Session, include_thinking: bool = False) -> Optional[SFTRecord]:
    messages = _build_messages(session, include_thinking=include_thinking)
    if not messages or not _has_user_and_assistant(messages):
        return None

    ingest_source = session.source
    if include_thinking:
        reasoning_source = _REASONING_SOURCE_BY_INGEST_SOURCE.get(
            ingest_source, "none"
        )
    else:
        reasoning_source = "none"

    return SFTRecord(
        messages=messages,
        source=ingest_source,
        session_id=session.session_id,
        cwd=session.cwd,
        reasoning_source=reasoning_source,
        redacted=True,
    )


def to_plain_record(session: Session) -> Optional[SFTRecord]:
    return to_sft_record(session, include_thinking=False)


def to_thinking_record(session: Session) -> Optional[SFTRecord]:
    if not _has_any_reasoning(session):
        return None
    return to_sft_record(session, include_thinking=True)
