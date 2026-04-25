"""Canonical dataclasses for the session-ingest pipeline.

All four source parsers (opencode/claude/codex/copilot) normalise their
native event streams into `Session` records built from `Turn` entries.
The format stage emits `SFTRecord` objects written to JSONL.

All fields are plain dataclasses with `to_dict`/`from_dict` for JSON
round-tripping; we avoid pydantic to keep the pipeline zero-dep beyond
the standard library.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


Role = Literal["system", "user", "assistant", "tool"]
Source = Literal["opencode", "claude", "codex", "copilot"]
ReasoningSource = Literal[
    "claude_thinking",
    "codex_reasoning",
    "opencode_reasoning_part",
    "none",
]


@dataclass
class ToolCall:
    """A single tool invocation issued by the assistant.

    `name` is the tool identifier as reported by the source (e.g. `bash`,
    `edit`, `read`). `arguments` is the raw JSON object passed by the model.
    `call_id` is the source-assigned correlation id used to match results.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    """Result of a tool invocation, keyed by `call_id`."""

    call_id: str
    output: str
    # Error string if the tool reported failure; None on success.
    error: Optional[str] = None


@dataclass
class Turn:
    """A single conversational turn.

    Tool calls and results are attached to the assistant turn that issued
    them; format.py splits them into synthetic assistant/tool messages
    matching the Qwen3.5 chat template.
    """

    role: Role
    content: str
    # Chain-of-thought text when the source preserves it (claude `thinking`,
    # codex `response_item.reasoning`, opencode `part.type=reasoning`).
    reasoning: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Unix ms timestamp of the turn when available; used only for ordering.
    timestamp_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.tool_calls:
            d["tool_calls"] = [dataclasses.asdict(tc) for tc in self.tool_calls]
        if self.tool_results:
            d["tool_results"] = [dataclasses.asdict(tr) for tr in self.tool_results]
        if self.timestamp_ms is not None:
            d["timestamp_ms"] = self.timestamp_ms
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        tcs = [ToolCall(**tc) for tc in d.get("tool_calls", [])]
        trs = [ToolResult(**tr) for tr in d.get("tool_results", [])]
        return cls(
            role=d["role"],
            content=d["content"],
            reasoning=d.get("reasoning"),
            tool_calls=tcs,
            tool_results=trs,
            timestamp_ms=d.get("timestamp_ms"),
        )


@dataclass
class Session:
    """A normalised conversation from any source."""

    source: Source
    session_id: str
    cwd: str
    created_at_ms: int
    turns: list[Turn] = field(default_factory=list)
    # Free-form source-specific metadata kept for debugging; never embedded
    # into SFT output.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "created_at_ms": self.created_at_ms,
            "turns": [t.to_dict() for t in self.turns],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        return cls(
            source=d["source"],
            session_id=d["session_id"],
            cwd=d.get("cwd", ""),
            created_at_ms=d.get("created_at_ms", 0),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            meta=d.get("meta", {}),
        )

    def dump_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class SFTRecord:
    """Final training record written to mantle-sft(-thinking).jsonl."""

    messages: list[dict[str, Any]]
    source: Source
    session_id: str
    cwd: str
    reasoning_source: ReasoningSource
    redacted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "source": self.source,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "reasoning_source": self.reasoning_source,
            "redacted": self.redacted,
        }

    def dump_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
