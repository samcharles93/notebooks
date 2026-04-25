"""Session filter: cwd allowlist, drop rules, and first-turn dedup.

Allowlist is seeded from the opencode directory census (see handoff
"Critical Context"). The fallback path-based scope check scans tool-call
arguments for Go/systems file extensions so sessions rooted outside the
allowlist but primarily editing in-scope files still qualify.

Drop rules:
    - fewer than 2 user turns
    - every assistant turn empty or aborted

Dedup:
    - exact match on the first user turn's content (opencode "continue"
      duplication artefact).
"""

from __future__ import annotations

import os
from typing import Iterable

from .types import Session


CWD_ALLOWLIST: tuple[str, ...] = (
    "/work/apps/mantle",
    "/work/apps/knowledge-builder",
    "/work/apps/knowledge-builder/attention",
    "/work/experiment",
    "/work/experiment/engines/ageLLM",
    "/work/apps/do",
    "/work/apps/oh-my-copilot",
    "/work/scratch/lume-engine",
    "/work/scratch/corpus_building",
    "/work/gamedev/silver-hills",
    "/work/gamedev/lume-engine",
    "/work/clones/stable-diffusion.cpp",
    "/work/clones/ollama/app",
)


# File extensions considered in-scope for the "tool path fallback" heuristic.
IN_SCOPE_SUFFIXES: tuple[str, ...] = (
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cu",
    ".cuh",
    ".rs",
    ".zig",
    "Taskfile.yml",
    "go.mod",
    "go.sum",
)


def _cwd_allowed(cwd: str) -> bool:
    cwd = cwd.rstrip("/")
    return any(cwd == a or cwd.startswith(a + "/") for a in CWD_ALLOWLIST)


def _touches_in_scope_file(session: Session) -> bool:
    """Scan tool-call arguments for in-scope file paths."""

    for turn in session.turns:
        for call in turn.tool_calls:
            for v in _iter_strings(call.arguments):
                if any(v.endswith(sfx) or sfx in v for sfx in IN_SCOPE_SUFFIXES):
                    return True
    return False


def _iter_strings(obj: object) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)


def in_scope(session: Session) -> bool:
    """Return True if the session should be kept."""

    if _cwd_allowed(session.cwd):
        return True
    # Fallback: session cwd may be an IDE root outside the allowlist (e.g.
    # `/work`) but the assistant edited Go/systems files in-scope.
    return _touches_in_scope_file(session)


def passes_quality_gate(session: Session) -> bool:
    """Drop thin or all-aborted sessions."""

    user_turns = sum(1 for t in session.turns if t.role == "user" and t.content.strip())
    if user_turns < 2:
        return False
    # Assistant turns must have at least one non-empty content or tool call.
    asst_signal = sum(
        1
        for t in session.turns
        if t.role == "assistant" and (t.content.strip() or t.tool_calls)
    )
    if asst_signal == 0:
        return False
    return True


def dedup_first_user_turn(sessions: list[Session]) -> list[Session]:
    """Remove sessions whose first user turn duplicates an earlier session's.

    Sessions are sorted by `created_at_ms` so the earliest survives, which
    matches the opencode "continue" semantics where the latest copy is the
    resumed branch. (Keeping the earliest discards the resumed branches that
    would otherwise inflate duplicate prompts in training.)
    """

    seen: set[str] = set()
    out: list[Session] = []
    for s in sorted(sessions, key=lambda x: x.created_at_ms):
        first_user = next(
            (t.content.strip() for t in s.turns if t.role == "user" and t.content.strip()),
            "",
        )
        key = first_user
        if not key:
            out.append(s)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def filter_sessions(sessions: list[Session]) -> list[Session]:
    """Apply scope, quality, and dedup filters in order."""

    kept = [s for s in sessions if in_scope(s) and passes_quality_gate(s)]
    return dedup_first_user_turn(kept)


# Overridable via env var for debugging; normalise trailing slash.
def configured_allowlist() -> tuple[str, ...]:
    env = os.environ.get("MANTLE_FT_CWD_ALLOWLIST")
    if not env:
        return CWD_ALLOWLIST
    return tuple(p.rstrip("/") for p in env.split(":") if p.strip())
