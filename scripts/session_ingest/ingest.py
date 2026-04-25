from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

from . import claude, codex, copilot, opencode
from .filter import filter_sessions
from .format import to_plain_record, to_thinking_record
from .redact import scrub_session
from .types import SFTRecord, Session


SOURCES = ("opencode", "claude", "codex", "copilot")

DEFAULT_OUT_DIR = Path("data/corpus")
DEFAULT_SESSIONS_DIR = DEFAULT_OUT_DIR / "sessions"


def _parse_source(name: str) -> list[Session]:
    if name == "opencode":
        return opencode.parse_all()
    if name == "claude":
        return claude.parse_all()
    if name == "codex":
        return codex.parse_all()
    if name == "copilot":
        return copilot.parse_all()
    raise ValueError(f"unknown source: {name}")


def _parse_all(names: Iterable[str]) -> list[Session]:
    out: list[Session] = []
    for n in names:
        out.extend(_parse_source(n))
    return out


def _dedup_by_first_user(sessions: list[Session]) -> list[Session]:
    seen: dict[str, Session] = {}
    for s in sorted(sessions, key=lambda x: x.created_at_ms or 0):
        first_user = next(
            (t.content for t in s.turns if t.role == "user" and t.content.strip()),
            None,
        )
        if not first_user:
            continue
        key = hashlib.sha256(first_user.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen[key] = s
    return list(seen.values())


def _write_sessions(sessions: list[Session], sessions_dir: Path) -> int:
    count = 0
    for s in sessions:
        sub = sessions_dir / s.source
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{s.session_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(s.to_dict(), fh, ensure_ascii=False)
        count += 1
    return count


def _load_sessions(sessions_dir: Path) -> list[Session]:
    out: list[Session] = []
    if not sessions_dir.exists():
        return out
    for src_dir in sorted(sessions_dir.iterdir()):
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.glob("*.json")):
            with p.open("r", encoding="utf-8") as fh:
                out.append(Session.from_dict(json.load(fh)))
    return out


def _write_jsonl(records: list[SFTRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def _write_combined(
    plain: list[SFTRecord], thinking: list[SFTRecord], path: Path
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in plain:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        for r in thinking:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(plain) + len(thinking)


def _manifest(
    plain: list[SFTRecord],
    thinking: list[SFTRecord],
    sources_counts: dict[str, int],
) -> dict:
    def _by(records: list[SFTRecord], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in records:
            v = getattr(r, key)
            out[v] = out.get(v, 0) + 1
        return out

    return {
        "sessions_parsed_by_source": sources_counts,
        "plain_records": len(plain),
        "thinking_records": len(thinking),
        "plain_by_source": _by(plain, "source"),
        "thinking_by_reasoning_source": _by(thinking, "reasoning_source"),
        "schema": {
            "record": {
                "messages": "list[{role, content, tool_calls?, tool_call_id?}]",
                "source": "opencode|claude|codex|copilot",
                "session_id": "str",
                "cwd": "str",
                "reasoning_source": (
                    "none|claude_thinking|codex_reasoning|"
                    "opencode_reasoning_part|copilot_reasoning_text"
                ),
                "redacted": "bool",
            }
        },
    }


def cmd_all(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    sessions_dir = out_dir / "sessions"
    sources = args.sources or list(SOURCES)

    print(f"[parse]  sources={sources}")
    sessions = _parse_all(sources)
    by_source = {n: sum(1 for s in sessions if s.source == n) for n in sources}
    print(f"[parse]  parsed {len(sessions)} sessions {by_source}")

    print("[filter] applying cwd allowlist + min-turn filter")
    sessions = filter_sessions(sessions)
    print(f"[filter] kept {len(sessions)} sessions")

    print("[dedup]  first-user-turn exact-match, earliest wins")
    sessions = _dedup_by_first_user(sessions)
    print(f"[dedup]  kept {len(sessions)} sessions")

    print("[redact] scrubbing tokens, emails, home paths")
    sessions = [scrub_session(s) for s in sessions]

    if args.persist_sessions:
        n = _write_sessions(sessions, sessions_dir)
        print(f"[persist] wrote {n} session JSONs to {sessions_dir}")

    print("[format] rendering Qwen3.5 messages (plain + thinking)")
    plain = [r for r in (to_plain_record(s) for s in sessions) if r]
    thinking = [r for r in (to_thinking_record(s) for s in sessions) if r]

    plain_path = out_dir / "mantle-sft.jsonl"
    thinking_path = out_dir / "mantle-sft-thinking.jsonl"
    combined_path = out_dir / "sft.jsonl"
    manifest_path = out_dir / "mantle-sft.manifest.json"

    _write_jsonl(plain, plain_path)
    _write_jsonl(thinking, thinking_path)
    _write_combined(plain, thinking, combined_path)

    manifest = _manifest(plain, thinking, by_source)
    manifest["outputs"] = {
        "plain": str(plain_path),
        "thinking": str(thinking_path),
        "combined": str(combined_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"[write]  {plain_path} ({len(plain)} records)")
    print(f"[write]  {thinking_path} ({len(thinking)} records)")
    print(f"[write]  {combined_path} ({len(plain) + len(thinking)} records)")
    print(f"[write]  {manifest_path}")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    sessions_dir = out_dir / "sessions"
    sources = args.sources or list(SOURCES)
    sessions = _parse_all(sources)
    n = _write_sessions(sessions, sessions_dir)
    print(f"[parse] wrote {n} sessions to {sessions_dir}")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    sessions_dir = Path(args.out_dir) / "sessions"
    sessions = _load_sessions(sessions_dir)
    kept = filter_sessions(sessions)
    kept = _dedup_by_first_user(kept)
    for p in sessions_dir.rglob("*.json"):
        p.unlink()
    n = _write_sessions(kept, sessions_dir)
    print(f"[filter] kept {n} sessions (from {len(sessions)})")
    return 0


def cmd_redact(args: argparse.Namespace) -> int:
    sessions_dir = Path(args.out_dir) / "sessions"
    sessions = _load_sessions(sessions_dir)
    scrubbed = [scrub_session(s) for s in sessions]
    for p in sessions_dir.rglob("*.json"):
        p.unlink()
    n = _write_sessions(scrubbed, sessions_dir)
    print(f"[redact] rewrote {n} sessions in-place")
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    sessions = _load_sessions(out_dir / "sessions")
    plain = [r for r in (to_plain_record(s) for s in sessions) if r]
    thinking = [r for r in (to_thinking_record(s) for s in sessions) if r]
    _write_jsonl(plain, out_dir / "mantle-sft.jsonl")
    _write_jsonl(thinking, out_dir / "mantle-sft-thinking.jsonl")
    _write_combined(plain, thinking, out_dir / "sft.jsonl")
    by_source = {n: sum(1 for s in sessions if s.source == n) for n in SOURCES}
    manifest = _manifest(plain, thinking, by_source)
    manifest["outputs"] = {
        "plain": str(out_dir / "mantle-sft.jsonl"),
        "thinking": str(out_dir / "mantle-sft-thinking.jsonl"),
        "combined": str(out_dir / "sft.jsonl"),
    }
    with (out_dir / "mantle-sft.manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(
        f"[format] wrote {len(plain)} plain + {len(thinking)} thinking "
        f"({len(plain) + len(thinking)} combined) records"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="session_ingest")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="output directory (default: data/corpus)",
    )
    p.add_argument(
        "--sources",
        nargs="*",
        choices=SOURCES,
        help="subset of sources (default: all four)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    all_p = sub.add_parser("all", help="run full pipeline end-to-end")
    all_p.add_argument(
        "--persist-sessions",
        action="store_true",
        help="also write per-session JSONs under sessions/ (for resumable runs)",
    )
    all_p.set_defaults(func=cmd_all)

    sub.add_parser("parse", help="parse raw sources -> sessions/").set_defaults(
        func=cmd_parse
    )
    sub.add_parser(
        "filter", help="filter + dedup sessions/ in-place"
    ).set_defaults(func=cmd_filter)
    sub.add_parser(
        "redact", help="redact sessions/ in-place"
    ).set_defaults(func=cmd_redact)
    sub.add_parser(
        "format", help="render sessions/ into SFT JSONL + manifest"
    ).set_defaults(func=cmd_format)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
