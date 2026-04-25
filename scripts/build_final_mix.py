#!/usr/bin/env python3
# Build final SFT mix: drop the 128 overlapping plain rows (whose sessions also
# have a thinking version), keep the 54 plain-only sessions + all 128 thinking
# rows. Result: 182 rows, ~73% thinking by token volume, no duplicate-trajectory
# pollution against the thinking corpus.

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAIN = ROOT / "data" / "corpus" / "mantle-sft.jsonl"
THINKING = ROOT / "data" / "corpus" / "mantle-sft-thinking.jsonl"
OUT = ROOT / "data" / "corpus" / "sft-final.jsonl"


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def char_count(row: dict) -> int:
    n = 0
    for m in row["messages"]:
        c = m.get("content") or ""
        if isinstance(c, list):
            c = json.dumps(c)
        n += len(c)
    return n


def main():
    plain = load(PLAIN)
    thinking = load(THINKING)
    thinking_sids = {r["session_id"] for r in thinking}

    plain_only = [r for r in plain if r["session_id"] not in thinking_sids]
    dropped = len(plain) - len(plain_only)

    final = plain_only + thinking

    with OUT.open("w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    plain_chars = sum(char_count(r) for r in plain_only)
    think_chars = sum(char_count(r) for r in thinking)
    total_chars = plain_chars + think_chars

    print(f"plain input:      {len(plain)} rows")
    print(f"thinking input:   {len(thinking)} rows")
    print(f"dropped (overlap): {dropped} rows")
    print(f"plain-only kept:  {len(plain_only)} rows")
    print(f"thinking kept:    {len(thinking)} rows")
    print(f"final total:      {len(final)} rows")
    print()
    print(f"plain-only chars: {plain_chars:,}")
    print(f"thinking chars:   {think_chars:,}")
    print(f"thinking share:   {100 * think_chars / total_chars:.1f}% (Unsloth target >=75%)")
    print(f"\nWrote {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
