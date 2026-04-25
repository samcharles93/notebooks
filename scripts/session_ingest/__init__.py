"""Session ingest pipeline for Mantle SFT corpus.

Parses agent transcripts from opencode, Claude Code, Codex CLI, and Copilot
into a canonical intermediate form, then redacts, filters, and formats them
for Qwen3.5-4B SFT training.

Pipeline stages (see plan/03-mantle-qa-synthesis.md):
    A. parse   - source-specific parser -> canonical Session records
    B. redact  - deterministic regex over all text fields
    C. filter  - cwd allowlist, drop rules, dedup
    D. format  - render to Qwen3.5 chat format (+ thinking variant)

CLI: python -m session_ingest.ingest <subcommand>
"""

__all__ = [
    "types",
    "redact",
    "filter",
    "format",
    "opencode",
    "claude",
    "codex",
    "copilot",
]
