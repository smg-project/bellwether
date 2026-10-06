"""bellwether models: the list of every generative checkpoint to record, one row per checkpoint.

The list comes from the engines' registries at pinned commits and from the Hugging Face Hub as it
is today; see ``docs/benchmark-sets.md``, "Which models", for the rule and ``rules.py`` for its
pieces. The rows go to a JSON Lines file, a summary to stdout, and notes to stderr.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .build import Row, hub_rows, missing_from_tier1, registry_only_rows, to_jsonl, unnamed
from .hub import HfHub
from .pins import PinError
from .registry import read_pinned

FAILED = 1


def run(args: argparse.Namespace) -> int:
    try:
        entries = read_pinned(args.cache, args.vllm_src, args.sglang_src)
    except PinError as err:
        print(f"bellwether models: {err}", file=sys.stderr)
        return FAILED
    built = datetime.now(UTC).date()
    if args.registry_only:
        rows = registry_only_rows(entries, built)
    else:
        rows = hub_rows(entries, HfHub(), built, log=_note)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_jsonl(rows))
    for note in unnamed(entries):
        _note(note)
    missing = missing_from_tier1(rows)
    if missing:
        _note(f"tier 1 names missing from the list: {', '.join(missing)}")
    print(_summary(rows, args.out))
    return 0


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def _summary(rows: list[Row], out: Path) -> str:
    tiers = Counter(row.tier for row in rows)
    modalities = Counter(row.modality for row in rows)
    statuses = Counter(row.status for row in rows)
    return "\n".join(
        [
            f"bellwether models: {len(rows)} rows in {out}",
            f"  tier 1: {tiers[1]}, tier 2: {tiers[2]}, tier 3: {tiers[3]}",
            f"  text: {modalities['text']}, multimodal: {modalities['multimodal']}",
            "  status: " + ", ".join(f"{status} {count}" for status, count in sorted(statuses.items())),
        ]
    )
