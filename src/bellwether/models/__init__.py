"""bellwether models: the list of every generative checkpoint to record, one row per checkpoint.

The list comes from the engines' registries at pinned commits and from the Hugging Face Hub as it
is today; see ``docs/benchmark-sets.md``, "Which models", for the rule and ``rules.py`` for its
pieces. The rows go to a JSON Lines file, a summary to stdout, and notes to stderr.

Without the Hub the list depends on the pins alone, so it is committed as ``models.jsonl`` and CI
builds it again and compares (``--check``). A list read from the Hub is that day's evidence: it goes
under ``runs/`` with its date and is published to smg-project/artifacts, never committed here.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import registry
from .build import Row, header, hub_rows, missing_from_tier1, registry_only_rows, set_aside, to_jsonl, unnamed
from .hub import HfHub
from .pins import PinError
from .registry import read_pinned

FAILED = 1
USAGE_ERROR = 2
COMMITTED = Path("models.jsonl")  # the registry-only list, at the repository's root


def run(args: argparse.Namespace) -> int:
    if args.check and not args.registry_only:
        print(
            "bellwether models: --check needs --registry-only; a list read from the Hub changes every day",
            file=sys.stderr,
        )
        return USAGE_ERROR
    try:
        entries, served = read_pinned(args.cache, args.vllm_src, args.sglang_src)
    except PinError as err:
        print(f"bellwether models: {err}", file=sys.stderr)
        return FAILED
    built = datetime.now(UTC).date()
    pins = (registry.VLLM, registry.SGLANG)
    if args.registry_only:
        rows = registry_only_rows(entries, built, served)
        text = header(pins, None) + to_jsonl(rows)
        out = args.out or COMMITTED
    else:
        rows = hub_rows(entries, HfHub(), built, log=_note, served=served)
        text = header(pins, built) + to_jsonl(rows)
        out = args.out or Path("runs") / f"models-{built.isoformat()}.jsonl"
    if args.check:
        return _check(out, text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    for note in unnamed(entries):
        _note(note)
    aside = set_aside(entries)
    if aside:
        _note(f"gpt-oss, set aside: {', '.join(aside)}")
    missing = missing_from_tier1(rows)
    if missing:
        _note(f"tier 1 names missing from the list: {', '.join(missing)}")
    print(_summary(rows, out))
    return 0


def _check(out: Path, text: str) -> int:
    """Compare a fresh registry-only list with the committed one, as the importers check their sets."""
    if out.is_file() and out.read_text() == text:
        print(f"{out}: equals a fresh build at the pins")
        return 0
    print(
        f"{out}: differs from a fresh build at the pins; run `bellwether models --registry-only` and commit it",
        file=sys.stderr,
    )
    return FAILED


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def _summary(rows: list[Row], out: Path) -> str:
    tiers = Counter(row.tier for row in rows)
    modalities = Counter(row.modality for row in rows)
    statuses = Counter(row.status for row in rows)
    return "\n".join(
        [
            f"bellwether models: {len(rows)} rows in {out}",
            f"  tier 1: {tiers[1]}, tier 2: {tiers[2]}, tier 3: {tiers[3]}"
            + (f", left to the Hub: {tiers[None]}" if tiers[None] else ""),
            f"  text: {modalities['text']}, multimodal: {modalities['multimodal']}"
            + (f", left to the Hub: {modalities[None]}" if modalities[None] else ""),
            "  status: " + ", ".join(f"{status} {count}" for status, count in sorted(statuses.items())),
        ]
    )
