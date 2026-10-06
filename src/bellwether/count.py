"""bellwether count: how many cases bellwether holds, per model, kind and source.

The source of a case is the dataset its corpus line was imported from (``origin.dataset``), or ``hand-written``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bellwether import jsonl, storage
from bellwether.manifest import KINDS, load_manifest
from bellwether.record import sets as set_tables

HAND_WRITTEN = "hand-written"


def set_sources(corpus: Path) -> dict[tuple[str, str], str]:
    """``(kind, set)`` -> the dataset a corpus set was imported from, read from the set's first record, in either form.

    A set is streamed, decompressed as it is read when compressed, and read no further than its first record: a source
    past ``corpus_sets.LIMIT`` can be gigabytes, and the first record names its dataset.
    """
    found: dict[tuple[str, str], str] = {}
    for kind in KINDS:
        if (corpus / kind).is_dir():
            for path in sorted(p for p in (corpus / kind).rglob("*") if p.is_file() and storage.stem(p) is not None):
                with storage.open_text(path) as lines:
                    _, first = next(jsonl.load(lines, path), (None, None))
                origin = first.get("origin") if first else None
                if isinstance(origin, dict):
                    found[(kind, storage.stem(path))] = origin["dataset"]
    return found


def counts(fixtures: Path, corpus: Path) -> list[dict]:
    """Cases per model, kind and source, from each model's ``sets.toml``: no fixture set is read."""
    source_of = set_sources(corpus)
    tally: Counter[tuple[str, str, str]] = Counter()
    for manifest_path in sorted(fixtures.glob("*/manifest.toml")):
        model = load_manifest(manifest_path).model
        for (kind, name), table in set_tables.read(manifest_path.parent / set_tables.FILE).items():
            tally[(model, kind, source_of.get((kind, name), HAND_WRITTEN))] += table["cases"]
    return [{"model": m, "kind": k, "source": s, "cases": n} for (m, k, s), n in sorted(tally.items())]


def render_markdown(rows: list[dict]) -> str:
    lines = ["| Model | Kind | Source | Cases |", "|---|---|---|---:|"]
    lines += [f"| {r['model']} | {r['kind']} | {r['source']} | {r['cases']} |" for r in rows]
    lines.append(f"| all | | | {sum(r['cases'] for r in rows)} |")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    rows = counts(args.fixtures, args.corpus)
    sys.stdout.write(json.dumps(rows, indent=1) + "\n" if args.format == "json" else render_markdown(rows))
    return 0
