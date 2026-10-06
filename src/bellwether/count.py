"""bellwether count: how many cases bellwether holds, per checkpoint, kind and source.

The source of a case is the dataset its corpus line was imported from (``origin.dataset``), or ``hand-written``. Every
checkpoint with a manifest is a row, as in the design's per-model table: a group's members show the cases of their
group, which is recorded once, and name it; a checkpoint whose group has recorded nothing yet still has a row, with no
kind and no cases. The ``all`` line counts each group's cases once.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from bellwether import jsonl, storage
from bellwether.manifest import KINDS, load_manifests
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
    """Cases per checkpoint, kind and source, from each group's ``sets.toml``: no fixture set is read."""
    source_of = set_sources(corpus)
    rows: list[dict] = []
    for manifest in load_manifests(fixtures):
        tally: Counter[tuple[str, str]] = Counter()
        for (kind, name), table in set_tables.read(fixtures / manifest.group_slug / set_tables.FILE).items():
            tally[(kind, source_of.get((kind, name), HAND_WRITTEN))] += table["cases"]
        row = {"model": manifest.model, "group": manifest.group_slug, "tier": manifest.tier}
        rows += [{**row, "kind": kind, "source": source, "cases": n} for (kind, source), n in tally.items()]
        if not tally:
            rows.append({**row, "kind": None, "source": None, "cases": 0})
    return sorted(rows, key=lambda r: (r["model"], r["kind"] or "", r["source"] or ""))


def render_markdown(rows: list[dict]) -> str:
    columns = ("model", "group", "tier", "kind", "source", "cases")
    lines = ["| Model | Group | Tier | Kind | Source | Cases |", "|---|---|---:|---|---|---:|"]
    lines += [_line(row[column] for column in columns) for row in rows]
    each_group_once = {(r["group"], r["kind"], r["source"]): r["cases"] for r in rows}
    lines.append(_line(["all", None, None, None, None, sum(each_group_once.values())]))
    return "\n".join(lines) + "\n"


def _line(cells: Iterable[object]) -> str:
    return "|" + "|".join(" " if cell is None else f" {cell} " for cell in cells) + "|"


def run(args: argparse.Namespace) -> int:
    rows = counts(args.fixtures, args.corpus)
    sys.stdout.write(json.dumps(rows, indent=1) + "\n" if args.format == "json" else render_markdown(rows))
    return 0
