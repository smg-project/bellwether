"""bellwether count: how many cases bellwether holds, per model, kind and source.

The source of a case is the dataset its corpus line was imported from (``origin.dataset``), or ``hand-written``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bellwether.manifest import KINDS, load_manifest
from bellwether.record.corpus import read_cases

HAND_WRITTEN = "hand-written"


def sources(corpus: Path) -> dict[tuple[str, str], str]:
    """``(kind, case name)`` -> the dataset the case came from, for every corpus line that carries an origin."""
    found: dict[tuple[str, str], str] = {}
    for kind in KINDS:
        if (corpus / kind).is_dir():
            for path in sorted((corpus / kind).rglob("*.jsonl")):
                for case in read_cases(path):
                    if case.origin:
                        found[(kind, case.name)] = case.origin["dataset"]
    return found


def counts(fixtures: Path, corpus: Path) -> list[dict]:
    origin_of = sources(corpus)
    tally: Counter[tuple[str, str, str]] = Counter()
    for manifest_path in sorted(fixtures.glob("*/manifest.toml")):
        model = load_manifest(manifest_path).model
        for kind in KINDS:
            for path in sorted((manifest_path.parent / kind).glob("*.jsonl")):
                for raw in path.read_text(encoding="utf-8").splitlines():
                    if raw.strip():
                        name = json.loads(raw)["id"].rsplit("/", 1)[1]
                        tally[(model, kind, origin_of.get((kind, name), HAND_WRITTEN))] += 1
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
