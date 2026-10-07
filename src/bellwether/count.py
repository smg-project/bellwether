"""bellwether count: how many cases bellwether holds, per checkpoint, kind and source.

The source of a case is the dataset its corpus line was imported from (``origin.dataset``), or ``hand-written``. Every
checkpoint with a manifest is a row, as in the design's per-model table: a group's members show the cases of their
group, which is recorded once, and name it; a checkpoint whose group has recorded nothing yet still has a row, with no
kind and no cases. The ``all`` line counts each group's cases once.

With ``--readme``, it writes two tables into the README instead, between ``README_BEGIN`` and ``README_END``: one row
per checkpoint group that has recorded anything, and one row per corpus source, read from every line of every set.
``--check`` compares them with the file and writes nothing, as CI runs it, so the tables cannot go stale.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from bellwether import jsonl, storage
from bellwether.manifest import KINDS, load_manifests
from bellwether.record import sets as set_tables

HAND_WRITTEN = "hand-written"
README_BEGIN = (
    "<!-- bellwether count --readme writes everything from here to the end marker: edit none of it by hand -->"
)
README_END = "<!-- end of what bellwether count --readme writes -->"
# Past this many licenses a source names them by count, as SWE-bench's lines carry each repository's.
MOST_LICENSES_NAMED = 4


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
                if first is not None and not isinstance(first, dict):
                    raise ValueError(f"{path}: line 1 is not a JSON object")
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


def group_rows(fixtures: Path, corpus: Path) -> list[dict]:
    """One row per checkpoint group, from its primary's manifest and ``sets.toml``: the model at its revision, how many
    checkpoints share the group, its tier, its cases by kind, the cases it refused, and where its sets came from."""
    source_of = set_sources(corpus)
    manifests = load_manifests(fixtures)
    checkpoints = Counter(manifest.group_slug for manifest in manifests)
    rows = []
    for manifest in manifests:
        if manifest.group is not None:
            continue
        cases: Counter[str] = Counter()
        refused = 0
        sources: set[str] = set()
        for (kind, name), table in set_tables.read(fixtures / manifest.slug / set_tables.FILE).items():
            cases[kind] += table["cases"]
            refused += table.get("rejected", 0)
            sources.add(source_of.get((kind, name), HAND_WRITTEN))
        rows.append(
            {
                "group": manifest.slug,
                "model": manifest.model,
                "revision": manifest.revision,
                "checkpoints": checkpoints[manifest.slug],
                "tier": manifest.tier,
                "render": cases["render"],
                "parse": cases["parse"],
                "refused": refused,
                "sources": sorted(sources),
                "sets_toml": fixtures / manifest.slug / set_tables.FILE,
            }
        )
    return sorted(rows, key=lambda row: row["group"])


def corpus_rows(corpus: Path) -> list[dict]:
    """One row per source (``origin.dataset``, or ``hand-written``): where its lines come from, the licenses they name,
    its sets and cases by kind, and the bytes they take plain and as stored. Every line of every set is read, a
    compressed set as it is decompressed."""
    rows: dict[str, dict] = {}
    for kind in KINDS:
        if not (corpus / kind).is_dir():
            continue
        for path in sorted(p for p in (corpus / kind).rglob("*") if p.is_file() and storage.stem(p) is not None):
            cases = plain = 0
            dataset, froms, licenses = HAND_WRITTEN, set(), set()
            with storage.open_text(path) as lines:
                for number, line in enumerate(lines, 1):
                    cases += 1
                    plain += len(line.encode("utf-8"))
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as err:
                        raise ValueError(f"{path}:{number}: not JSON: {err}") from err
                    if not isinstance(record, dict):
                        raise ValueError(f"{path}: line {number} is not a JSON object")
                    origin = record.get("origin")
                    if isinstance(origin, dict):
                        dataset = origin["dataset"]
                        # A line built from others' parts (shapes pairs a BFCL row with a GSM8K one) names theirs.
                        parts = origin.get("parts") or [origin]
                        froms.add(
                            " and ".join(sorted({_short(part.get("source") or part["dataset"]) for part in parts}))
                        )
                        licenses |= {part["license"] for part in parts if part.get("license")}
            row = rows.setdefault(
                dataset,
                {
                    "source": dataset,
                    "from": set(),
                    "licenses": set(),
                    "sets": Counter(),
                    "cases": Counter(),
                    "plain": 0,
                    "stored": 0,
                    "forms": set(),
                },
            )
            row["from"] |= froms or {"written in this repository"}
            row["licenses"] |= licenses
            row["sets"][kind] += 1
            row["cases"][kind] += cases
            row["plain"] += plain
            row["stored"] += path.stat().st_size
            row["forms"].add("zstd in Git LFS" if storage.is_compressed(path) else "plain")
    return [rows[name] for name in sorted(rows)]


def _short(source: str) -> str:
    """A source with its full commit id cut to eight characters, as the models table writes revisions."""
    return re.sub(r"@([0-9a-f]{8})[0-9a-f]{32}\b", r"@\1", source)


def _size(size: int) -> str:
    return f"{size / 1e6:.1f} MB" if size < 1e9 else f"{size / 1e9:.2f} GB"


def render_readme(groups: list[dict], sources: list[dict], base: Path) -> str:
    """The README's block: the two tables, between the markers."""
    recorded = [row for row in groups if row["render"] or row["parse"] or row["refused"]]
    rest = [row for row in groups if row not in recorded]
    lines = [
        README_BEGIN,
        "",
        "### Models",
        "",
        "Each checkpoint group is recorded once, by its primary, for every checkpoint that shares its tokenizer and "
        "template; the counts are each group's `sets.toml`.",
        "",
        "| Group | Model | Checkpoints | Tier | Render cases | Parse cases | Refused | Sources |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    lines += [
        _line(
            [
                f"[{row['group']}]({os.path.relpath(row['sets_toml'], base)})",
                f"{row['model']} @ {row['revision'][:8]}",
                row["checkpoints"],
                row["tier"],
                row["render"],
                row["parse"],
                row["refused"],
                ", ".join(row["sources"]),
            ]
        )
        for row in recorded
    ]
    lines.append(
        _line(
            [
                "all",
                None,
                sum(row["checkpoints"] for row in recorded),
                None,
                sum(row["render"] for row in recorded),
                sum(row["parse"] for row in recorded),
                sum(row["refused"] for row in recorded),
                None,
            ]
        )
    )
    if rest:
        groups_word = "group" if len(rest) == 1 else "groups"
        held = sum(row["checkpoints"] for row in rest)
        checkpoints_word = "checkpoint" if held == 1 else "checkpoints"
        verb = "has" if len(rest) == 1 else "have"
        lines += [
            "",
            f"{len(rest)} more {groups_word} ({held} {checkpoints_word}) {verb} a manifest and nothing recorded yet.",
        ]
    lines += [
        "",
        "### Corpus",
        "",
        "The cases every group is recorded over, by the source they were imported from.",
        "",
        "| Source | From | Licenses | Render sets | Render cases | Parse sets | Parse cases | Plain | Stored |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sources:
        licenses = sorted(row["licenses"])
        named = ", ".join(licenses) if len(licenses) <= MOST_LICENSES_NAMED else f"{len(licenses)} licenses"
        lines.append(
            _line(
                [
                    row["source"],
                    ", ".join(sorted(row["from"])),
                    named or None,
                    row["sets"]["render"],
                    row["cases"]["render"],
                    row["sets"]["parse"],
                    row["cases"]["parse"],
                    _size(row["plain"]),
                    f"{_size(row['stored'])}, {' and '.join(sorted(row['forms']))}",
                ]
            )
        )
    lines.append(
        _line(
            [
                "all",
                None,
                None,
                sum(row["sets"]["render"] for row in sources),
                sum(row["cases"]["render"] for row in sources),
                sum(row["sets"]["parse"] for row in sources),
                sum(row["cases"]["parse"] for row in sources),
                _size(sum(row["plain"] for row in sources)),
                _size(sum(row["stored"] for row in sources)),
            ]
        )
    )
    return "\n".join([*lines, "", README_END])


def write_readme(path: Path, block: str, check: bool) -> int:
    """Put ``block`` between the README's markers, or with ``check`` say whether it is already there."""
    text = path.read_text(encoding="utf-8")
    start, end = text.find(README_BEGIN), text.find(README_END)
    if start < 0 or end < start:
        print(
            f"bellwether count: {path} has no tables to replace: it needs the two markers {README_BEGIN!r} and "
            f"{README_END!r}",
            file=sys.stderr,
        )
        return 1
    fresh = text[:start] + block + text[end + len(README_END) :]
    if check:
        if fresh != text:
            print(
                f"bellwether count: {path}'s tables are not what fixtures/ and corpus/ give; run `bellwether count "
                f"--readme {path}`",
                file=sys.stderr,
            )
            return 1
        return 0
    path.write_text(fresh, encoding="utf-8")
    return 0


def run(args: argparse.Namespace) -> int:
    try:
        if args.readme is not None:
            groups, sources = group_rows(args.fixtures, args.corpus), corpus_rows(args.corpus)
            block = render_readme(groups, sources, args.readme.resolve().parent)
            return write_readme(args.readme, block, args.check)
        rows = counts(args.fixtures, args.corpus)
    except (OSError, ValueError) as err:
        print(f"bellwether count: {err}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(rows, indent=1) + "\n" if args.format == "json" else render_markdown(rows))
    return 0
