"""Corpus sets as every importer writes them, and the check that the corpus is what a fresh import writes.

A set file holds one JSON line per case, in the order the importer built them, each ending in "\\n". ``json.dumps``
keeps each line's keys in the order the importer built them and writes non-ASCII text raw (``ensure_ascii=False``),
so a fresh import of the same pinned data is byte-identical to the last one and ``check`` can compare bytes. This
module imports nothing beyond the standard library.
"""

from __future__ import annotations

import json
from pathlib import Path

KINDS = ("render", "parse")


def text(lines: list[dict]) -> str:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)


def write(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, prefix: str) -> list[Path]:
    """Write every set, and remove the ``<prefix>*`` files the import no longer writes: the prefix is the importer's."""
    written = []
    for (kind, name), lines in sorted(sets.items()):
        path = corpus_dir / kind / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text(lines).encode("utf-8"))
        written.append(path)
    for kind in KINDS:
        for stale in sorted((corpus_dir / kind).glob(f"{prefix}*.jsonl")):
            if stale not in written:
                stale.unlink()
    return written


def check(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, prefix: str, unit: str) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes.

    ``unit`` names what one set comes from (a BFCL category, a GSM8K split) in the line for a stale ``<prefix>*`` file.
    """
    expected = {corpus_dir / kind / f"{name}.jsonl": text(lines) for (kind, name), lines in sets.items()}
    problems = []
    for path, content in sorted(expected.items()):
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != content.encode("utf-8"):
            problems.append(f"{path}: differs from a fresh import")
    for kind in KINDS:
        for path in sorted((corpus_dir / kind).glob(f"{prefix}*.jsonl")):
            if path not in expected:
                problems.append(f"{path}: no {unit} writes it")
    return problems
