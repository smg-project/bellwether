"""Corpus sets as every importer writes them, and the check that the corpus is what a fresh import writes.

A set file holds one JSON line per case, in the order the importer built them, each ending in "\\n". ``json.dumps``
keeps each line's keys in the order the importer built them and writes non-ASCII text raw (``ensure_ascii=False``),
so a fresh import of the same pinned data is byte-identical to the last one and ``check`` can compare bytes.
``report`` and ``report_skipped`` print what an import keeps and leaves out, in the same words for every importer.
This module imports nothing beyond the standard library.
"""

from __future__ import annotations

import json
from pathlib import Path

KINDS = ("render", "parse")


def _json(value) -> str:
    """``value`` as a set file holds it: keys in the order they were built, non-ASCII text raw."""
    return json.dumps(value, ensure_ascii=False)


def text(lines: list[dict]) -> str:
    return "".join(_json(line) + "\n" for line in lines)


def leave_out_repeats(
    sets: dict[tuple[str, str], list[dict]],
) -> tuple[dict[tuple[str, str], list[dict]], list[tuple[str, str]]]:
    """The sets without the cases that repeat an earlier one, and ``(left-out name, name it repeats)`` for each.

    A case repeats an earlier one when it is of the same kind and its ``request`` (render), or its ``request`` and
    ``message`` (parse), are equal to the earlier case's as the line writes them: JSON with the keys in the order the
    importer built them, since a template can see key order. Names, notes and origin do not count. The key is the
    request's JSON and then the message's, or ``""`` for a render case; no JSON text is empty, so a render case and a
    parse case never share a key. Earlier means in the order the importer built its sets: the sets in the dict's order,
    and each set's lines in order. A repeat is left out because it tests nothing its first case does not, and a count
    that included it would overstate the corpus.

    The kept sets have the same keys in the same order, each with its lines in order; a set whose every case repeats
    an earlier one stays, empty. The pairs come in the order of the cases left out, each naming the kept case.
    """
    first: dict[tuple[str, str], str] = {}
    kept: dict[tuple[str, str], list[dict]] = {}
    repeats: list[tuple[str, str]] = []
    for (kind, name), lines in sets.items():
        kept[(kind, name)] = []
        for line in lines:
            compared = (_json(line["request"]), _json(line["message"]) if kind == "parse" else "")
            if compared in first:
                repeats.append((line["name"], first[compared]))
            else:
                first[compared] = line["name"]
                kept[(kind, name)].append(line)
    return kept, repeats


def report(
    dataset: str,
    sets: dict[tuple[str, str], list[dict]],
    kept: dict[tuple[str, str], list[dict]],
    repeats: list[tuple[str, str]],
    corpus_dir: Path,
) -> None:
    """Print what an import leaves out as repeats, and what it keeps.

    First each case ``leave_out_repeats`` left out, with the case it repeats; then each kept set's file, by kind and
    name, with its count of cases and of repeats left out; then the total for ``dataset``'s sets.
    """
    for name, first in repeats:
        print(f"no case {name}: it repeats {first}")
    for (kind, name), lines in sorted(kept.items()):
        left_out = len(sets[(kind, name)]) - len(lines)
        repeated = f", {left_out} left out as repeats" if left_out else ""
        print(f"{corpus_dir / kind / f'{name}.jsonl'}: {len(lines)} cases{repeated}")
    total = sum(len(lines) for lines in kept.values())
    print(f"{corpus_dir}: {total} cases in the {len(kept)} {dataset} sets, {len(repeats)} left out as repeats")


def report_skipped(skipped: list[tuple[str, str]], lost: str = "case") -> None:
    """Print each reason once, with how many rows it left out and every one of them.

    ``lost`` is what such a row does not get: ``case`` when it gets none (GSM8K), ``parse case`` when it keeps its
    render case (BFCL).
    """
    rows_by_reason: dict[str, list[str]] = {}
    for row, why in skipped:
        rows_by_reason.setdefault(why, []).append(row)
    for why, rows in rows_by_reason.items():
        print(f"no {lost} for {len(rows)} row(s) ({', '.join(rows)}): {why}")


def write(
    sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, prefix: str, files: dict[str, bytes] | None = None
) -> list[Path]:
    """Write every set and ``files``, and remove the ``<prefix>*`` set files the import no longer writes.

    ``files`` are the import's other files, such as a dataset's license, as bytes by path under ``corpus_dir``. The
    prefix is the importer's.
    """
    written = []
    for (kind, name), lines in sorted(sets.items()):
        path = corpus_dir / kind / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text(lines).encode("utf-8"))
        written.append(path)
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    for kind in KINDS:
        for stale in sorted((corpus_dir / kind).glob(f"{prefix}*.jsonl")):
            if stale not in written:
                stale.unlink()
    return written


def check(
    sets: dict[tuple[str, str], list[dict]],
    corpus_dir: Path,
    prefix: str,
    unit: str,
    files: dict[str, bytes] | None = None,
) -> list[str]:
    """One line per set file, or file of ``files``, that differs from a fresh import; empty when none does.

    ``files`` are as ``write`` takes them. ``unit`` names what one set comes from (a BFCL category, a GSM8K split) in
    the line for a stale ``<prefix>*`` file.
    """
    expected = {
        corpus_dir / kind / f"{name}.jsonl": text(lines).encode("utf-8") for (kind, name), lines in sets.items()
    }
    expected.update({corpus_dir / relative: content for relative, content in (files or {}).items()})
    problems = []
    for path, content in sorted(expected.items()):
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != content:
            problems.append(f"{path}: differs from a fresh import")
    for kind in KINDS:
        for path in sorted((corpus_dir / kind).glob(f"{prefix}*.jsonl")):
            if path not in expected:
                problems.append(f"{path}: no {unit} writes it")
    return problems
