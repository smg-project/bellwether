"""Corpus sets as every importer writes them, and the check that the corpus is what a fresh import writes.

A set file holds one JSON line per case, in the order the importer built them, each ending in "\\n". ``json.dumps``
keeps each line's keys in the order the importer built them and writes non-ASCII text raw (``ensure_ascii=False``),
so a fresh import of the same pinned data is byte-identical to the last one and ``check`` can compare bytes.

An import's sets are stored in one form: plain JSON Lines while they take at most ``LIMIT`` bytes in all, and past it
every one of them compressed with zstd and kept in Git LFS (``<name>.jsonl.zst``), the fixtures' form
(``bellwether.storage``). ``report`` and ``report_skipped`` print what an import keeps and leaves out, in the same words
for every importer. This module imports nothing beyond the standard library until a compressed set is read or written.
"""

from __future__ import annotations

import json
from pathlib import Path

from bellwether import storage

KINDS = ("render", "parse")
# Bytes of plain JSON Lines an import's sets may take in all and stay plain, so that a change to an importer reads as a
# diff of cases; past it they are stored compressed, as docs/benchmark-sets.md (Storage) decided.
LIMIT = 50_000_000


def _json(value) -> str:
    """``value`` as a set file holds it: keys in the order they were built, non-ASCII text raw."""
    return json.dumps(value, ensure_ascii=False)


def text(lines: list[dict]) -> str:
    return "".join(_json(line) + "\n" for line in lines)


def set_files(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> dict[tuple[str, str], tuple[Path, bytes]]:
    """Each set's file under ``corpus_dir`` and its plain content, by kind and name, in the form the import's sets take.

    The form is the import's, not each set's: ``<kind>/<name>.jsonl`` while the sets take at most ``LIMIT`` bytes in all
    as plain JSON Lines, and ``<kind>/<name>.jsonl.zst`` for every one of them past it.
    """
    contents = {key: text(lines).encode("utf-8") for key, lines in sets.items()}
    suffix = storage.COMPRESSED_SUFFIX if sum(len(content) for content in contents.values()) > LIMIT else ".jsonl"
    return {(kind, name): (corpus_dir / kind / f"{name}{suffix}", data) for (kind, name), data in contents.items()}


def prefixed(corpus_dir: Path, prefix: str) -> list[Path]:
    """The ``<prefix>*`` set files in the kinds' directories, in either form: the files an importer's prefix claims."""
    patterns = (f"{prefix}*.jsonl", f"{prefix}*{storage.COMPRESSED_SUFFIX}")
    return sorted(path for kind in KINDS for pattern in patterns for path in (corpus_dir / kind).glob(pattern))


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
    name, with its count of cases, of repeats left out when there are any, and, for a parse set, of distinct messages;
    then the same three numbers for ``dataset``'s sets. Cases that are not repeats can still carry the same message,
    which is all a parser sees, so messages are counted apart, compared as the line writes them, and the total counts
    each message once across the sets: no count overstates what the sets hold.
    """
    for name, first in repeats:
        print(f"no case {name}: it repeats {first}")
    paths = {key: path for key, (path, _) in set_files(kept, corpus_dir).items()}
    all_messages: set[str] = set()
    for (kind, name), lines in sorted(kept.items()):
        counts = [f"{len(lines)} cases"]
        left_out = len(sets[(kind, name)]) - len(lines)
        if left_out:
            counts.append(f"{left_out} left out as repeats")
        if kind == "parse":
            messages = {_json(line["message"]) for line in lines}
            all_messages |= messages
            counts.append(f"{len(messages)} distinct messages")
        print(f"{paths[(kind, name)]}: {', '.join(counts)}")
    total = sum(len(lines) for lines in kept.values())
    print(
        f"{corpus_dir}: {total} cases in the {len(kept)} {dataset} sets, {len(repeats)} left out as repeats, "
        f"{len(all_messages)} distinct messages"
    )


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
    """Write every set and ``files``, then remove the ``<prefix>*`` set files the import no longer writes.

    The sets are written in the import's form (``set_files``), so a set stored in the other form is removed too: an
    import keeps one form. Nothing is removed until everything is written, so a write cut short leaves the old files.
    ``files`` are the import's other files, such as a dataset's license, as bytes by path under ``corpus_dir``; they are
    written as they are and do not count toward ``LIMIT``. The prefix is the importer's.
    """
    written = []
    for _, (path, content) in sorted(set_files(sets, corpus_dir).items()):
        storage.write(path, content)
        written.append(path)
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    for stale in prefixed(corpus_dir, prefix):
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

    The sets are compared in the form a fresh import writes them (``set_files``), by plain content, so a compressed set
    passes whatever bytes its compressor wrote; a set stored in the other form is named with the form ``LIMIT`` gives
    it. A set Git LFS has not fetched is named with the command that fetches it. ``files`` are as ``write`` takes them.
    ``unit`` names what one set comes from (a BFCL category, a GSM8K split) in the line for a stale ``<prefix>*`` file.
    """
    fresh = set_files(sets, corpus_dir)
    size = sum(len(content) for _, content in fresh.values())
    expected = dict(fresh.values())
    expected.update({corpus_dir / relative: content for relative, content in (files or {}).items()})
    problems = []
    for path, content in sorted(expected.items()):
        if not path.is_file():
            problems.append(f"{path}: missing")
            continue
        try:
            stored = storage.plain_bytes(path)
        except ValueError as err:  # a Git LFS pointer: the content is not here to compare
            problems.append(str(err))
            continue
        if stored != content:
            problems.append(f"{path}: differs from a fresh import")
    other_form = {
        path.with_name(f"{name}{'.jsonl' if storage.is_compressed(path) else storage.COMPRESSED_SUFFIX}"): path
        for (_, name), (path, _) in fresh.items()
    }
    for path in prefixed(corpus_dir, prefix):
        if path in expected:
            continue
        if path in other_form:
            written = other_form[path]
            limit = f"{'past' if storage.is_compressed(written) else 'within'} the {LIMIT} that stay plain"
            reason = f"the {prefix}* sets take {size} bytes as plain JSON Lines, {limit}"
            problems.append(f"{path}: a fresh import writes this set as {written.name}: {reason}")
        else:
            problems.append(f"{path}: no {unit} writes it")
    return problems
