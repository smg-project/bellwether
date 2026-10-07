"""Corpus sets as every importer writes them, and the check that the corpus is what a fresh import writes.

A set file holds one JSON line per case, in the order the importer built them, each ending in "\\n". ``json.dumps``
keeps each line's keys in the order the importer built them and writes non-ASCII text raw (``ensure_ascii=False``),
so a fresh import of the same pinned data is byte-identical to the last one, and ``check`` compares plain content byte
for byte.

An import's sets are stored in one form: plain JSON Lines while they take at most ``LIMIT`` bytes in all, and past it
every one of them compressed with zstd and kept in Git LFS (``<name>.jsonl.zst``), the fixtures' form
(``bellwether.storage``). ``report`` and ``report_skipped`` print what an import keeps and leaves out, in the same words
for every importer. This module imports nothing beyond the standard library until a compressed set is read or written.

``write`` and ``check`` hold one set at a time, and no set's content whole: a set goes to ``storage`` as its lines,
encoded one at a time (``_Encoded``), and is compressed, written and compared as they come. They take an import's sets
in either of two shapes:

- a mapping ``{(kind, name): lines}``, every set in hand, as an importer that holds its sets passes them after leaving
  out repeats (``leave_out_repeats``) and printing its ``report``. They are written and checked as given, and their
  form is the one ``LIMIT`` gives their total, counted first.
- an iterable of ``(kind, name, lines)``, for an import too large to hold, whose importer builds each set as it reads
  its source. The repeat rule applies as the sets come, keeping a sha256 of each case, and a ``Summary`` collects what
  ``report_summary`` prints afterwards. The total is known only at the end: an importer that knows its form declares it
  (``form``), and ``write`` refuses a declaration the total contradicts. Without one, the sets are written plain, and
  compressed afterwards, one at a time, if their total passes ``LIMIT``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from bellwether import storage

KINDS = ("render", "parse")
# Bytes of plain JSON Lines an import's sets may take in all and stay plain, so that a change to an importer reads as a
# diff of cases; past it they are stored compressed, as docs/benchmark-sets.md (Storage) decided.
LIMIT = 50_000_000
# The forms an import may declare for its sets: plain JSON Lines, or compressed with zstd. An import that declares none
# has the form LIMIT gives its total.
FORMS = ("plain", "zstd")

Sets = Mapping[tuple[str, str], list[dict]] | Iterable[tuple[str, str, Iterable[dict]]]


def _json(value) -> str:
    """``value`` as a set file holds it: keys in the order they were built, non-ASCII text raw."""
    return json.dumps(value, ensure_ascii=False)


def text(lines: list[dict]) -> str:
    return "".join(_json(line) + "\n" for line in lines)


def _line_bytes(line: dict) -> bytes:
    return (_json(line) + "\n").encode("utf-8")


class _Encoded:
    """A set's plain content, ``text`` encoded, as ``storage.Pieces``: its lines encoded one at a time, again each time
    it is read, and its size in bytes, counted once. No copy of the content is made whole, nor a str of it, which would
    take four bytes a character once one character is past U+FFFF."""

    def __init__(self, lines: list[dict]) -> None:
        self._lines = lines
        self._size: int | None = None

    def __iter__(self) -> Iterator[bytes]:
        return (_line_bytes(line) for line in self._lines)

    def __len__(self) -> int:
        if self._size is None:
            self._size = sum(len(piece) for piece in self)
        return self._size


class _FilePieces:
    """A plain set file's bytes as ``storage.Pieces``, read a piece at a time: what a stream's sets, written plain, are
    compressed from once their total passes ``LIMIT``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def __iter__(self) -> Iterator[bytes]:
        with self._path.open("rb") as handle:
            while piece := handle.read(storage.READ_SIZE):
                yield piece

    def __len__(self) -> int:
        return self._path.stat().st_size


def _plain_size(lines: Iterable[dict]) -> int:
    return sum(len(_line_bytes(line)) for line in lines)


def _form_of(size: int) -> str:
    """The form ``LIMIT`` gives an import whose sets take ``size`` bytes in all as plain JSON Lines."""
    return "zstd" if size > LIMIT else "plain"


def _path(corpus_dir: Path, kind: str, name: str, form: str) -> Path:
    return corpus_dir / kind / f"{name}{storage.COMPRESSED_SUFFIX if form == 'zstd' else '.jsonl'}"


def _other_form(path: Path) -> Path:
    """The set file of the same name in the other form."""
    suffix = ".jsonl" if storage.is_compressed(path) else storage.COMPRESSED_SUFFIX
    return path.with_name(f"{storage.stem(path)}{suffix}")


def _prefixed(corpus_dir: Path, prefix: str) -> list[Path]:
    """The ``<prefix>*`` set files in the kinds' directories, in either form: the files an importer's prefix claims."""
    patterns = (f"{prefix}*.jsonl", f"{prefix}*{storage.COMPRESSED_SUFFIX}")
    return sorted(path for kind in KINDS for pattern in patterns for path in (corpus_dir / kind).glob(pattern))


def _declared(form: str | None) -> None:
    if form is not None and form not in FORMS:
        raise ValueError(f"form {form!r} is not 'plain' or 'zstd'")


def _contradiction(corpus_dir: Path, prefix: str, size: int, form: str, where: str = "") -> str:
    """Why ``form``, declared for the ``<prefix>*`` sets, is not the one ``LIMIT`` gives their ``size`` bytes."""
    limit = f"{'past' if size > LIMIT else 'within'} the {LIMIT} that stay plain"
    other = "zstd" if form == "plain" else "plain"
    return (
        f"{corpus_dir}: the {prefix}* sets take {size} bytes as plain JSON Lines{where}, {limit}, "
        f"but the import declares form {form!r}; declare form {other!r}"
    )


def _repeat_key(kind: str, line: dict) -> bytes:
    """What the repeat rule compares, as a sha256: the request's JSON, a newline, and for a parse case the message's.
    No JSON text ``_json`` writes holds a newline or is empty, so the parts cannot run into each other and a render
    case's key is never a parse case's."""
    message = _json(line["message"]) if kind == "parse" else ""
    return hashlib.sha256(f"{_json(line['request'])}\n{message}".encode()).digest()


def _message_key(line: dict) -> bytes:
    return hashlib.sha256(_json(line["message"]).encode()).digest()


def leave_out_repeats(
    sets: dict[tuple[str, str], list[dict]],
) -> tuple[dict[tuple[str, str], list[dict]], list[tuple[str, str]]]:
    """The sets without the cases that repeat an earlier one, and ``(left-out name, name it repeats)`` for each.

    A case repeats an earlier one when it is of the same kind and its ``request`` (render), or its ``request`` and
    ``message`` (parse), are equal to the earlier case's as the line writes them: JSON with the keys in the order the
    importer built them, since a template can see key order. Names, notes and origin do not count. The rule keeps a
    sha256 of each case (``_repeat_key``), not its JSON. Earlier means in the order the importer built its sets: the
    sets in the dict's order, and each set's lines in order. A repeat is left out because it tests nothing its first
    case does not, and a count that included it would overstate the corpus.

    The kept sets have the same keys in the same order, each with its lines in order; a set whose every case repeats
    an earlier one stays, empty. The pairs come in the order of the cases left out, each naming the kept case.
    """
    first: dict[bytes, str] = {}
    kept: dict[tuple[str, str], list[dict]] = {}
    repeats: list[tuple[str, str]] = []
    for (kind, name), lines in sets.items():
        kept[(kind, name)] = []
        for line in lines:
            key = _repeat_key(kind, line)
            if key in first:
                repeats.append((line["name"], first[key]))
            else:
                first[key] = line["name"]
                kept[(kind, name)].append(line)
    return kept, repeats


@dataclass(frozen=True)
class SetCount:
    """One set as the report counts it: its cases kept, its cases left out as repeats, its distinct messages."""

    kind: str
    name: str
    cases: int
    left_out: int
    messages: int


@dataclass
class Summary:
    """What ``write`` or ``check`` kept of an import's sets and left out, set by set, for ``report_summary``: the
    counts of each set, each case left out with the case it repeats, a sha256 of each distinct message, and the sets'
    plain bytes in all and the form they are stored in."""

    sets: list[SetCount] = field(default_factory=list)
    repeats: list[tuple[str, str]] = field(default_factory=list)
    messages: set[bytes] = field(default_factory=set)
    size: int = 0
    form: str = "plain"

    def count(self, kind: str, name: str, lines: list[dict], left_out: int) -> None:
        messages = {_message_key(line) for line in lines} if kind == "parse" else set()
        self.messages |= messages
        self.sets.append(SetCount(kind, name, len(lines), left_out, len(messages)))


def _report(dataset: str, summary: Summary, corpus_dir: Path) -> None:
    for name, first in summary.repeats:
        print(f"no case {name}: it repeats {first}")
    for count in sorted(summary.sets, key=lambda count: (count.kind, count.name)):
        counts = [f"{count.cases} cases"]
        if count.left_out:
            counts.append(f"{count.left_out} left out as repeats")
        if count.kind == "parse":
            counts.append(f"{count.messages} distinct messages")
        print(f"{_path(corpus_dir, count.kind, count.name, summary.form)}: {', '.join(counts)}")
    total = sum(count.cases for count in summary.sets)
    print(
        f"{corpus_dir}: {total} cases in the {len(summary.sets)} {dataset} sets, {len(summary.repeats)} left out as "
        f"repeats, {len(summary.messages)} distinct messages"
    )


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
    each message once across the sets: no count overstates what the sets hold. ``report_summary`` prints the same for
    an import that streams its sets.
    """
    summary = Summary(repeats=list(repeats), size=sum(_plain_size(lines) for lines in kept.values()))
    summary.form = _form_of(summary.size)
    for (kind, name), lines in kept.items():
        summary.count(kind, name, lines, len(sets[(kind, name)]) - len(lines))
    _report(dataset, summary, corpus_dir)


def report_summary(dataset: str, summary: Summary, corpus_dir: Path) -> None:
    """Print what ``report`` prints, from the ``Summary`` that ``write`` filled as it streamed an import's sets."""
    _report(dataset, summary, corpus_dir)


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


def _sets(sets: Sets, summary: Summary) -> Iterator[tuple[str, str, list[dict]]]:
    """Each set, ``(kind, name, lines)``, one at a time, counted into ``summary``.

    A mapping's sets come as given, in the order of their kinds and names. An iterable's come in its order, each
    without the cases that repeat an earlier one, by ``leave_out_repeats``'s rule, and each case left out goes into
    ``summary.repeats``. A set name an iterable gives twice is refused: its second set would overwrite the first. A
    streamed set is let go of before the next is asked for, so that the importer builds the next with only one held.
    """
    if isinstance(sets, Mapping):
        for (kind, name), lines in sorted(sets.items()):
            summary.count(kind, name, lines, 0)
            yield kind, name, lines
        return
    first: dict[bytes, str] = {}
    seen: set[tuple[str, str]] = set()
    for kind, name, lines in sets:
        if (kind, name) in seen:
            raise ValueError(f"the import gives the set {kind}/{name} twice")
        seen.add((kind, name))
        kept: list[dict] = []
        left_out = 0
        for line in lines:
            key = _repeat_key(kind, line)
            if key in first:
                summary.repeats.append((line["name"], first[key]))
                left_out += 1
            else:
                first[key] = line["name"]
                kept.append(line)
        summary.count(kind, name, kept, left_out)
        yield kind, name, kept
        del lines, kept


def write(
    sets: Sets,
    corpus_dir: Path,
    prefix: str,
    files: dict[str, bytes] | None = None,
    *,
    form: str | None = None,
    summary: Summary | None = None,
) -> list[Path]:
    """Write every set and ``files``, then remove the ``<prefix>*`` set files the import no longer writes.

    The sets are written one at a time (``_sets``), in the import's form, so a set stored in the other form is removed
    too: an import keeps one form. A mapping's form is the one ``LIMIT`` gives its total, or a declared ``form`` the
    total agrees with. An iterable's is its declared ``form``: one declared plain is refused before the set that would
    take it past ``LIMIT`` is written, and one declared compressed that ends within ``LIMIT`` is refused after its sets
    are written, before anything is removed. Without a declaration an iterable's sets are written plain, and compressed
    afterwards, one at a time from their plain files, if their total passes ``LIMIT``.

    Each file is written whole or not at all (``storage.write``), and nothing is removed until everything is written,
    so a write cut short or refused removes no file: a set it had not reached keeps its old file, and a set it had
    reached holds the new content. ``files`` are the import's other files, such as a dataset's
    license, as bytes by path under ``corpus_dir``; they are written as they are and do not count toward ``LIMIT``. The
    prefix is the importer's. ``summary``, when given, collects the counts ``report_summary`` prints.
    """
    _declared(form)
    summary = Summary() if summary is None else summary
    known = None
    if isinstance(sets, Mapping):
        size = sum(_plain_size(lines) for lines in sets.values())
        known = _form_of(size)
        if form is not None and form != known:
            raise ValueError(_contradiction(corpus_dir, prefix, size, form))
    writing = form or known or "plain"
    written: dict[tuple[str, str], Path] = {}
    for kind, name, lines in _sets(sets, summary):
        content = _Encoded(lines)
        summary.size += len(content)
        if form == "plain" and summary.size > LIMIT:
            raise ValueError(_contradiction(corpus_dir, prefix, summary.size, form, f" by {kind}/{name}"))
        path = _path(corpus_dir, kind, name, writing)
        storage.write(path, content)
        written[(kind, name)] = path
        del lines, content  # let the set go before the next is built
    final = _form_of(summary.size)
    if form is not None and form != final:
        raise ValueError(_contradiction(corpus_dir, prefix, summary.size, form))
    if final != writing:
        for key, path in written.items():
            compressed = _other_form(path)
            storage.write(compressed, _FilePieces(path))
            written[key] = compressed
    summary.form = final
    paths = [written[key] for key in sorted(written)]
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths.append(path)
    for stale in _prefixed(corpus_dir, prefix):
        if stale not in paths:
            stale.unlink()
    return paths


def _compare(paths: dict[str, Path], lines: list[dict]) -> tuple[int, dict[str, str | None]]:
    """A set's plain size, and for each of ``paths`` (by form) the line naming how its file differs from the set's
    lines, or None when it holds them; the file is compared as it is read, with the lines as they are encoded
    (``storage.holds``), so no content of the set is held whole.

    A Git LFS pointer, or a frame zstd cannot decompress, is named with ``storage.plain_bytes``'s reason.
    """
    content = _Encoded(lines)
    problems: dict[str, str | None] = {}
    for form, path in paths.items():
        try:
            problems[form] = None if storage.holds(path, content) else f"{path}: differs from a fresh import"
        except ValueError as err:  # a Git LFS pointer, or a frame zstd cannot decompress: nothing to compare
            problems[form] = str(err)
    return len(content), problems


def check(
    sets: Sets,
    corpus_dir: Path,
    prefix: str,
    unit: str,
    files: dict[str, bytes] | None = None,
    *,
    form: str | None = None,
    summary: Summary | None = None,
) -> list[str]:
    """One line per set file, or file of ``files``, that differs from a fresh import; empty when none does.

    The sets are compared one at a time (``_sets``), in the form a fresh import writes them, by plain content, so a
    compressed set passes whatever bytes its compressor wrote; a set stored in the other form is named with the form
    ``LIMIT`` gives it. An iterable without a declared ``form`` is compared with whichever form each set is stored in,
    and named against the form its total gives once every set has come. A set Git LFS has not fetched is named with
    the command that fetches it. A declared ``form`` the total contradicts is named last. ``files`` are as ``write``
    takes them. ``unit`` names what one set comes from (a BFCL category, a GSM8K split) in the line for a stale
    ``<prefix>*`` file. ``summary``, when given, collects the counts ``report_summary`` prints.
    """
    _declared(form)
    summary = Summary() if summary is None else summary
    known = None
    if isinstance(sets, Mapping):
        known = _form_of(sum(_plain_size(lines) for lines in sets.values()))
    compared = [form or known] if (form or known) else list(FORMS)
    found: dict[tuple[str, str], dict[str, str | None]] = {}
    for kind, name, lines in _sets(sets, summary):
        paths = {each: _path(corpus_dir, kind, name, each) for each in compared}
        size, problems = _compare({each: path for each, path in paths.items() if path.is_file()}, lines)
        summary.size += size
        found[(kind, name)] = problems
        del lines  # let the set go before the next is built
    expected_form = form or known or _form_of(summary.size)
    summary.form = expected_form
    expected: dict[Path, str | None] = {}
    for (kind, name), problems in found.items():
        path = _path(corpus_dir, kind, name, expected_form)
        expected[path] = problems[expected_form] if expected_form in problems else f"{path}: missing"
    for relative, content in (files or {}).items():
        path = corpus_dir / relative
        if not path.is_file():
            expected[path] = f"{path}: missing"
            continue
        try:
            stored = storage.plain_bytes(path)
        except ValueError as err:  # a Git LFS pointer: the content is not here to compare
            expected[path] = str(err)
            continue
        expected[path] = None if stored == content else f"{path}: differs from a fresh import"
    lines_out = [problem for _, problem in sorted(expected.items()) if problem]
    other_form = {_other_form(path): path for path in expected if storage.stem(path) is not None}
    for path in _prefixed(corpus_dir, prefix):
        if path in expected:
            continue
        if path in other_form:
            written = other_form[path]
            limit = f"{'past' if storage.is_compressed(written) else 'within'} the {LIMIT} that stay plain"
            reason = f"the import's sets take {summary.size} bytes as plain JSON Lines, {limit}"
            lines_out.append(f"{path}: a fresh import writes this set as {written.name}: {reason}")
        else:
            lines_out.append(f"{path}: no {unit} writes it")
    if form is not None and form != _form_of(summary.size):
        lines_out.append(_contradiction(corpus_dir, prefix, summary.size, form))
    return lines_out
