"""The audio sets an importer writes and ``--check`` compares: ``corpus/audio/<dataset>.jsonl`` and the clips under
``corpus/audios/<dataset>/``, in the shape of the image sets (``image_sets``), with the audio catalogue in place of the
image catalogue.

A line is written in ``audio.FIELDS`` order: ``name`` (``<dataset>-<nnnnn>-<sha8>``), ``path``, ``bytes``, ``sha256``,
the catalogue the container's headers give (``container``, ``codec``, ``sample_rate``, ``channels``, ``bits``,
``duration_s``), ``notes`` and ``origin``. A clip whose bytes repeat an earlier clip's gives no line and is named with
the clip it repeats; bytes the catalogue does not read as audio give no line and are named with the reason. The
file's extension follows the container, ``.opus`` for Opus in Ogg. Writing, checking and reporting are the image
sets' (``storage.write_whole`` per file, the set file last, pointer-or-content identity on check).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from bellwether import audio, storage

from . import image_sets
from .image_sets import Found, Summary, identity, read_set

KIND = audio.KIND
AUDIOS = audio.AUDIOS
# The file extension by container, and by codec where the container's name is not the convention.
EXTENSIONS = {
    "flac": "flac",
    "wav": "wav",
    "ogg": "ogg",
    "mp3": "mp3",
    "mpeg": "mpa",
    "m4a": "m4a",
    "mp4": "mp4",
    "mov": "mov",
}
CODEC_EXTENSIONS = {("ogg", "opus"): "opus"}


def set_path(corpus_dir: Path, dataset: str) -> Path:
    return corpus_dir / KIND / f"{dataset}.jsonl"


def audios_dir(corpus_dir: Path, dataset: str) -> Path:
    return corpus_dir / AUDIOS / dataset


def parquet_clips(path: Path, columns: Iterable[str]) -> Iterator[tuple[int, str, bytes]]:
    """``(row, column, bytes)`` for each cell of ``columns`` that holds bytes, in row then column order: the datasets
    library stores audio as it stores images, a struct of the bytes and a path, which ``image_sets.parquet_images``
    reads as data."""
    return image_sets.parquet_images(path, columns)


def _line(dataset: str, index: int, found: Found) -> tuple[dict, str] | None:
    """A found clip's line and the reason it gets none, in ``audio.FIELDS`` order."""
    try:
        what = audio.catalogue(found.data)
    except audio.NotAudio as err:
        return None, str(err)
    digest = hashlib.sha256(found.data).hexdigest()
    name = f"{dataset}-{index:05d}-{digest[:8]}"
    extension = CODEC_EXTENSIONS.get((what.container, what.codec), EXTENSIONS[what.container])
    line = {
        "name": name,
        "path": f"{AUDIOS}/{dataset}/{name}.{extension}",
        "bytes": len(found.data),
        "sha256": digest,
        **asdict(what),
        "notes": found.notes,
        "origin": found.origin,
    }
    return line, ""


def _json(line: dict) -> str:
    return json.dumps(line, ensure_ascii=False)


def _lines(dataset: str, found: Iterable[Found], summary: Summary) -> Iterator[tuple[dict, bytes]]:
    """Each kept clip's line with its bytes, in order; repeats and unreadable rows go into ``summary``."""
    seen: dict[str, str] = {}
    index = 0
    for clip in found:
        where = clip.notes
        digest = hashlib.sha256(clip.data).hexdigest()
        if digest in seen:
            summary.repeats.append((where, seen[digest]))
            continue
        line, why = _line(dataset, index + 1, clip)
        if line is None:
            summary.skipped.append((where, why))
            continue
        seen[digest] = line["name"]
        index += 1
        summary.lines += 1
        summary.bytes += line["bytes"]
        yield line, clip.data


def write(
    dataset: str, found: Iterable[Found], corpus_dir: Path, files: dict[str, bytes] | None = None
) -> tuple[list[Path], Summary]:
    """Write the dataset's clips and its set, then ``files`` (its licence copy), and remove what the import no longer
    writes under its audios directory; the image sets' ``write``, line for line."""
    summary = Summary()
    directory = audios_dir(corpus_dir, dataset)
    written: list[Path] = []
    lines: list[str] = []
    for line, data in _lines(dataset, found, summary):
        path = corpus_dir / line["path"]
        if not image_sets._holds(path, data):
            storage.write_whole(path, data)
        written.append(path)
        lines.append(_json(line) + "\n")
    target = set_path(corpus_dir, dataset)
    storage.write_whole(target, "".join(lines).encode("utf-8"))
    written.append(target)
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    kept = set(written)
    if directory.is_dir():
        for stale in sorted(p for p in directory.iterdir() if p.is_file() and p not in kept):
            stale.unlink()
    return written, summary


def check(
    dataset: str, found: Iterable[Found], corpus_dir: Path, files: dict[str, bytes] | None = None
) -> tuple[list[str], Summary]:
    """One line per difference between a fresh import and the committed set, its files and ``files``; empty when
    there is none; the image sets' ``check``, line for line."""
    summary = Summary()
    target = set_path(corpus_dir, dataset)
    problems: list[str] = []
    committed: list[str] = []
    if target.is_file():
        committed = [piece for piece in target.read_text(encoding="utf-8").split("\n")]
        committed = [f"{piece}\n" for piece in committed[:-1]] if committed and committed[-1] == "" else committed
    else:
        problems.append(f"{target}: missing")
    expected: set[Path] = set()
    number = 0
    for line, _data in _lines(dataset, found, summary):
        number += 1
        text = _json(line) + "\n"
        if target.is_file() and (number > len(committed) or committed[number - 1] != text):
            problems.append(f"{target}:{number}: differs from a fresh import")
        path = corpus_dir / line["path"]
        expected.add(path)
        if not path.is_file():
            problems.append(f"{path}: missing")
        else:
            try:
                digest, size = identity(path)
            except (OSError, ValueError) as err:
                problems.append(str(err))
                continue
            if (digest, size) != (line["sha256"], line["bytes"]):
                problems.append(f"{path}: holds sha256 {digest}, {size} bytes; a fresh import gives {line['sha256']}")
    if target.is_file() and len(committed) > number:
        problems.append(f"{target}: {len(committed) - number} line(s) past the {number} a fresh import writes")
    directory = audios_dir(corpus_dir, dataset)
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.is_file() and p not in expected):
            problems.append(f"{path}: no row of the import writes it")
    for relative, content in sorted((files or {}).items()):
        path = corpus_dir / relative
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != content:
            problems.append(f"{path}: differs from a fresh import")
    return problems, summary


def report(dataset: str, summary: Summary, corpus_dir: Path) -> None:
    """Print what an import left out and what it kept, in the words of the image imports."""
    for where, first in summary.repeats:
        print(f"no line for {where}: it repeats {first}")
    for where, why in summary.skipped:
        print(f"no line for {where}: {why}")
    print(
        f"{set_path(corpus_dir, dataset)}: {summary.lines} clips, {summary.bytes} bytes, "
        f"{len(summary.repeats)} left out as repeats, {len(summary.skipped)} rows with no clip"
    )


__all__ = ["Found", "Summary", "check", "parquet_clips", "read_set", "report", "set_path", "write"]
