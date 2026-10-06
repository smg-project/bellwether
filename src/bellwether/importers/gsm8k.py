"""GSM8K's grade school math problems as corpus sets: per split, a render set and two parse sets.

The data is OpenAI's grade-school-math repository at a pinned commit: ``train.jsonl`` and ``test.jsonl``, one problem
per line, ``{"question", "answer"}``. The answer is the worked solution, whose lines carry calculator annotations such
as ``<<16-3-4=9>>``, then a last line ``#### <final answer>``. Each file is fetched by its path at the commit and
checked against its sha256. This module, like the fetcher and the set writer it uses, imports nothing beyond the
standard library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bellwether import jsonl

from . import corpus_sets, github

OWNER = "openai"
REPO = "grade-school-math"
COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
SOURCE = f"github:{OWNER}/{REPO}@{COMMIT}"
LICENSE = "MIT"
LICENSE_FILE = "LICENSE"
LICENSE_SHA256 = "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b"
# Where the import writes the pinned LICENSE, under the corpus root: MIT asks that its notice go with every copy.
LICENSE_COPY = "licenses/gsm8k-LICENSE"
DATA = "grade_school_math/data"
# Each split's file, by the sha256 of its bytes, in the order the sets are built.
SHA256 = {
    "train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
}
FINAL = "#### "
# The parse sets per split, each named by its message's parts in order, in the words every importer uses for them:
# reasoning, content and calls.
SHAPES = ("reasoning-content", "content")


class Unusable(ValueError):
    """A GSM8K row that cannot become a case."""


def check_license(text: bytes) -> None:
    """Refuse a LICENSE file that is not the MIT License.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. The opening line is checked as well,
    so that moving the pins to another commit whose license is no longer MIT cannot pass on an updated hash alone.
    """
    if not text.decode("utf-8").startswith("MIT License"):
        raise ValueError(f"{LICENSE_FILE} at {COMMIT}: not the MIT License; review it before importing")


def read_rows(data: bytes, where: str) -> list[tuple[int, dict]]:
    """Each row of a split's file with its 0-based line index, in file order; ``where`` names the file in an error."""
    return [(number - 1, row) for number, row in jsonl.loads(data.decode("utf-8"), where)]


def split_answer(answer: str) -> tuple[str, str]:
    """The worked solution and the final answer: the text before the answer's last line, ``#### <final answer>``.

    Both are kept as written: the solution with its calculator annotations, without the newline that ends it, and the
    final answer as the rest of the last line. A final answer that starts or ends with whitespace is ``Unusable``:
    stripped, it would change the reasoning-content set's content and not the content set's, and no line would say so.
    """
    solution, _, last = answer.rpartition("\n")
    if not last.startswith(FINAL):
        raise Unusable("the answer's last line is not '#### <final answer>'")
    final = last[len(FINAL) :]
    if final != final.strip():
        raise Unusable("the final answer after '#### ' starts or ends with whitespace")
    return solution, final


def data_file(split: str) -> str:
    return f"{DATA}/{split}.jsonl"


def set_name(split: str, shape: str = "") -> str:
    return f"gsm8k-{split}-{shape}" if shape else f"gsm8k-{split}"


def messages_for(answer: str) -> dict[str, dict]:
    """The assistant message of each parse set, in the shapes the hand-written parse cases use.

    ``reasoning-content`` is the output of a model that thinks: the worked solution as ``reasoning_content``, then the
    final answer as ``content``. ``content`` is the output of one that does not: the whole answer, ``#### `` line
    included, as ``content``, with no ``reasoning_content`` key.
    """
    solution, final = split_answer(answer)
    return {"reasoning-content": {"reasoning_content": solution, "content": final}, "content": {"content": answer}}


def origin(split: str, row: int) -> dict:
    """Where a case came from, in the BFCL importer's key order: the split's file and the row's 0-based line index."""
    return {
        "dataset": "gsm8k",
        "source": SOURCE,
        "sha256": SHA256[split],
        "file": data_file(split),
        "row": row,
        "license": LICENSE,
    }


def lines_for(split: str, row: int, problem: dict) -> dict[tuple[str, str], dict]:
    """The row's line in each set of its split, by ``(kind, set name)``; ``Unusable`` when it cannot become a case."""
    if not problem["question"].strip():
        raise Unusable("the question is empty")
    messages = messages_for(problem["answer"])
    request = {"messages": [{"role": "user", "content": problem["question"]}]}
    tail = {"notes": f"GSM8K {split} row {row}", "origin": origin(split, row)}
    lines = {("render", set_name(split)): {"name": f"{set_name(split)}-{row}", "request": request, **tail}}
    for shape, message in messages.items():
        name = set_name(split, shape)
        lines[("parse", name)] = {"name": f"{name}-{row}", "request": request, "message": message, **tail}
    return lines


def build_sets(
    files: dict[str, bytes], skipped: list[tuple[str, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)`` from each split's file: a render set and a parse set per message shape.

    Each set has a line per row, in file order. A row that cannot become a case is left out of every set of its split,
    so that the three hold the same rows, and is appended to ``skipped`` with its reason.
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    for split, data in files.items():
        sets[("render", set_name(split))] = []
        for shape in SHAPES:
            sets[("parse", set_name(split, shape))] = []
        for row, problem in read_rows(data, data_file(split)):
            try:
                lines = lines_for(split, row, problem)
            except Unusable as err:
                if skipped is not None:
                    skipped.append((f"{split} row {row}", str(err)))
                continue
            for key, line in lines.items():
                sets[key].append(line)
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, license_text: bytes) -> list[Path]:
    """Write every set and the pinned LICENSE, and remove ``gsm8k-*`` set files no split writes any more."""
    return corpus_sets.write(sets, corpus_dir, "gsm8k-", {LICENSE_COPY: license_text})


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, license_text: bytes) -> list[str]:
    """One line per set file, or the LICENSE copy, that differs from a fresh import; empty when none does."""
    return corpus_sets.check(sets, corpus_dir, "gsm8k-", "GSM8K split", {LICENSE_COPY: license_text})


def fetch(path: str, sha256: str, cache: Path) -> bytes:
    """The bytes of ``path`` in the repository at the pinned commit, checked against ``sha256``."""
    return github.fetch(OWNER, REPO, COMMIT, path, sha256, cache=cache).read_bytes()


def run(args: argparse.Namespace) -> int:
    license_text = fetch(LICENSE_FILE, LICENSE_SHA256, args.cache)
    check_license(license_text)
    files = {split: fetch(data_file(split), sha256, args.cache) for split, sha256 in SHA256.items()}
    skipped: list[tuple[str, str]] = []
    sets = build_sets(files, skipped)
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    if args.check:
        problems = check_sets(kept, args.corpus, license_text)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the GSM8K sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    corpus_sets.report("GSM8K", sets, kept, repeats, args.corpus)
    corpus_sets.report_skipped(skipped)
    write_sets(kept, args.corpus, license_text)
    return 0
