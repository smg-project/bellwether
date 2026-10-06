"""SWE-bench Verified and SWE-bench's test split as corpus sets: an issue as the user turn, its gold patch as one call.

SWE-bench has no prompt and no tools, so the request and the messages are bellwether's framing, not SWE-bench's:

- the request: a system turn naming the repository and the base commit, then the problem statement verbatim as the
  user turn, with the hints after ``HINTS`` when the row has any; ``tools`` is ``submit_patch`` alone, on every case;
- a call case: the gold patch as the only argument of one ``submit_patch`` call;
- a content case: the gold patch in a fenced ``diff`` block.

Verified is imported whole. SWE-bench's test split holds all of Verified's rows, so a test row that Verified already
gave is imported once, in the Verified sets. A row whose repository's license is copyleft goes to its family's
``-copyleft`` sets. The files are read through the Hugging Face cache (``hf.fetch``) and parsed with ``pyarrow``; the
sets are written and checked by the set writer the importers share (``corpus_sets``).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import corpus_sets, hf


@dataclass(frozen=True)
class Source:
    """One pinned parquet file of a Hugging Face dataset, and the set family it becomes."""

    family: str
    label: str
    repo: str
    revision: str
    file: str
    sha256: str
    card_sha256: str

    @property
    def source(self) -> str:
        return f"hf://datasets/{self.repo}@{self.revision}"


VERIFIED = Source(
    family="verified",
    label="SWE-bench Verified",
    repo="SWE-bench/SWE-bench_Verified",
    revision="78f471bf655a3137b2e8a75af1501690ec009ec3",
    file="data/test-00000-of-00001.parquet",
    sha256="030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25",
    card_sha256="923e6b481ff75c709737251e602bdc311a9be49235b5c20107366747f5640fe4",
)
TEST = Source(
    family="test",
    label="SWE-bench test",
    repo="SWE-bench/SWE-bench",
    revision="c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2",
    file="data/test-00000-of-00001.parquet",
    sha256="d4f5a245c75319fa8240c540674958c4d491e82edf274b144d43836bdcbc4567",
    card_sha256="6433ee353e763e5c1e6f3e9ef4e9872d7f2e9639effc2e52cc5521f419444764",
)
SOURCES = (VERIFIED, TEST)
DATASET = "swebench"

# The license of each repository's code, which is each row's license: the dataset cards state none. Read on
# 2026-10-06 from the GitHub license API and each repository's LICENSE file on its default branch; a row whose
# repository is not here stops the import until its license is reviewed.
LICENSES = {
    "astropy/astropy": "BSD-3-Clause",
    "django/django": "BSD-3-Clause",
    # Not on the SPDX list: Matplotlib's own PSF-style license agreement, LICENSE/LICENSE.
    "matplotlib/matplotlib": "LicenseRef-Matplotlib",
    "mwaskom/seaborn": "BSD-3-Clause",
    "pallets/flask": "BSD-3-Clause",
    "psf/requests": "Apache-2.0",
    "pydata/xarray": "Apache-2.0",
    # GitHub's license API names it GPL-2.0, which does not tell "only" from "or later".
    "pylint-dev/pylint": "GPL-2.0",
    "pytest-dev/pytest": "MIT",
    "scikit-learn/scikit-learn": "BSD-3-Clause",
    # GitHub: NOASSERTION; LICENSE.rst grants it as the "two clause BSD license".
    "sphinx-doc/sphinx": "BSD-2-Clause",
    # GitHub: NOASSERTION; LICENSE is BSD-3-Clause, with the notices of the code it bundles.
    "sympy/sympy": "BSD-3-Clause",
}
# A row under one of these goes to the `-copyleft` sets, kept apart from the others.
COPYLEFT = frozenset({"GPL-2.0"})

COLUMNS = ("repo", "instance_id", "base_commit", "patch", "problem_statement", "hints_text")
# A row with either of these blank has no case to give; none has at the pinned revisions.
EMPTY = (("problem statement", "problem_statement"), ("patch", "patch"))
SYSTEM = "You are working on the {repo} repository at commit {base_commit}."
HINTS = "\n\nHints:\n"
# bellwether's framing: the data has no tool, so this is the one every case offers.
SUBMIT_PATCH = {
    "type": "function",
    "function": {
        "name": "submit_patch",
        "description": "Submit a patch that resolves the issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "A unified diff that resolves the issue, applied at the repository's base commit.",
                }
            },
            "required": ["patch"],
        },
    },
}


def read_rows(path: Path) -> list[dict]:
    """The rows of a parquet file in file order, with the columns the import uses."""
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=list(COLUMNS)).to_pylist()


def blank(text: str | None) -> bool:
    return text is None or not text.strip()


def request_for(row: dict) -> dict:
    """The request for one row: the repository and base commit as the system turn, the issue as the user turn.

    The user turn is the problem statement verbatim, then, when the row has hints that are not blank, ``HINTS`` and the
    hints verbatim.
    """
    system = SYSTEM.format(repo=row["repo"], base_commit=row["base_commit"])
    user = row["problem_statement"]
    if not blank(row["hints_text"]):
        user += HINTS + row["hints_text"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return {"messages": messages, "tools": [copy.deepcopy(SUBMIT_PATCH)]}


def call_message(patch: str) -> dict:
    """The gold patch as one ``submit_patch`` call, its arguments JSON written with raw Unicode as BFCL's are."""
    arguments = json.dumps({"patch": patch}, ensure_ascii=False)
    return {
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "submit_patch", "arguments": arguments}}],
    }


def content_message(patch: str) -> dict:
    """The gold patch as content: a fenced ``diff`` block.

    A closing fence must start its own line, so a patch that does not end with a newline gets one before it. Every
    patch at the pinned revisions ends with one, and none holds a line of backticks that would close the block early.
    """
    return {"content": "```diff\n" + patch + ("" if patch.endswith("\n") else "\n") + "```"}


def slug(instance_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")


def set_name(family: str, form: str | None, copyleft: bool) -> str:
    """``swebench-<family>`` for render, ``swebench-<family>-<form>`` for parse, and ``-copyleft`` after either."""
    return "-".join(["swebench", family, *([form] if form else []), *(["copyleft"] if copyleft else [])])


def origin(source: Source, row_id: str, spdx: str) -> dict:
    """Where a case came from. A parse case's message comes from the same row, its ``patch``."""
    found = {"dataset": DATASET, "source": source.source, "sha256": source.sha256, "file": source.file}
    return {**found, "row": row_id, "license": spdx}


def build_sets(
    sources: Sequence[tuple[Source, list[dict]]],
    skipped: list[tuple[str, str]] | None = None,
    repeated: list[str] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: for every row a render case, a call case and a content case.

    A row whose problem statement or patch is empty gets no cases, and is appended to ``skipped`` with its reason.
    A row whose instance an earlier source already gave (SWE-bench's test split holds all of Verified) gets no cases of
    its own, and its id is appended to ``repeated``; it must equal the earlier row in every column the import reads,
    or the import stops.
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    first: dict[str, tuple[Source, dict]] = {}
    owner: dict[tuple[str, str], str] = {}

    def add(kind: str, name: str, line: dict, row_id: str) -> None:
        # A case name is unique across every set of a kind (record/corpus.py).
        if (kind, line["name"]) in owner:
            twin = owner[(kind, line["name"])]
            raise ValueError(f"rows {twin!r} and {row_id!r} both become the case name {line['name']}")
        owner[(kind, line["name"])] = row_id
        sets.setdefault((kind, name), []).append(line)

    for source, rows in sources:
        for row in rows:
            row_id = row["instance_id"]
            if row_id in first:
                earlier, kept = first[row_id]
                differ = [column for column in COLUMNS if row[column] != kept[column]]
                if differ:
                    raise ValueError(
                        f"{row_id}: the {source.label} row differs from the {earlier.label} row in {', '.join(differ)}"
                    )
                if repeated is not None:
                    repeated.append(row_id)
                continue
            first[row_id] = (source, row)
            empty = next((what for what, column in EMPTY if blank(row[column])), None)
            if empty is not None:
                if skipped is not None:
                    skipped.append((row_id, f"the {empty} is empty"))
                continue
            if row["repo"] not in LICENSES:
                raise ValueError(f"{row_id}: {row['repo']} is not in the reviewed license table; review it first")
            spdx = LICENSES[row["repo"]]
            copyleft = spdx in COPYLEFT
            request = request_for(row)
            notes = f"{source.label} {row_id}"
            found = origin(source, row_id, spdx)
            name = slug(row_id)
            render = {"name": f"swebench-{source.family}-{name}", "request": request, "notes": notes, "origin": found}
            add("render", set_name(source.family, None, copyleft), render, row_id)
            for form, message, probe in (
                ("call", call_message(row["patch"]), "the gold patch as one submit_patch call"),
                ("content", content_message(row["patch"]), "the gold patch in a diff block"),
            ):
                line = {
                    "name": f"swebench-{source.family}-{form}-{name}",
                    "request": request,
                    "message": message,
                    "notes": f"{notes}: {probe}",
                    "origin": found,
                }
                add("parse", set_name(source.family, form, copyleft), line, row_id)
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``swebench-*`` files the import no longer writes."""
    return corpus_sets.write(sets, corpus_dir, "swebench-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, "swebench-", "SWE-bench set")


def run(args: argparse.Namespace) -> int:
    sources = []
    for source in SOURCES:
        card = hf.fetch(source.repo, source.revision, hf.CARD, source.card_sha256).read_text("utf-8")
        hf.check_card_license(source.repo, card, None)  # the cards state no license; each row's is its repository's
        sources.append((source, read_rows(hf.fetch(source.repo, source.revision, source.file, source.sha256))))
    skipped: list[tuple[str, str]] = []
    repeated: list[str] = []
    sets = build_sets(sources, skipped=skipped, repeated=repeated)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            pinned = ", ".join(source.source for source in SOURCES)
            print(f"{args.corpus}: the SWE-bench sets equal a fresh import of {pinned}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    rows_by_reason: dict[str, list[str]] = {}
    for row_id, why in skipped:
        rows_by_reason.setdefault(why, []).append(row_id)
    for why, row_ids in rows_by_reason.items():
        print(f"no cases for {len(row_ids)} row(s) ({', '.join(row_ids)}): {why}")
    if repeated:
        print(
            f"{len(repeated)} {TEST.label} row(s) are also {VERIFIED.label} rows; each is imported once, in the "
            "Verified sets"
        )
    write_sets(sets, args.corpus)
    return 0
