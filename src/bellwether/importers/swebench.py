"""SWE-bench Verified and SWE-bench's test split as corpus sets: an issue as the user turn, its gold patch as one call.

SWE-bench has no prompt and no tools, so the request and the messages are bellwether's framing, not SWE-bench's:

- the request: a system turn naming the repository and the base commit, then the problem statement verbatim as the
  user turn, with the hints after ``HINTS`` when the row has any; ``tools`` is ``submit_patch`` alone, on every case;
- a call case: the gold patch as the only argument of one ``submit_patch`` call;
- a content case: the gold patch in a fenced ``diff`` block.

Verified is imported whole. SWE-bench's test split holds all of Verified's rows, so a test row that Verified already
gave is imported once, in the Verified sets. The files are read through the Hugging Face cache (``hf.fetch``) and parsed
with ``pyarrow``; the sets are written and checked by the set writer the importers share (``corpus_sets``).

The dataset cards state no license. A row's code, its gold patch, is under its repository's license at the row's base
commit, read from the repository's license file there (``license_of``). ``swebench_licenses.json`` pins that file and
any NOTICE file at every base commit, by repository, commit and sha256; the import fetches them (``github.fetch``),
copies them next to the sets, and names on each parse line the license, its holder and the copies that go with it. A
row whose code is copyleft goes to its family's ``-copyleft`` sets.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import corpus_sets, github, hf


@dataclass(frozen=True)
class Source:
    """One pinned parquet file of a Hugging Face dataset, and the set family it becomes."""

    family: str
    label: str
    dataset_id: str
    revision: str
    file: str
    sha256: str
    card_sha256: str

    @property
    def uri(self) -> str:
        """The pinned revision, as ``origin.source`` names it."""
        return f"hf://datasets/{self.dataset_id}@{self.revision}"


VERIFIED = Source(
    family="verified",
    label="SWE-bench Verified",
    dataset_id="SWE-bench/SWE-bench_Verified",
    revision="78f471bf655a3137b2e8a75af1501690ec009ec3",
    file="data/test-00000-of-00001.parquet",
    sha256="030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25",
    card_sha256="923e6b481ff75c709737251e602bdc311a9be49235b5c20107366747f5640fe4",
)
TEST = Source(
    family="test",
    label="SWE-bench test",
    dataset_id="SWE-bench/SWE-bench",
    revision="c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2",
    file="data/test-00000-of-00001.parquet",
    sha256="d4f5a245c75319fa8240c540674958c4d491e82edf274b144d43836bdcbc4567",
    card_sha256="6433ee353e763e5c1e6f3e9ef4e9872d7f2e9639effc2e52cc5521f419444764",
)
SOURCES = (VERIFIED, TEST)
DATASET = "swebench"
# The repositories' license files that the import copies next to its sets, under the corpus root, and the table that
# pins them: for every row's base commit, its repository's license file, its NOTICE file if any, and the full text of
# the Apache License where the license file only points to it, each by repository, commit and sha256. Built from the
# repositories' history by scripts/swebench_licenses.py.
LICENSE_DIR = "licenses"
LICENSE_TABLE = Path(__file__).with_name("swebench_licenses.json")
# Who holds the copyright in each repository's code, as its license or NOTICE file names them. The license files of
# xarray and pylint are the Apache License and the GPL alone: xarray's README.rst names "xarray Developers", and
# pylint's files name the contributors listed in CONTRIBUTORS.txt.
HOLDERS = {
    "astropy/astropy": "Astropy Developers",
    "django/django": "Django Software Foundation and individual contributors",
    "matplotlib/matplotlib": "Matplotlib Development Team",
    "mwaskom/seaborn": "Michael L. Waskom",
    "pallets/flask": "Pallets",
    "psf/requests": "Kenneth Reitz",
    "pydata/xarray": "xarray Developers",
    "pylint-dev/pylint": "the pylint contributors",
    "pytest-dev/pytest": "Holger Krekel and others",
    "scikit-learn/scikit-learn": "The scikit-learn developers",
    "sphinx-doc/sphinx": "the Sphinx team",
    "sympy/sympy": "SymPy Development Team",
}
# The GPL's text does not say whether a program is under that version only or any later one; the repository does.
# pylint declares GPL-2.0-or-later in setup.cfg or pyproject.toml at every base commit (scripts/swebench_licenses.py
# checks it).
GPL_DECLARED = {"pylint-dev/pylint": "GPL-2.0-or-later"}
# Every license the import can read from a license file, and whether code under it is copyleft: such rows go to the
# -copyleft sets. A license file that reads as none of these stops the import until it is reviewed.
LICENSE_KINDS = {
    "Apache-2.0": "permissive",
    "BSD-2-Clause": "permissive",
    "BSD-3-Clause": "permissive",
    "GPL-2.0-only": "copyleft",
    "GPL-2.0-or-later": "copyleft",
    "ISC": "permissive",
    "LicenseRef-Matplotlib": "permissive",  # not on the SPDX list: Matplotlib's own license agreement
    "MIT": "permissive",
}
# A render line holds only the issue text and its hints, comments by GitHub users under no license that is established;
# SPDX's word for that.
NO_LICENSE = "NOASSERTION"


@dataclass(frozen=True)
class RowLicense:
    """The license of a row's code: what its repository's license file grants at the row's base commit."""

    spdx: str
    holder: str
    notices: tuple[str, ...]  # the copied files that go with the code, as paths under the corpus root

    @property
    def copyleft(self) -> bool:
        return LICENSE_KINDS[self.spdx] == "copyleft"


def license_of(text: str, repository: str) -> str:
    """The SPDX id of the license that a license file's own words grant.

    Read from the text, not from what anyone says of it: the first terms the file states, up to its first disclaimer,
    so that the licenses of bundled code which some files list after their own do not count. A text that is none of
    the licenses in ``LICENSE_KINDS``, or the GPL for a repository with no entry in ``GPL_DECLARED``, stops the import.
    """
    words = " ".join(text.split())
    own = words.split("THIS SOFTWARE IS PROVIDED", 1)[0]
    if words.startswith("GNU GENERAL PUBLIC LICENSE Version 2, June 1991"):
        if repository not in GPL_DECLARED:
            raise ValueError(f"{repository}: the GPL's text alone cannot tell 'only' from 'or later'; declare it first")
        return GPL_DECLARED[repository]
    if words.startswith("License agreement for matplotlib versions 1.3.0 and later"):
        return "LicenseRef-Matplotlib"
    if (
        words.startswith("Apache License Version 2.0, January 2004")
        or "Licensed under the Apache License, Version 2.0" in own
    ):
        return "Apache-2.0"
    if "Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee" in own:
        return "ISC"
    if "Permission is hereby granted, free of charge, to any person obtaining a copy of this software" in own:
        return "MIT"
    if "Redistribution and use in source and binary forms, with or without modification, are permitted" in own:
        return "BSD-3-Clause" if "endorse or promote products derived from this software" in own else "BSD-2-Clause"
    raise ValueError(f"{repository}: a license file the import cannot read as a license it knows; review it first")


def load_license_table() -> dict:
    """The committed table of license files: ``files`` pins each one, ``versions`` names those at each base commit."""
    return json.loads(LICENSE_TABLE.read_text("utf-8"))


def licenses_from(table: dict, texts: dict[str, bytes]) -> dict[tuple[str, str], RowLicense]:
    """``(repository, base commit) -> RowLicense`` for every base commit in ``table``, read from the files' ``texts``.

    Code under the Apache License goes with a full text of it (Apache-2.0 4(a)), which a license file that only points
    to the License is not, so the table joins one to it; a base commit whose files hold none stops the import.
    """
    licenses = {}
    for version in table["versions"]:
        repository = version["repository"]
        names = [name for name in (version["license"], version["notice"], version["full_text"]) if name]
        spdx = license_of(texts[version["license"]].decode("utf-8"), repository)
        if spdx == "Apache-2.0" and not any(
            b" ".join(texts[name].split()).startswith(b"Apache License Version 2.0, January 2004") for name in names
        ):
            raise ValueError(f"{repository}: {version['license']} goes with no full text of the Apache License")
        if repository not in HOLDERS:
            raise ValueError(f"{repository}: no copyright holder is named for it; name it first")
        notices = tuple(f"{LICENSE_DIR}/{name}" for name in names)
        for commit in version["commits"]:
            licenses[(repository, commit)] = RowLicense(spdx, HOLDERS[repository], notices)
    return licenses


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
    return {"messages": messages, "tools": [SUBMIT_PATCH]}


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


def origin(source: Source, row: dict, code: RowLicense | None) -> dict:
    """Where a case came from: the row, its repository, and the license of what the line holds.

    A render line (``code`` None) holds the issue text and its hints alone: no license is established for them. A parse
    line's message is the row's gold patch, code under its repository's license at the base commit, which names its
    copyright holder and the copied files that go with it. A parse case's message comes from the same row, its
    ``patch``, so no other file is named.
    """
    found = {"dataset": DATASET, "source": source.uri, "sha256": source.sha256, "file": source.file}
    found |= {"row": row["instance_id"], "repository": row["repo"]}
    if code is None:
        return {**found, "license": NO_LICENSE}
    return {**found, "license": code.spdx, "copyright": code.holder, "notices": list(code.notices)}


def build_sets(
    sources: Sequence[tuple[Source, list[dict]]],
    licenses: dict[tuple[str, str], RowLicense],
    skipped: list[tuple[str, str]] | None = None,
    repeated: list[str] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: for every row a render case, a call case and a content case.

    ``licenses`` gives the license of each row's code by its repository and base commit (``licenses_from``); a row
    whose base commit it does not name stops the import. A row whose code is copyleft goes to the ``-copyleft`` sets.
    A row whose problem statement or patch is empty gets no cases, and is appended to ``skipped`` with its reason.
    A row whose instance an earlier source already gave (SWE-bench's test split holds all of Verified) gets no cases of
    its own, and its id is appended to ``repeated``; it must equal the earlier row in every column the import reads,
    or the import stops.
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    first: dict[str, tuple[Source, dict]] = {}
    owner: dict[tuple[str, str], str] = {}

    def add(kind: str, into: str, line: dict, row_id: str) -> None:
        # A case name is unique across every set of a kind (record/corpus.py).
        if (kind, line["name"]) in owner:
            twin = owner[(kind, line["name"])]
            raise ValueError(f"rows {twin!r} and {row_id!r} both become the case name {line['name']}")
        owner[(kind, line["name"])] = row_id
        sets.setdefault((kind, into), []).append(line)

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
            code = licenses.get((row["repo"], row["base_commit"]))
            if code is None:
                raise ValueError(
                    f"{row_id}: {row['repo']} at {row['base_commit']} is not in {LICENSE_TABLE.name}; "
                    "rebuild it with scripts/swebench_licenses.py"
                )
            request = request_for(row)
            notes = f"{source.label} {row_id}"
            case_slug = slug(row_id)
            render = {
                "name": f"swebench-{source.family}-{case_slug}",
                "request": request,
                "notes": notes,
                "origin": origin(source, row, None),
            }
            add("render", set_name(source.family, None, code.copyleft), render, row_id)
            for form, message, probe in (
                ("call", call_message(row["patch"]), "the gold patch as one submit_patch call"),
                ("content", content_message(row["patch"]), "the gold patch in a diff block"),
            ):
                line = {
                    "name": f"swebench-{source.family}-{form}-{case_slug}",
                    "request": request,
                    "message": message,
                    "notes": f"{notes}: {probe}",
                    "origin": origin(source, row, code),
                }
                add("parse", set_name(source.family, form, code.copyleft), line, row_id)
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, files: dict[str, bytes]) -> list[Path]:
    """Write every set and ``files``, and remove ``swebench-*`` set files the import no longer writes.

    ``files`` are the import's other files, the copied license files, as bytes by path under ``corpus_dir``.
    """
    written = corpus_sets.write(sets, corpus_dir, "swebench-")
    for relative, content in sorted(files.items()):
        path = corpus_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written.append(path)
    return written


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, files: dict[str, bytes]) -> list[str]:
    """One line per set file, or file of ``files``, that differs from a fresh import; empty when none does."""
    problems = corpus_sets.check(sets, corpus_dir, "swebench-", "SWE-bench set")
    for relative, content in sorted(files.items()):
        path = corpus_dir / relative
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != content:
            problems.append(f"{path}: differs from a fresh import")
    return problems


def license_texts(table: dict, cache: Path) -> dict[str, bytes]:
    """The bytes of every file the license table pins, fetched by repository, commit and sha256."""
    texts = {}
    for name, pin in table["files"].items():
        owner, repository = pin["repository"].split("/")
        texts[name] = github.fetch(
            owner, repository, pin["commit"], pin["path"], pin["sha256"], cache=cache
        ).read_bytes()
    return texts


def run(args: argparse.Namespace) -> int:
    sources = []
    for source in SOURCES:
        # The cards state no license: each row's code is under its repository's, read below.
        card = hf.fetch(source.dataset_id, source.revision, hf.CARD, source.card_sha256).read_text("utf-8")
        hf.check_card_license(source.dataset_id, card, None)
        sources.append((source, read_rows(hf.fetch(source.dataset_id, source.revision, source.file, source.sha256))))
    table = load_license_table()
    texts = license_texts(table, args.cache)
    files = {f"{LICENSE_DIR}/{name}": text for name, text in texts.items()}
    skipped: list[tuple[str, str]] = []
    repeated: list[str] = []
    sets = build_sets(sources, licenses_from(table, texts), skipped=skipped, repeated=repeated)
    if args.check:
        problems = check_sets(sets, args.corpus, files)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            pinned = ", ".join(source.uri for source in SOURCES)
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
    write_sets(sets, args.corpus, files)
    return 0
