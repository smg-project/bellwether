"""SWE-bench Verified and SWE-bench's test split as corpus sets: an issue as the user turn, its gold patch as one call.

SWE-bench has no prompt and no tools, so the request and the messages are bellwether's framing, not SWE-bench's:

- the request: a system turn naming the repository and the base commit, then the problem statement verbatim as the
  user turn, with the hints after ``HINTS`` when the row has any; ``tools`` is ``submit_patch`` alone, on every case;
- a call case: the gold patch as the only argument of one ``submit_patch`` call;
- a content case: the gold patch in a fenced ``diff`` block.

Verified is imported whole. SWE-bench's test split holds all of Verified's rows, so a test row that Verified already
gave is imported once, in the Verified sets. The files are read with ``hf.fetch`` into the importers' cache and parsed
with ``pyarrow``; the sets are written and checked, and what the import keeps and leaves out is printed, by the set
writer the importers share (``corpus_sets``).

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
        """The pinned revision, as ``origin.source`` names it (``hf.source``)."""
        return hf.source(self.dataset_id, self.revision)


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
# Rows whose gold patch carries code that its own file, or the patch itself, puts under terms other than the
# repository's license: another license, or another holder's license file. The copied license files cover the
# repository's terms only, so these rows are left out. Found by reading, for every row, each file the patch touches at
# the base commit and every line of the patch for license and copyright statements, leaving out each repository's own
# headers, and reviewing what is left by hand (scripts/swebench_licenses.py --terms prints it). A file whose header
# names another copyright holder under the repository's own license does not count.
OTHER_TERMS = {
    **dict.fromkeys(
        ("astropy__astropy-6938", "astropy__astropy-7218", "astropy__astropy-8707", "astropy__astropy-13417"),
        "astropy/io/fits/ is under PyFITS's license, licenses/PYFITS.rst",
    ),
    **dict.fromkeys(
        ("astropy__astropy-14508", "astropy__astropy-14528", "astropy__astropy-14578", "astropy__astropy-14598"),
        "astropy/io/fits/ is under PyFITS's license, licenses/PYFITS.rst",
    ),
    **dict.fromkeys(
        ("astropy__astropy-14163", "astropy__astropy-14213", "astropy__astropy-14484"),
        "astropy/units/quantity_helper/function_helpers.py holds code from NumPy, under licenses/NUMPY_LICENSE.rst",
    ),
    "astropy__astropy-13158": "astropy/modeling/functional_models.py holds code from cpfX, under its own BSD license",
    **dict.fromkeys(
        ("django__django-11374", "django__django-11638", "django__django-11848", "django__django-13915"),
        "django/utils/http.py holds code from Python, under the PSF license, LICENSE.python",
    ),
    "django__django-13410": "django/core/files/locks.py holds code from a Python Cookbook recipe and Roundup",
    "matplotlib__matplotlib-23198": "lib/matplotlib/backends/qt_editor/figureoptions.py is under the MIT License",
    "matplotlib__matplotlib-26341": "lib/matplotlib/sankey.py states its own license, BSD",
    "mwaskom__seaborn-2766": (
        "the patch adds seaborn/external/version.py from packaging, under Apache-2.0 or BSD-2-Clause"
    ),
    **dict.fromkeys(
        ("mwaskom__seaborn-2996", "mwaskom__seaborn-3216"),
        "seaborn/_compat.py holds code from matplotlib, under matplotlib's license",
    ),
    "psf__requests-2466": "requests/packages/__init__.py is from pip, under the MIT License",
    "psf__requests-2678": "requests/packages/urllib3/ is urllib3, under the MIT License",
    **dict.fromkeys(
        ("pydata__xarray-3631", "pydata__xarray-4339", "pydata__xarray-4758", "pydata__xarray-5233"),
        "the patch touches code from pandas, under pandas' license and holders",
    ),
    **dict.fromkeys(
        ("pydata__xarray-6135", "pydata__xarray-7019", "pydata__xarray-7444"),
        "the patch touches code from pandas, under pandas' license and holders",
    ),
    **dict.fromkeys(
        ("pydata__xarray-5682", "pydata__xarray-7052", "pydata__xarray-7179"),
        "xarray/plot/utils.py holds code from seaborn, under licenses/SEABORN_LICENSE",
    ),
    "scikit-learn__scikit-learn-10427": (
        "the patch adds sklearn/externals/_pilutil.py from SciPy, under SciPy's license"
    ),
    "scikit-learn__scikit-learn-14067": (
        "the patch adds sklearn/externals/_scipy_linalg.py from SciPy, under SciPy's license"
    ),
    **dict.fromkeys(
        ("scikit-learn__scikit-learn-12938", "scikit-learn__scikit-learn-13584"),
        "sklearn/utils/_pprint.py is from Python, under the PSF license",
    ),
    "scikit-learn__scikit-learn-7760": "sklearn/utils/_unittest_backport.py is from Python, under the PSF license",
    **dict.fromkeys(
        ("scikit-learn__scikit-learn-12486", "scikit-learn__scikit-learn-13467", "scikit-learn__scikit-learn-14898"),
        "sklearn/metrics/scorer.py states its own license, Simplified BSD",
    ),
    **dict.fromkeys(
        (
            "sphinx-doc__sphinx-7234",
            "sphinx-doc__sphinx-7557",
            "sphinx-doc__sphinx-7757",
            "sphinx-doc__sphinx-7831",
            "sphinx-doc__sphinx-8007",
            "sphinx-doc__sphinx-8278",
            "sphinx-doc__sphinx-8362",
            "sphinx-doc__sphinx-9261",
            "sphinx-doc__sphinx-9281",
            "sphinx-doc__sphinx-9461",
            "sphinx-doc__sphinx-9654",
            "sphinx-doc__sphinx-9797",
            "sphinx-doc__sphinx-9931",
            "sphinx-doc__sphinx-9997",
        ),
        "sphinx/util/inspect.py holds code from Python, under the PSF license",
    ),
    **dict.fromkeys(
        ("sphinx-doc__sphinx-7356", "sphinx-doc__sphinx-7374"),
        "sphinx/util/nodes.py holds code from docutils, placed in the public domain",
    ),
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
# The text bellwether writes into a SWE-bench case, which the dataset does not have, by the short names that
# ``origin.written`` lists in the order of the line's fields: the system turn; HINTS, when the row has hints; the one
# tool; and in a parse case's message, the call around the gold patch or the diff block's fences.
WRITTEN = {
    "system": "system prompt",
    "hints": "hints separator",
    "tool": "submit_patch tool",
    "call": "submit_patch call",
    "content": "code fences",
}
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


def has_hints(row: dict) -> bool:
    """Whether the user turn carries the row's hints after ``HINTS``: when they are not blank."""
    return not blank(row["hints_text"])


def request_for(row: dict) -> dict:
    """The request for one row: the repository and base commit as the system turn, the issue as the user turn.

    The user turn is the problem statement verbatim, then, when the row has hints that are not blank, ``HINTS`` and the
    hints verbatim.
    """
    system = SYSTEM.format(repo=row["repo"], base_commit=row["base_commit"])
    user = row["problem_statement"]
    if has_hints(row):
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


def origin(source: Source, row: dict, code: RowLicense | None, form: str | None) -> dict:
    """Where a case came from: the row, its repository, the license of what the line holds, and what bellwether wrote.

    A render line (``code`` and ``form`` None) holds the issue text and its hints alone: no license is established for
    them. A parse line's message is the row's gold patch, code under its repository's license at the base commit, which
    names its copyright holder and the copied files that go with it; ``form`` is the message's, ``call`` or ``content``.
    A parse case's message comes from the same row, its ``patch``, so no other file is named. ``written`` names the text
    bellwether wrote into the line (``WRITTEN``).
    """
    found = {"dataset": DATASET, "source": source.uri, "sha256": source.sha256, "file": source.file}
    found |= {"row": row["instance_id"], "repository": row["repo"]}
    if code is None:
        found["license"] = NO_LICENSE
    else:
        found |= {"license": code.spdx, "copyright": code.holder, "notices": list(code.notices)}
    written = [WRITTEN["system"], *([WRITTEN["hints"]] if has_hints(row) else []), WRITTEN["tool"]]
    return {**found, "written": [*written, *([WRITTEN[form]] if form else [])]}


def build_sets(
    sources: Sequence[tuple[Source, list[dict]]],
    licenses: dict[tuple[str, str], RowLicense],
    skipped: list[tuple[str, str]] | None = None,
    repeated: list[str] | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: for every row a render case, a call case and a content case.

    ``licenses`` gives the license of each row's code by its repository and base commit (``licenses_from``); a row
    whose base commit it does not name stops the import. A row whose code is copyleft goes to the ``-copyleft`` sets.
    A row whose problem statement or patch is empty, or whose patch carries code under other terms (``OTHER_TERMS``),
    gets no cases, and is appended to ``skipped`` with its reason.
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
            if row_id in OTHER_TERMS:
                if skipped is not None:
                    skipped.append((row_id, OTHER_TERMS[row_id]))
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
                "origin": origin(source, row, None, None),
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
                    "origin": origin(source, row, code, form),
                }
                add("parse", set_name(source.family, form, code.copyleft), line, row_id)
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, files: dict[str, bytes]) -> list[Path]:
    """Write every set and the copied license files (``files``), and remove ``swebench-*`` sets no longer written."""
    return corpus_sets.write(sets, corpus_dir, "swebench-", files)


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path, files: dict[str, bytes]) -> list[str]:
    """One line per set file, or copied license file, that differs from a fresh import; empty when none does."""
    return corpus_sets.check(sets, corpus_dir, "swebench-", "SWE-bench set", files)


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
        hf.check_card_license(source.dataset_id, source.revision, source.card_sha256, None, cache=args.cache)
        path = hf.fetch(source.dataset_id, source.revision, source.file, source.sha256, cache=args.cache)
        sources.append((source, read_rows(path)))
    table = load_license_table()
    texts = license_texts(table, args.cache)
    files = {f"{LICENSE_DIR}/{name}": text for name, text in texts.items()}
    skipped: list[tuple[str, str]] = []
    repeated: list[str] = []
    sets = build_sets(sources, licenses_from(table, texts), skipped=skipped, repeated=repeated)
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    if args.check:
        problems = check_sets(kept, args.corpus, files)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            pinned = ", ".join(source.uri for source in SOURCES)
            print(f"{args.corpus}: the SWE-bench sets equal a fresh import of {pinned}")
        return 1 if problems else 0
    corpus_sets.report("SWE-bench", sets, kept, repeats, args.corpus)
    corpus_sets.report_skipped(skipped)
    if repeated:
        print(
            f"{len(repeated)} {TEST.label} row(s) are also {VERIFIED.label} rows; each is imported once, in the "
            "Verified sets"
        )
    write_sets(kept, args.corpus, files)
    return 0
