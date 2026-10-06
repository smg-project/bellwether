import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bellwether.cli import main
from bellwether.importers import corpus_sets, github, hf, pinned, swebench

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
COMMIT = "d26b2424437dabeeca94d7900b37d2df4410da0c"
ISSUE = "UsernameValidator allows a trailing newline\r\nDescription\n"
PATCH = (
    "diff --git a/django/core/validators.py b/django/core/validators.py\n"
    "--- a/django/core/validators.py\n"
    "+++ b/django/core/validators.py\n"
    "@@ -1 +1 @@\n"
    "-    regex = r'^[\\w.@+-]+$'\n"
    "+    regex = r'^[\\w.@+-]+\\Z'\t# \"Café\"\n"
)


def row(**fields) -> dict:
    base = {
        "repo": "django/django",
        "instance_id": "django__django-11099",
        "base_commit": COMMIT,
        "patch": PATCH,
        "problem_statement": ISSUE,
        "hints_text": "",
    }
    return {**base, **fields}


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


def test_the_request_is_the_repository_at_its_commit_the_issue_verbatim_and_one_tool():
    request = swebench.request_for(row())
    assert request == {
        "messages": [
            {"role": "system", "content": f"You are working on the django/django repository at commit {COMMIT}."},
            {"role": "user", "content": ISSUE},
        ],
        "tools": [SUBMIT_PATCH],
    }
    assert list(request) == ["messages", "tools"]
    assert list(request["tools"][0]) == ["type", "function"]
    assert list(request["tools"][0]["function"]) == ["name", "description", "parameters"]
    assert list(request["tools"][0]["function"]["parameters"]) == ["type", "properties", "required"]


def test_hints_follow_the_issue_after_a_fixed_separator_and_blank_hints_add_nothing():
    hinted = swebench.request_for(row(hints_text="Use \\Z, not $.\n"))
    assert hinted["messages"][1]["content"] == ISSUE + "\n\nHints:\nUse \\Z, not $.\n"
    for blank in ("", "\n", "\n\n", None):
        assert swebench.request_for(row(hints_text=blank))["messages"][1]["content"] == ISSUE


def test_the_call_carries_the_patch_as_its_one_argument_in_json_with_raw_unicode():
    assert swebench.call_message('a\t"b"\\c é\n') == {
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "submit_patch", "arguments": '{"patch": "a\\t\\"b\\"\\\\c é\\n"}'},
            }
        ],
    }
    assert json.loads(swebench.call_message(PATCH)["tool_calls"][0]["function"]["arguments"]) == {"patch": PATCH}


def test_the_content_is_the_patch_in_a_diff_block_whose_closing_fence_starts_a_line():
    assert swebench.content_message(PATCH) == {"content": "```diff\n" + PATCH + "```"}
    assert swebench.content_message("-a\n+b") == {"content": "```diff\n-a\n+b\n```"}


def write_parquet(path, rows: list[dict]):
    pq.write_table(pa.Table.from_pylist([{**r, "test_patch": "", "FAIL_TO_PASS": ["t"]} for r in rows]), path)
    return path


BSD_3 = (
    "Copyright (c) Django Software Foundation and individual contributors.\nAll rights reserved.\n\n"
    "Redistribution and use in source and binary forms, with or without modification,\n"
    "are permitted provided that the following conditions are met:\n\n"
    "    1. Redistributions of source code must retain the above copyright notice,\n"
    "       this list of conditions and the following disclaimer.\n\n"
    "    2. Redistributions in binary form must reproduce the above copyright\n"
    "       notice, this list of conditions and the following disclaimer.\n\n"
    "    3. Neither the name of Django nor the names of its contributors may be used\n"
    "       to endorse or promote products derived from this software without\n"
    "       specific prior written permission.\n\n"
    'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"\n'
)
# Two clauses for the project, then a bundled library's three after the first disclaimer, as Sphinx's LICENSE has.
BSD_2 = (
    "License for Sphinx\n==================\n\nCopyright (c) 2007-2019 by the Sphinx team (see AUTHORS file).\n"
    "All rights reserved.\n\nRedistribution and use in source and binary forms, with or without\n"
    "modification, are permitted provided that the following conditions are\nmet:\n\n"
    "* Redistributions of source code must retain the above copyright\n  notice.\n\n"
    "* Redistributions in binary form must reproduce the above copyright\n  notice.\n\n"
    'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"AS IS".\n\n'
    "Licenses for incorporated software\n==================================\n\n"
    "Redistribution and use in source and binary forms ... Neither the name of the copyright holder may be\n"
    "used to endorse or promote products derived from this software.\n"
)
MIT = (
    "The MIT License (MIT)\n\nCopyright (c) 2004 Holger Krekel and others\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy of\n"
    'this software and associated documentation files (the "Software"), to deal in\n'
)
ISC = (
    "Copyright (c) 2012 Kenneth Reitz.\n\n"
    "Permission to use, copy, modify, and/or distribute this software for any\n"
    "purpose with or without fee is hereby granted, provided that the above\n"
    "copyright notice and this permission notice appear in all copies.\n"
)
APACHE_NOTICE = (
    "Copyright 2017 Kenneth Reitz\n\n"
    '   Licensed under the Apache License, Version 2.0 (the "License");\n'
    "   you may not use this file except in compliance with the License.\n"
)
APACHE_TEXT = (
    "\n                                 Apache License\n                           Version 2.0, January 2004\n"
    "                        http://www.apache.org/licenses/\n\n"
    "   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"
)
MATPLOTLIB = (
    "License agreement for matplotlib versions 1.3.0 and later\n"
    "=========================================================\n\n"
    "1. This LICENSE AGREEMENT is between the Matplotlib Development Team\n"
    '("MDT"), and the Individual or Organization\n'
)
GPL_2 = (
    "                    GNU GENERAL PUBLIC LICENSE\n                       Version 2, June 1991\n\n"
    " Copyright (C) 1989, 1991 Free Software Foundation, Inc.,\n"
)
LGPL_2_1 = "                  GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 2.1, February 1999\n"


@pytest.mark.parametrize(
    ("text", "repository", "spdx"),
    [
        (BSD_3, "django/django", "BSD-3-Clause"),
        (BSD_2, "sphinx-doc/sphinx", "BSD-2-Clause"),
        (MIT, "pytest-dev/pytest", "MIT"),
        (ISC, "psf/requests", "ISC"),
        (APACHE_NOTICE, "psf/requests", "Apache-2.0"),
        (APACHE_TEXT, "pydata/xarray", "Apache-2.0"),
        (MATPLOTLIB, "matplotlib/matplotlib", "LicenseRef-Matplotlib"),
        (GPL_2, "pylint-dev/pylint", "GPL-2.0-or-later"),
    ],
    ids=["bsd-3", "bsd-2", "mit", "isc", "apache-notice", "apache-text", "matplotlib", "gpl-2"],
)
def test_a_license_file_is_read_as_the_license_its_own_words_grant(text, repository, spdx):
    assert swebench.license_of(text, repository) == spdx


def test_a_license_text_the_import_does_not_know_or_a_gpl_whose_version_is_not_declared_stops_it():
    with pytest.raises(
        ValueError, match="astroid/astroid: a license file the import cannot read as a license it knows"
    ):
        swebench.license_of(LGPL_2_1, "astroid/astroid")
    with pytest.raises(ValueError, match="org/gpl: the GPL's text alone cannot tell 'only' from 'or later'"):
        swebench.license_of(GPL_2, "org/gpl")


def committed_licenses() -> tuple[dict, dict[str, bytes], dict]:
    """The committed license table, the copies of its files in the corpus, and the license of every base commit."""
    table = swebench.load_license_table()
    texts = {name: (CORPUS / swebench.LICENSE_DIR / name).read_bytes() for name in table["files"]}
    return table, texts, swebench.licenses_from(table, texts)


# What each repository's license files grant at the base commits the rows use: requests moved from ISC to Apache-2.0
# in 2013, and pylint declares GPL-2.0-or-later in its packaging metadata.
REPOSITORY_LICENSES = {
    "astropy/astropy": {"BSD-3-Clause"},
    "django/django": {"BSD-3-Clause"},
    "matplotlib/matplotlib": {"LicenseRef-Matplotlib"},
    "mwaskom/seaborn": {"BSD-3-Clause"},
    "pallets/flask": {"BSD-3-Clause"},
    "psf/requests": {"ISC", "Apache-2.0"},
    "pydata/xarray": {"Apache-2.0"},
    "pylint-dev/pylint": {"GPL-2.0-or-later"},
    "pytest-dev/pytest": {"MIT"},
    "scikit-learn/scikit-learn": {"BSD-3-Clause"},
    "sphinx-doc/sphinx": {"BSD-2-Clause"},
    "sympy/sympy": {"BSD-3-Clause"},
}


def test_each_copied_license_file_is_the_pinned_one():
    table, texts, _ = committed_licenses()
    assert {name: hashlib.sha256(text).hexdigest() for name, text in texts.items()} == {
        name: pin["sha256"] for name, pin in table["files"].items()
    }


def test_every_base_commit_reads_as_its_repositorys_license():
    _, _, licenses = committed_licenses()
    found: dict[str, set[str]] = {}
    for (repository, _), row_license in licenses.items():
        found.setdefault(repository, set()).add(row_license.spdx)
    assert found == REPOSITORY_LICENSES


def test_each_copyright_holder_is_named_in_its_repositorys_license_or_notice_file():
    table, texts, _ = committed_licenses()
    for version in table["versions"]:
        repository = version["repository"]
        if repository in ("pydata/xarray", "pylint-dev/pylint"):
            continue  # their license files are the Apache and GPL texts alone, which name no holder
        files = b"\n".join(texts[name] for name in (version["license"], version["notice"]) if name)
        assert swebench.HOLDERS[repository] in " ".join(files.decode("utf-8").split()), (repository, version["license"])


def test_requests_rows_based_in_2012_are_isc_and_later_ones_apache_with_the_licenses_full_text():
    _, texts, licenses = committed_licenses()
    for commit in ("27b55a74d7b9bd2f8c60fd0ee342bcbbf40e0a66", "a0df2cbb10419037d11d04352b3175405ab52941"):
        assert licenses[("psf/requests", commit)].spdx == "ISC"  # requests-774 and requests-863
    later = licenses[("psf/requests", "2d763c90ae6ccee1a3c64bf70add776a2ba395ef")]  # requests-4106, 2017
    assert later.spdx == "Apache-2.0"
    [notice, full_text] = [texts[path.removeprefix(f"{swebench.LICENSE_DIR}/")] for path in later.notices]
    assert b"Licensed under the Apache License, Version 2.0" in notice
    assert full_text.split()[:4] == [b"Apache", b"License", b"Version", b"2.0,"]


def test_the_committed_requests_cases_based_in_2012_say_isc():
    found = {}
    for form in ("call", "content"):
        for raw in (CORPUS / "parse" / f"swebench-test-{form}.jsonl").read_bytes().split(b"\n"):
            if b'"repository": "psf/requests"' in raw:
                line = json.loads(raw)
                found[(form, line["origin"]["row"])] = line["origin"]["license"]
    for form in ("call", "content"):
        assert found[(form, "psf__requests-774")] == found[(form, "psf__requests-863")] == "ISC"
        assert found[(form, "psf__requests-4106")] == "Apache-2.0"


DJANGO_COPY = f"swebench-django-django-{COMMIT[:12]}-LICENSE"
PYLINT_COPY = f"swebench-pylint-dev-pylint-{COMMIT[:12]}-LICENSE"
DJANGO = swebench.RowLicense(
    "BSD-3-Clause", "Django Software Foundation and individual contributors", (f"licenses/{DJANGO_COPY}",)
)
PYLINT_LICENSE = swebench.RowLicense("GPL-2.0-or-later", "the pylint contributors", (f"licenses/{PYLINT_COPY}",))
# The licenses of the made-up rows below: every one is based at COMMIT.
LICENSES = {("django/django", COMMIT): DJANGO, ("pylint-dev/pylint", COMMIT): PYLINT_LICENSE}


def test_rows_are_read_from_parquet_in_file_order_with_the_columns_the_import_uses(tmp_path):
    rows = [row(instance_id="django__django-11100", hints_text="Hint.\n"), row()]
    assert swebench.read_rows(write_parquet(tmp_path / "test.parquet", rows)) == rows


CARD = "---\ndataset_info:\n  features:\n  - name: patch\n---\n\nlicense: mit, in the text, does not count\n"


VERIFIED_ORIGIN = {
    "dataset": "swebench",
    "source": "hf:datasets/SWE-bench/SWE-bench_Verified@78f471bf655a3137b2e8a75af1501690ec009ec3",
    "sha256": "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25",
    "file": "data/test-00000-of-00001.parquet",
}
PYLINT = row(repo="pylint-dev/pylint", instance_id="pylint-dev__pylint-4551")


def test_sets_per_family_and_form_with_copyleft_rows_in_their_own_sets():
    test_row = row(instance_id="django__django-10097")
    sets = swebench.build_sets([(swebench.VERIFIED, [row(), PYLINT]), (swebench.TEST, [test_row])], LICENSES)
    assert sorted(sets) == [
        ("parse", "swebench-test-call"),
        ("parse", "swebench-test-content"),
        ("parse", "swebench-verified-call"),
        ("parse", "swebench-verified-call-copyleft"),
        ("parse", "swebench-verified-content"),
        ("parse", "swebench-verified-content-copyleft"),
        ("render", "swebench-test"),
        ("render", "swebench-verified"),
        ("render", "swebench-verified-copyleft"),
    ]
    # A render line holds only the issue text and hints, comments by GitHub users with no license established.
    render_origin = {
        **VERIFIED_ORIGIN,
        "row": "django__django-11099",
        "repository": "django/django",
        "license": "NOASSERTION",
        "written": ["system prompt", "submit_patch tool"],
    }
    # A parse line's message is the gold patch: code under the license of its repository at the row's base commit.
    parse_origin = {
        **{key: value for key, value in render_origin.items() if key != "written"},
        "license": "BSD-3-Clause",
        "copyright": "Django Software Foundation and individual contributors",
        "notices": [f"licenses/{DJANGO_COPY}"],
    }
    call_origin = {**parse_origin, "written": ["system prompt", "submit_patch tool", "submit_patch call"]}
    content_origin = {**parse_origin, "written": ["system prompt", "submit_patch tool", "code fences"]}
    request = swebench.request_for(row())
    assert sets[("render", "swebench-verified")] == [
        {
            "name": "swebench-verified-django-django-11099",
            "request": request,
            "notes": "SWE-bench Verified django__django-11099",
            "origin": render_origin,
        }
    ]
    [call] = sets[("parse", "swebench-verified-call")]
    assert call == {
        "name": "swebench-verified-call-django-django-11099",
        "request": request,
        "message": swebench.call_message(PATCH),
        "notes": "SWE-bench Verified django__django-11099: the gold patch as one submit_patch call",
        "origin": call_origin,
    }
    assert list(call) == ["name", "request", "message", "notes", "origin"]
    assert list(call["origin"]) == [
        "dataset", "source", "sha256", "file", "row", "repository", "license", "copyright", "notices", "written"
    ]  # fmt: skip
    [content] = sets[("parse", "swebench-verified-content")]
    assert content["name"] == "swebench-verified-content-django-django-11099"
    assert content["message"] == swebench.content_message(PATCH)
    assert content["notes"] == "SWE-bench Verified django__django-11099: the gold patch in a diff block"
    assert content["origin"] == content_origin
    [copyleft] = sets[("render", "swebench-verified-copyleft")]
    assert copyleft["name"] == "swebench-verified-pylint-dev-pylint-4551"
    pylint = {**VERIFIED_ORIGIN, "row": "pylint-dev__pylint-4551", "repository": "pylint-dev/pylint"}
    assert copyleft["origin"] == {**pylint, "license": "NOASSERTION", "written": ["system prompt", "submit_patch tool"]}
    for form, wrote in (("call", "submit_patch call"), ("content", "code fences")):
        [line] = sets[("parse", f"swebench-verified-{form}-copyleft")]
        assert line["origin"] == {
            **pylint,
            "license": "GPL-2.0-or-later",
            "copyright": "the pylint contributors",
            "notices": [f"licenses/{PYLINT_COPY}"],
            "written": ["system prompt", "submit_patch tool", wrote],
        }
    [tested] = sets[("render", "swebench-test")]
    assert tested["name"] == "swebench-test-django-django-10097"
    assert tested["notes"] == "SWE-bench test django__django-10097"
    assert tested["origin"] == {
        "dataset": "swebench",
        "source": "hf:datasets/SWE-bench/SWE-bench@c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2",
        "sha256": "d4f5a245c75319fa8240c540674958c4d491e82edf274b144d43836bdcbc4567",
        "file": "data/test-00000-of-00001.parquet",
        "row": "django__django-10097",
        "repository": "django/django",
        "license": "NOASSERTION",
        "written": ["system prompt", "submit_patch tool"],
    }


def test_origin_names_the_text_bellwether_wrote_in_each_line_in_the_order_of_its_fields():
    hinted = row(instance_id="django__django-11100", hints_text="Hint.\n")
    sets = swebench.build_sets([(swebench.VERIFIED, [row(), hinted])], LICENSES)
    written = {line["name"]: line["origin"]["written"] for lines in sets.values() for line in lines}
    request = ["system prompt", "submit_patch tool"]
    hinted_request = ["system prompt", "hints separator", "submit_patch tool"]
    assert written == {
        "swebench-verified-django-django-11099": request,
        "swebench-verified-call-django-django-11099": [*request, "submit_patch call"],
        "swebench-verified-content-django-django-11099": [*request, "code fences"],
        "swebench-verified-django-django-11100": hinted_request,
        "swebench-verified-call-django-django-11100": [*hinted_request, "submit_patch call"],
        "swebench-verified-content-django-django-11100": [*hinted_request, "code fences"],
    }
    assert all(list(line["origin"])[-1] == "written" for lines in sets.values() for line in lines)


def test_the_corpus_readme_says_what_each_written_name_stands_for():
    readme = (CORPUS / "README.md").read_text("utf-8")
    for name in swebench.WRITTEN.values():
        assert f"`{name}`" in readme, name


def names(lines: list[dict]) -> list[str]:
    return [line["name"] for line in lines]


def test_test_rows_that_are_verified_rows_are_left_to_the_verified_sets_and_reported():
    sources = [(swebench.VERIFIED, [row()]), (swebench.TEST, [row(), row(instance_id="django__django-10097")])]
    sets = swebench.build_sets(sources, LICENSES)
    assert names(sets[("render", "swebench-test")]) == ["swebench-test-django-django-10097"]
    assert names(sets[("parse", "swebench-test-call")]) == ["swebench-test-call-django-django-10097"]
    assert names(sets[("render", "swebench-verified")]) == ["swebench-verified-django-django-11099"]
    repeated: list[str] = []
    swebench.build_sets(sources, LICENSES, repeated=repeated)
    assert repeated == ["django__django-11099"]


def test_a_repeated_row_that_differs_from_the_first_stops_the_import():
    sources = [(swebench.VERIFIED, [row()]), (swebench.TEST, [row(hints_text="New.\n", patch="-a\n+b\n")])]
    with pytest.raises(
        ValueError,
        match="django__django-11099: the SWE-bench test row differs from the SWE-bench "
        "Verified row in patch, hints_text",
    ):
        swebench.build_sets(sources, LICENSES)


def test_rows_with_an_empty_patch_or_problem_statement_are_skipped_and_each_is_named():
    rows = [
        row(instance_id="django__django-1", patch=""),
        row(instance_id="django__django-2", problem_statement="\n"),
        row(),
    ]
    sets = swebench.build_sets([(swebench.VERIFIED, rows)], LICENSES)
    assert names(sets[("render", "swebench-verified")]) == ["swebench-verified-django-django-11099"]
    assert names(sets[("parse", "swebench-verified-content")]) == ["swebench-verified-content-django-django-11099"]
    skipped: list[tuple[str, str]] = []
    swebench.build_sets(
        [(swebench.VERIFIED, [*rows, row(instance_id="django__django-3", patch=None)])], LICENSES, skipped=skipped
    )
    assert skipped == [
        ("django__django-1", "the patch is empty"),
        ("django__django-2", "the problem statement is empty"),
        ("django__django-3", "the patch is empty"),
    ]


def test_a_row_whose_patch_carries_code_under_other_terms_is_left_out_and_named(monkeypatch):
    terms = "django/core/validators.py holds code from elsewhere, under its own license"
    monkeypatch.setitem(swebench.OTHER_TERMS, "django__django-11099", terms)
    skipped: list[tuple[str, str]] = []
    sets = swebench.build_sets(
        [(swebench.VERIFIED, [row(), row(instance_id="django__django-11100")])], LICENSES, skipped=skipped
    )
    assert skipped == [("django__django-11099", terms)]
    assert all("11099" not in line["name"] for lines in sets.values() for line in lines)
    assert names(sets[("render", "swebench-verified")]) == ["swebench-verified-django-django-11100"]


# Lines 896, 846, 400 and 1332 of the parse call set, as #46's review found them: a patch to requests' copy of urllib3
# (MIT), a file from packaging (Apache-2.0 or BSD-2-Clause), Python's code in Django (PSF), docutils' (public domain).
REVIEWED_OTHER_TERMS = (
    "psf__requests-2678",
    "mwaskom__seaborn-2766",
    "django__django-13915",
    "sphinx-doc__sphinx-7356",
)


def test_the_rows_the_review_found_under_other_terms_are_listed_and_not_in_the_committed_corpus():
    assert set(REVIEWED_OTHER_TERMS) <= set(swebench.OTHER_TERMS)
    for kind in ("render", "parse"):
        for path in sorted((CORPUS / kind).glob("swebench-*.jsonl")):
            for row_id in REVIEWED_OTHER_TERMS:
                assert f'"row": "{row_id}"'.encode() not in path.read_bytes(), (path.name, row_id)


def test_a_row_whose_base_commit_is_not_in_the_license_table_stops_the_import():
    rows = [row(repo="numpy/numpy", instance_id="numpy__numpy-1")]
    with pytest.raises(ValueError, match=f"numpy__numpy-1: numpy/numpy at {COMMIT} is not in swebench_licenses.json"):
        swebench.build_sets([(swebench.VERIFIED, rows)], LICENSES)


def test_two_rows_with_one_case_name_stop_the_import():
    twin = row(instance_id="Django__Django-11099")
    with pytest.raises(
        ValueError,
        match="rows 'django__django-11099' and 'Django__Django-11099' both become the case "
        "name swebench-verified-django-django-11099",
    ):
        swebench.build_sets([(swebench.VERIFIED, [row(), twin])], LICENSES)


def test_written_sets_and_license_copies_check_clean_and_a_changed_missing_or_stale_file_is_reported(tmp_path):
    sets, corpus = swebench.build_sets([(swebench.VERIFIED, [row(), PYLINT])], LICENSES), tmp_path / "corpus"
    files = {f"licenses/{DJANGO_COPY}": BSD_3.encode(), f"licenses/{PYLINT_COPY}": GPL_2.encode()}
    (corpus / "render").mkdir(parents=True)
    for other in ("common.jsonl", "bfcl-simple-python.jsonl", "swebench-old.jsonl"):
        (corpus / "render" / other).write_text("{}\n")
    swebench.write_sets(sets, corpus, files)
    assert not (corpus / "render" / "swebench-old.jsonl").exists()
    assert (corpus / "render" / "common.jsonl").read_text() == "{}\n"
    assert (corpus / "render" / "bfcl-simple-python.jsonl").read_text() == "{}\n"
    assert (corpus / "licenses" / DJANGO_COPY).read_bytes() == BSD_3.encode()
    text = (corpus / "parse" / "swebench-verified-call.jsonl").read_bytes().decode("utf-8")
    assert "Café" in text and "\\r\\n" in text and text.endswith("}\n") and text.count("\n") == 1
    assert swebench.check_sets(sets, corpus, files) == []
    (corpus / "parse" / "swebench-verified-call.jsonl").write_text("{}\n")
    (corpus / "parse" / "swebench-verified-content-copyleft.jsonl").unlink()
    (corpus / "render" / "swebench-stale.jsonl").write_text("{}\n")
    (corpus / "licenses" / DJANGO_COPY).write_text(MIT)
    (corpus / "licenses" / PYLINT_COPY).unlink()
    assert swebench.check_sets(sets, corpus, files) == [
        f"{corpus / 'licenses' / DJANGO_COPY}: differs from a fresh import",
        f"{corpus / 'licenses' / PYLINT_COPY}: missing",
        f"{corpus / 'parse' / 'swebench-verified-call.jsonl'}: differs from a fresh import",
        f"{corpus / 'parse' / 'swebench-verified-content-copyleft.jsonl'}: missing",
        f"{corpus / 'render' / 'swebench-stale.jsonl'}: no SWE-bench set writes it",
    ]


def fake_hub(tmp_path, verified: list[dict], test: list[dict], cards: dict[str, str] | None = None):
    """``hf.fetch`` over local files: each source's parquet file and card, recording what was asked for.

    Each source's card is ``CARD``, which states no license, unless ``cards`` gives its repository another.
    """
    files = {
        (swebench.VERIFIED.dataset_id, swebench.VERIFIED.file): write_parquet(tmp_path / "verified.parquet", verified),
        (swebench.TEST.dataset_id, swebench.TEST.file): write_parquet(tmp_path / "test.parquet", test),
    }
    for source in swebench.SOURCES:
        card = tmp_path / f"{source.family}-README.md"
        card.write_text((cards or {}).get(source.dataset_id, CARD))
        files[(source.dataset_id, "README.md")] = card
    calls: list[tuple] = []

    def fetch(repo, revision, filename, sha256, cache=pinned.CACHE):
        calls.append((repo, revision, filename, sha256, cache))
        return files[(repo, filename)]

    return fetch, calls


def fake_licenses(tmp_path, monkeypatch, texts: dict[str, str]) -> list[tuple]:
    """A license table naming one license file per repository at ``COMMIT``, and ``github.fetch`` serving it.

    Returns the fetches made, as ``(owner, repo, commit, path, sha256)``.
    """
    files, versions, served, calls = {}, [], {}, []
    for repository, text in texts.items():
        name = f"swebench-{repository.replace('/', '-')}-{COMMIT[:12]}-LICENSE"
        sha256 = hashlib.sha256(text.encode()).hexdigest()
        files[name] = {"repository": repository, "commit": COMMIT, "path": "LICENSE", "sha256": sha256}
        versions.append(
            {"repository": repository, "license": name, "notice": None, "full_text": None, "commits": [COMMIT]}
        )
        served[(repository, COMMIT, "LICENSE", sha256)] = tmp_path / "github" / name
        served[(repository, COMMIT, "LICENSE", sha256)].parent.mkdir(parents=True, exist_ok=True)
        served[(repository, COMMIT, "LICENSE", sha256)].write_text(text)

    def fetch(owner, repo, commit, path, sha256, cache=pinned.CACHE):
        calls.append((owner, repo, commit, path, sha256))
        return served[(f"{owner}/{repo}", commit, path, sha256)]

    monkeypatch.setattr(swebench, "load_license_table", lambda: {"files": files, "versions": versions})
    monkeypatch.setattr(github, "fetch", fetch)
    return calls


def test_the_command_writes_then_checks_and_names_what_it_leaves_out(tmp_path, monkeypatch, capsys):
    empty = row(instance_id="django__django-1", patch="")
    tested = row(instance_id="django__django-10097", problem_statement="Another issue.\n")
    # The same request and patch as django__django-11099: each of its cases repeats one of that row's.
    repeat = row(instance_id="django__django-10098")
    fetch, calls = fake_hub(tmp_path, [row(), PYLINT, empty], [row(), tested, repeat])
    monkeypatch.setattr(hf, "fetch", fetch)
    fetched = fake_licenses(tmp_path, monkeypatch, {"django/django": BSD_3, "pylint-dev/pylint": GPL_2})
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    command = ["import", "swebench", "--corpus", str(corpus), "--cache", str(cache)]
    assert main([*command, "--check"]) == 1
    err = capsys.readouterr().err
    assert f"{corpus / 'render' / 'swebench-verified.jsonl'}: missing" in err
    assert f"{corpus / 'licenses' / DJANGO_COPY}: missing" in err
    assert main(command) == 0
    assert (corpus / "licenses" / DJANGO_COPY).read_text() == BSD_3
    assert (corpus / "licenses" / PYLINT_COPY).read_text() == GPL_2
    assert main([*command, "--check"]) == 0
    out = capsys.readouterr().out.splitlines()
    for line in (
        "no case swebench-test-django-django-10098: it repeats swebench-verified-django-django-11099",
        "no case swebench-test-call-django-django-10098: it repeats swebench-verified-call-django-django-11099",
        f"{corpus / 'render' / 'swebench-verified.jsonl'}: 1 cases",
        f"{corpus / 'render' / 'swebench-test.jsonl'}: 1 cases, 1 left out as repeats",
        f"{corpus / 'parse' / 'swebench-test-call.jsonl'}: 1 cases, 1 left out as repeats, 1 distinct messages",
        f"{corpus / 'parse' / 'swebench-verified-call-copyleft.jsonl'}: 1 cases, 1 distinct messages",
        f"{corpus}: 9 cases in the 9 SWE-bench sets, 3 left out as repeats, 2 distinct messages",
        "no case for 1 row(s) (django__django-1): the patch is empty",
        "1 SWE-bench test row(s) are also SWE-bench Verified rows; each is imported once, in the Verified sets",
    ):
        assert line in out
    assert out[-1].startswith(f"{corpus}: the SWE-bench sets equal a fresh import of hf:datasets/")
    cards = {(s.dataset_id, s.revision, "README.md", s.card_sha256, cache) for s in swebench.SOURCES}
    rows = {(s.dataset_id, s.revision, s.file, s.sha256, cache) for s in swebench.SOURCES}
    assert set(calls) == cards | rows and len(cards | rows) == 4
    django = ("django", "django", COMMIT, "LICENSE", hashlib.sha256(BSD_3.encode()).hexdigest())
    assert django in fetched
    (corpus / "licenses" / DJANGO_COPY).write_text(MIT)
    assert main([*command, "--check"]) == 1
    assert capsys.readouterr().err == f"{corpus / 'licenses' / DJANGO_COPY}: differs from a fresh import\n"


@pytest.mark.parametrize("source", swebench.SOURCES, ids=lambda source: source.family)
def test_the_command_refuses_a_dataset_card_that_states_a_license_and_writes_nothing(tmp_path, monkeypatch, source):
    cards = {source.dataset_id: "---\nlicense: mit\n---\n"}
    fetch, _ = fake_hub(tmp_path, [row()], [row(instance_id="django__django-10097")], cards=cards)
    monkeypatch.setattr(hf, "fetch", fetch)
    pinned_card = re.escape(hf.source(source.dataset_id, source.revision))
    refused = f"^{pinned_card} README.md: the card's license is 'mit', not the reviewed None; review it"
    with pytest.raises(ValueError, match=refused):
        main(["import", "swebench", "--corpus", str(tmp_path / "corpus")])
    assert not (tmp_path / "corpus").exists()


def test_the_command_writes_every_set_compressed_past_the_limit_and_checks_them(tmp_path, monkeypatch):
    fetch, _ = fake_hub(tmp_path, [row()], [row(instance_id="django__django-10097", problem_statement="Other.\n")])
    monkeypatch.setattr(hf, "fetch", fetch)
    fake_licenses(tmp_path, monkeypatch, {"django/django": BSD_3})
    monkeypatch.setattr(corpus_sets, "LIMIT", 1)
    corpus = tmp_path / "corpus"
    assert main(["import", "swebench", "--corpus", str(corpus)]) == 0
    assert sorted(path.name for path in (corpus / "render").iterdir()) == [
        "swebench-test.jsonl.zst",
        "swebench-verified.jsonl.zst",
    ]
    assert (corpus / "licenses" / DJANGO_COPY).read_text() == BSD_3  # beside the sets, plain and outside the limit
    assert main(["import", "swebench", "--corpus", str(corpus), "--check"]) == 0
