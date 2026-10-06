"""What a verify run concludes: which cases pass given the known differences, as JSON, JUnit XML and text.

Results are written as they come, so a run of any size holds one case at a time: each case to look at is printed at
once, and the JSON report and the JUnit XML are put together at the end from temporary files.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO
from xml.sax.saxutils import quoteattr

from bellwether import __version__
from bellwether.manifest import Manifest

from .render import CannotVerify

VERDICTS = ("match", "regression", "rejected", "missing", "measurement_failed")
EXCUSABLE = ("regression", "rejected")
SETUP = ("missing", "measurement_failed")  # verdicts about the setup, never excused
LEADING_KEYS = ("id", "model", "set", "verdict", "passed", "known")
WITHOUT_CASE = "listed, but there is no such case; remove the entry"
JUNIT_COUNTS = (("failure", "failures"), ("error", "errors"), ("skipped", "skipped"))


def load_known(path: Path) -> dict[str, str]:
    """Known differences: a TOML table mapping a fixture id to the reason SMG is expected to differ on it."""
    try:
        known = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise CannotVerify(f"{path}: {err}") from None
    bad = sorted(key for key, reason in known.items() if not isinstance(reason, str) or not reason.strip())
    if bad:
        raise CannotVerify(f'{path}: each entry is "<fixture id>" = "<reason>"; not so for {", ".join(bad)}')
    return known


def judge(result: dict, known: dict[str, str]) -> None:
    """Mark a case passed or not.

    A case passes when it matches and is not listed, or when it is a regression or rejected and is listed. A listed
    case that matches fails, so an entry goes as soon as SMG is fixed and the list cannot rot. A missing capture
    line and a failed measurement fail even when listed: they are about the setup (which file is read, whether the
    ``rid`` reached the engine, what answered), not about how SMG renders.
    """
    reason = known.get(result["id"])
    result["known"] = reason
    if result["verdict"] == "match":
        result["passed"] = reason is None
    else:
        result["passed"] = reason is not None and result["verdict"] in EXCUSABLE


def known_without_case(known: dict[str, str], manifests: list[Manifest], judged: set[str]) -> list[str]:
    """Listed ids in a verified model's render fixtures that name no case: the corpus dropped or renamed them.

    ``judged`` holds the listed ids the run judged. Entries for other models or kinds are left alone; this run says
    nothing about them.
    """
    namespaces = tuple(f"{manifest.slug}/render/" for manifest in manifests)
    return sorted(case_id for case_id in known if case_id.startswith(namespaces) and case_id not in judged)


class Writer:
    """The run's results, written as they come: text for each case to look at, and the JSON report and the JUnit XML
    from temporary files when the run finishes.

    Nothing is written to ``report`` or ``junit`` unless the run finishes.
    """

    def __init__(self, *, report: Path | None, junit: Path | None, kind: str = "render") -> None:
        self.kind = kind
        self.report, self.junit = report, junit
        self.counts = dict.fromkeys(VERDICTS, 0)
        self.passed = self.failed = 0
        self._cases = _scratch() if report else None
        self._suites: dict[str, _Suite] = {}

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        for scratch in [self._cases, *(suite.scratch for suite in self._suites.values())]:
            if scratch is not None:
                scratch.close()

    def add(self, result: dict) -> None:
        self.counts[result["verdict"]] += 1
        if result["passed"]:
            self.passed += 1
        else:
            self.failed += 1
        if not (result["verdict"] == "match" and result["passed"] and result["status"] == 200):
            print(f"{result['verdict']} {result['id']}: {describe(result)}")
        if self._cases is not None:
            # The verdict and whether it passes first; the response body, which can be long, last.
            entry = {
                **{k: result[k] for k in LEADING_KEYS},
                **{k: v for k, v in result.items() if k not in LEADING_KEYS},
            }
            self._cases.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if self.junit:
            self._suite(f"{result['model']} {self.kind}").add(self._testcase(result))

    def finish(self, *, provenance: dict, known_without_case: list[str], models_without_cases: list[str]) -> dict:
        """Print the totals, write the JSON report and the JUnit XML, and return the report without its cases."""
        cases = self.passed + self.failed
        written = {
            "kind": self.kind,
            "provenance": provenance,
            "summary": {"cases": cases, **self.counts, "passed": self.passed, "failed": self.failed},
            "passed": self.failed == 0 and not known_without_case,
            "known_without_case": known_without_case,
            "models_without_cases": models_without_cases,
        }
        for case_id in known_without_case:
            print(f"known {case_id}: {WITHOUT_CASE}")
        for model in models_without_cases:
            print(f"{model}: no render cases")
        counts = ", ".join(f"{self.counts[verdict]} {verdict}" for verdict in VERDICTS)
        tally = f"{self.passed} pass, {self.failed} fail"
        if known_without_case:
            tally += f", {len(known_without_case)} listed without a case"
        print(f"{self.kind}: {cases} cases ({counts}); {tally}; {'passed' if written['passed'] else 'failed'}")
        if self.report:
            self._write_json(written)
        if self.junit:
            self._write_junit(known_without_case)
        return written

    def _testcase(self, result: dict) -> ET.Element:
        """One testcase. A verdict about the case (``regression``, ``rejected``, a listed case that matches) is a
        failure; one about the setup (``missing``, ``measurement_failed``) is an error, the test not having run; a
        listed known difference is skipped with its reason."""
        classname = f"{result['id'].split('/')[0]}/{self.kind}/{result['set']}"
        testcase = ET.Element("testcase", classname=classname, name=result["id"])
        if result["passed"]:
            if result["known"] is not None:
                ET.SubElement(testcase, "skipped", message=f"known difference: {result['known']}")
            return testcase
        tag = "error" if result["verdict"] in SETUP else "failure"
        failure_type = "known-but-matches" if result["verdict"] == "match" else result["verdict"]
        element = ET.SubElement(testcase, tag, type=failure_type, message=describe(result))
        if result["verdict"] == "regression":
            details = ("index", "lengths", "window", "text_equal", "text")
            element.text = json.dumps({key: result[key] for key in details if key in result}, ensure_ascii=False)
        return testcase

    def _suite(self, name: str) -> _Suite:
        suite = self._suites.get(name)
        if suite is None:
            suite = self._suites[name] = _Suite(name)
        return suite

    def _write_json(self, written: dict) -> None:
        self.report.parent.mkdir(parents=True, exist_ok=True)
        with self.report.open("w", encoding="utf-8") as out:
            out.write("{\n")
            for key, value in written.items():
                out.write(f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},\n")
            out.write(' "cases": [')
            self._cases.seek(0)
            for number, line in enumerate(self._cases):
                out.write(("\n  " if number == 0 else ",\n  ") + line.rstrip("\n"))
            out.write("\n ]\n}\n")

    def _write_junit(self, known_without_case: list[str]) -> None:
        suites = list(self._suites.values())
        if known_without_case:
            gone = _Suite("known differences")
            for case_id in known_without_case:
                testcase = ET.Element("testcase", classname="known differences", name=case_id)
                ET.SubElement(testcase, "failure", type="known-without-case", message=WITHOUT_CASE)
                gone.add(testcase)
            suites.append(gone)
        totals = {"tests": sum(suite.tests for suite in suites)}
        for _, attribute in JUNIT_COUNTS:
            totals[attribute] = sum(suite.counts[attribute] for suite in suites)
        self.junit.parent.mkdir(parents=True, exist_ok=True)
        with self.junit.open("w", encoding="utf-8") as out:
            out.write("<?xml version='1.0' encoding='utf-8'?>\n")
            out.write(f"<testsuites name={quoteattr(f'bellwether verify {self.kind}')}{_attributes(totals)}>\n")
            for suite in suites:
                suite.write(out)
            out.write("</testsuites>\n")
        for suite in suites:
            suite.scratch.close()


class _Suite:
    """One JUnit testsuite, its testcases kept in a temporary file until the run finishes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tests = 0
        self.counts = {attribute: 0 for _, attribute in JUNIT_COUNTS}
        self.scratch = _scratch()

    def add(self, testcase: ET.Element) -> None:
        self.tests += 1
        for tag, attribute in JUNIT_COUNTS:
            self.counts[attribute] += testcase.find(tag) is not None
        ET.indent(testcase, space="  ", level=2)
        self.scratch.write("    " + ET.tostring(testcase, encoding="unicode") + "\n")

    def write(self, out: IO[str]) -> None:
        out.write(f"  <testsuite name={quoteattr(self.name)}{_attributes({'tests': self.tests, **self.counts})}>\n")
        self.scratch.seek(0)
        for line in self.scratch:
            out.write(line)
        out.write("  </testsuite>\n")


def _attributes(counts: dict[str, int]) -> str:
    return "".join(f' {name}="{value}"' for name, value in counts.items())


def _scratch() -> IO[str]:
    return tempfile.TemporaryFile("w+", encoding="utf-8")


def provenance(*, url: str, capture: Path, known: Path | None, manifests: list[Manifest]) -> dict:
    return {
        "bellwether": bellwether_version(),
        "smg": url,
        "capture": str(capture),
        "known": None if known is None else str(known),
        "manifests": [{"model": m.model, "revision": m.revision, "path": str(m.path)} for m in manifests],
    }


def bellwether_version() -> dict:
    """The bellwether version, and the commit when it runs from a checkout of its own repository.

    An installed wheel has no commit, and a virtual environment inside another repository must not report that
    repository's commit, so the checkout counts only when this package is its ``src/bellwether``.
    """
    package = Path(__file__).resolve().parents[1]
    found = {"version": __version__, "commit": None, "dirty": None}
    try:
        if (Path(_git(package, "rev-parse", "--show-toplevel")) / "src" / "bellwether").resolve() != package:
            return found
        found["commit"] = _git(package, "rev-parse", "HEAD")
        found["dirty"] = bool(_git(package, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        pass
    return found


def _git(where: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(where), *args], capture_output=True, text=True, check=True).stdout.strip()


def describe(result: dict) -> str:
    """Why a case has its verdict, and what its known-difference entry, if any, does to it."""
    detail = ""
    if result["verdict"] == "regression":
        lengths = result["lengths"]
        detail = f"ids differ from index {result['index']} (reference {lengths['reference']} ids, smg {lengths['smg']})"
        if result["text_equal"] is True:
            detail += "; same text, so tokenization"
        elif result["text_equal"] is False:
            detail += f"; text differs from character {result['text']['index']}, so rendering"
        else:
            detail += "; the capture line does not carry the text"
    elif result["verdict"] == "rejected":
        detail = f"HTTP {result['status']} {result['code']}: {result['message']}"
    elif result["verdict"] == "missing":
        detail = "SMG answered, but no capture line carries this case's id"
    elif result["verdict"] == "measurement_failed":
        detail = f"HTTP {result['status']}: {result['message']}; not SMG refusing the request, so nothing was measured"
    elif result["known"] is not None:
        detail = "matches"
    if result["verdict"] in ("match", "regression") and result["status"] != 200:
        detail = f"{detail}; SMG then answered HTTP {result['status']}: {result['message']}".lstrip("; ")
    if result["known"] is not None:
        if result["passed"]:
            detail += f"; known: {result['known']}"
        elif result["verdict"] == "match":
            detail += f", but it is listed as a known difference ({result['known']}); remove the entry"
        else:
            detail += f"; listed as a known difference, but {result['verdict'].replace('_', ' ')} is about the setup"
    return detail
