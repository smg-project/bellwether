"""What a verify run concludes: which cases pass given the known differences, as JSON, JUnit XML and text.

Results are written as they come, so a run of any size holds one case at a time: each case to look at is printed at
once, and the JSON report and the JUnit XML are put together at the end from temporary files.
"""

from __future__ import annotations

import json
import re
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
ENTRY_KEYS = {"verdict", "code", "reason", "issue"}
FIXTURE_ID = re.compile(r"(?P<slug>[a-z0-9.-]+)/(?P<kind>render|parse|tokenize|detokenize)/[a-z0-9-]+")  # the schema's
WITHOUT_CASE = "listed, but there is no such case; remove the entry"
JUNIT_COUNTS = (("failure", "failures"), ("error", "errors"), ("skipped", "skipped"))


def load_known(path: Path) -> dict[str, dict]:
    """Known differences: one TOML table per fixture id, stating what SMG is known to do on that case.

    ``verdict`` is ``regression`` or ``rejected``, the verdicts about the case; ``missing`` and
    ``measurement_failed`` are about the setup and are never known differences. A ``rejected`` entry gives the
    ``code`` SMG refuses the case with, a string or the number 400 as SMG answers it. ``reason`` says why, and
    ``issue`` links the issue that tracks it. A key verify does not read is refused, so a misspelt one cannot pass
    unseen.
    """
    try:
        known = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise CannotVerify(f"{path}: {err}") from None
    for case_id, entry in known.items():
        problem = _entry_problem(entry)
        if problem is not None:
            raise CannotVerify(f"{path}: {case_id} {problem}")
    return known


def _entry_problem(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return 'is not a table: an entry is ["<fixture id>"] with a verdict, a reason and an issue'
    stray = sorted(set(entry) - ENTRY_KEYS)
    if stray:
        return f"has keys verify does not read: {', '.join(stray)}"
    if entry.get("verdict") not in EXCUSABLE:
        return f"has no verdict of {' or '.join(EXCUSABLE)}; one about the setup is never a known difference"
    code = entry.get("code")
    if entry["verdict"] == "rejected" and not (isinstance(code, str) or type(code) is int):
        return "is rejected without the code SMG refuses it with"
    if entry["verdict"] != "rejected" and "code" in entry:
        return "has a code, which only a rejected entry carries"
    if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
        return "has no reason"
    if not isinstance(entry.get("issue"), str) or not entry["issue"].startswith("https://"):
        return "has no issue, a link starting with https://"
    return None


def judge(result: dict, known: dict[str, dict]) -> None:
    """Mark a case passed or not.

    A case that is not listed passes when it matches. A listed case passes only while it has the outcome its entry
    states: the verdict, and for ``rejected`` the code. Any other outcome fails: a match, so an entry goes as soon
    as SMG is fixed; another verdict or code, so a changed failure is looked at again; and ``missing`` or a failed
    measurement, which are about the setup (which file is read, whether the ``rid`` reached the engine, what
    answered), not about how SMG renders.
    """
    entry = known.get(result["id"])
    result["known"] = entry
    if entry is None:
        result["passed"] = result["verdict"] == "match"
    else:
        result["passed"] = result["verdict"] == entry["verdict"] and result.get("code") == entry.get("code")


def known_without_case(
    known: dict[str, dict], fixtures: Path, manifests: list[Manifest], listed: set[str], every_set: bool
) -> tuple[list[str], list[str]]:
    """The listed ids that name no case, and those for cases this run does not verify.

    ``listed`` holds the listed ids among the run's cases. Any other id names no case when it is not a fixture id,
    when no manifest under ``fixtures`` has its slug (a misspelt slug, a model whose manifest is gone), or when it is
    a render id of a verified model, which the corpus dropped or renamed, in a run of every set. The rest, render ids
    of models not verified in this run or of sets it did not select, and ids of other kinds, are counted and listed
    but not judged.
    """
    slugs = {path.parent.name for path in fixtures.glob("*/manifest.toml")}
    verified = {manifest.slug for manifest in manifests}
    without, outside = [], []
    for case_id in sorted(set(known) - listed):
        found = FIXTURE_ID.fullmatch(case_id)
        verified_render = found is not None and found["slug"] in verified and found["kind"] == "render"
        if found is None or found["slug"] not in slugs or (verified_render and every_set):
            without.append(case_id)
        else:
            outside.append(case_id)
    return without, outside


class Writer:
    """The run's results, written as they come: text for each case to look at, and the JSON report and the JUnit XML
    from temporary files when the run finishes.

    A run that stops before its last case still finishes: the cases answered are reported with their verdicts, the
    case it stopped at with the error, and every case after it as not sent. Nothing is written to ``report`` or
    ``junit`` if the run gives no verdict at all.
    """

    def __init__(self, *, report: Path | None, junit: Path | None, kind: str = "render") -> None:
        self.kind = kind
        self.report, self.junit = report, junit
        self.counts = dict.fromkeys(VERDICTS, 0)
        self.passed = self.failed = 0
        self._cases = _scratch() if report else None
        self._not_sent = _scratch() if report else None
        self._suites: dict[str, _Suite] = {}
        self.stopped: dict | None = None  # the case the run stopped at, and why
        self.unsent = 0

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, *exc_info: object) -> None:
        for scratch in [self._cases, self._not_sent, *(suite.scratch for suite in self._suites.values())]:
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

    def stop(self, model: str, set_name: str, case_id: str | None, error: str) -> None:
        """The run stopped at this case (None: between cases, at a set it could not read again) for ``error``."""
        self.stopped = {"case": case_id, "error": error}
        if case_id is not None and self.junit:
            testcase = ET.Element("testcase", classname=_classname(case_id, self.kind, set_name), name=case_id)
            ET.SubElement(testcase, "error", type="stopped", message=error)
            self._suite(f"{model} {self.kind}").add(testcase)

    def not_sent(self, model: str, set_name: str, case_id: str) -> None:
        """A case after the one the run stopped at."""
        self.unsent += 1
        if self._not_sent is not None:
            self._not_sent.write(json.dumps(case_id, ensure_ascii=False) + "\n")
        if self.junit:
            testcase = ET.Element("testcase", classname=_classname(case_id, self.kind, set_name), name=case_id)
            message = f"not sent: the run stopped at {self.stopped['case'] or 'a set it could not read'}"
            ET.SubElement(testcase, "error", type="not-sent", message=message)
            self._suite(f"{model} {self.kind}").add(testcase)

    def finish(
        self,
        *,
        provenance: dict,
        capture: dict,
        known_without_case: list[str],
        known_outside_run: list[str],
        models_without_cases: list[str],
    ) -> dict:
        """Print the totals, write the JSON report and the JUnit XML, and return the report without its cases."""
        cases = self.passed + self.failed
        summary = {"cases": cases, **self.counts, "passed": self.passed, "failed": self.failed, "not_sent": self.unsent}
        written = {
            "kind": self.kind,
            "provenance": provenance,
            "summary": summary,
            "passed": self.failed == 0 and not known_without_case and self.stopped is None,
            "stopped": self.stopped,
            "capture": capture,
            "known_without_case": known_without_case,
            "known_outside_run": known_outside_run,
            "models_without_cases": models_without_cases,
        }
        for case_id in known_without_case:
            print(f"known {case_id}: {WITHOUT_CASE}")
        if known_outside_run:
            print(f"{len(known_outside_run)} listed known differences are for cases this run does not verify")
        for model in models_without_cases:
            print(f"{model}: no render cases")
        lines = f"capture: {capture['lines']} lines written during the run, {capture['joined']} joined a case"
        lines += f", {capture['other']} did not" + ("; a line was still being written" if capture["unfinished"] else "")
        print(lines)
        counts = ", ".join(f"{self.counts[verdict]} {verdict}" for verdict in VERDICTS)
        tally = f"{self.passed} pass, {self.failed} fail"
        if known_without_case:
            tally += f", {len(known_without_case)} listed without a case"
        if self.stopped is not None:
            at = self.stopped["case"] or "a set it could not read"
            outcome = f"stopped at {at}; {self.unsent} cases not sent"
        else:
            outcome = "passed" if written["passed"] else "failed"
        print(f"{self.kind}: {cases} cases ({counts}); {tally}; {outcome}")
        if self.report:
            self._write_json(written)
        if self.junit:
            self._write_junit(known_without_case)
        return written

    def _testcase(self, result: dict) -> ET.Element:
        """One testcase. A verdict about the case (``regression``, ``rejected``, a listed case that matches) is a
        failure; one about the setup (``missing``, ``measurement_failed``) is an error, the test not having run; a
        listed known difference is skipped with its reason."""
        testcase = ET.Element(
            "testcase", classname=_classname(result["id"], self.kind, result["set"]), name=result["id"]
        )
        if result["passed"]:
            if result["known"] is not None:
                entry = result["known"]
                message = f"known {_outcome(entry)}: {entry['reason']} ({entry['issue']})"
                ET.SubElement(testcase, "skipped", message=message)
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
            for key, scratch in (("cases", self._cases), ("not_sent", self._not_sent)):
                out.write(f' "{key}": [')
                scratch.seek(0)
                for number, line in enumerate(scratch):
                    out.write(("\n  " if number == 0 else ",\n  ") + line.rstrip("\n"))
                out.write("\n ]" if key == "not_sent" else "\n ],\n")
            out.write("\n}\n")

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


def _classname(case_id: str, kind: str, set_name: str) -> str:
    return f"{case_id.split('/')[0]}/{kind}/{set_name}"


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
    entry = result["known"]
    if entry is not None:
        if result["passed"]:
            detail += f"; known: {entry['reason']} ({entry['issue']})"
        elif result["verdict"] == "match":
            detail += f", but it is listed as a known {_outcome(entry)} ({entry['reason']}); remove the entry"
        elif result["verdict"] in SETUP:
            words = result["verdict"].replace("_", " ")
            detail += f"; listed as a known {_outcome(entry)}, but {words} is about the setup"
        else:
            detail += f"; listed as a known {_outcome(entry)}, which this is not ({entry['issue']})"
    return detail


def _outcome(entry: dict) -> str:
    """What a known-difference entry states: its verdict, and the code of a refusal."""
    return f"{entry['verdict']} {entry['code']}" if entry["verdict"] == "rejected" else entry["verdict"]
