"""What a verify run concludes: which cases pass given the known differences, as JSON, JUnit XML and text."""

from __future__ import annotations

import json
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

from bellwether import __version__
from bellwether.manifest import Manifest

from .render import CannotVerify

VERDICTS = ("match", "regression", "rejected", "missing")
EXCUSABLE = ("regression", "rejected")
LEADING_KEYS = ("id", "model", "set", "verdict", "passed", "known")
WITHOUT_CASE = "listed, but there is no such case; remove the entry"


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


def judge(results: list[dict], known: dict[str, str]) -> None:
    """Mark each case passed or not.

    A case passes when it matches and is not listed, or when it is a regression or rejected and is listed. A listed
    case that matches fails, so an entry goes as soon as SMG is fixed and the list cannot rot. A missing capture
    line fails even when listed: it is about the setup, which file is read and whether the ``rid`` reached the
    engine, not about how SMG renders.
    """
    for result in results:
        reason = known.get(result["id"])
        result["known"] = reason
        if result["verdict"] == "match":
            result["passed"] = reason is None
        else:
            result["passed"] = reason is not None and result["verdict"] in EXCUSABLE


def known_without_case(known: dict[str, str], manifests: list[Manifest], results: list[dict]) -> list[str]:
    """Listed ids in a verified model's render fixtures that name no case: the corpus dropped or renamed them.

    Entries for other models or kinds are left alone; this run says nothing about them.
    """
    namespaces = tuple(f"{manifest.slug}/render/" for manifest in manifests)
    ids = {result["id"] for result in results}
    return sorted(case_id for case_id in known if case_id.startswith(namespaces) and case_id not in ids)


def build(
    results: list[dict],
    without_case: list[str],
    *,
    url: str,
    capture: Path,
    known: Path | None,
    manifests: list[Manifest],
) -> dict:
    passed = sum(result["passed"] for result in results)
    summary = {"cases": len(results), **{verdict: 0 for verdict in VERDICTS}}
    for result in results:
        summary[result["verdict"]] += 1
    provenance = {
        "bellwether": bellwether_version(),
        "smg": url,
        "capture": str(capture),
        "known": None if known is None else str(known),
        "manifests": [{"model": m.model, "revision": m.revision, "path": str(m.path)} for m in manifests],
    }
    return {
        "kind": "render",
        "provenance": provenance,
        "summary": {**summary, "passed": passed, "failed": len(results) - passed},
        "passed": passed == len(results) and not without_case,
        "known_without_case": without_case,
        # The verdict and whether it passes first; the response body, which can be long, last.
        "cases": [
            {**{k: r[k] for k in LEADING_KEYS}, **{k: r[k] for k in r if k not in LEADING_KEYS}} for r in results
        ],
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


def write_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict) -> None:
    """One testcase per case, one suite per model.

    ``regression`` and ``missing`` are failures; ``rejected`` is an error, SMG's own error answer; a listed known
    difference is skipped with its reason. Known entries that name no case fail in a suite of their own.
    """
    kind = report["kind"]
    root = ET.Element("testsuites", name=f"bellwether verify {kind}")
    suites: dict[str, ET.Element] = {}
    for case in report["cases"]:
        name = f"{case['model']} {kind}"
        suite = suites.get(name)
        if suite is None:
            suite = suites[name] = ET.SubElement(root, "testsuite", name=name)
        classname = f"{case['id'].split('/')[0]}/{kind}/{case['set']}"
        testcase = ET.SubElement(suite, "testcase", classname=classname, name=case["id"])
        if case["passed"]:
            if case["known"] is not None:
                ET.SubElement(testcase, "skipped", message=f"known difference: {case['known']}")
            continue
        tag = "error" if case["verdict"] == "rejected" else "failure"
        failure_type = "known-but-matches" if case["verdict"] == "match" else case["verdict"]
        element = ET.SubElement(testcase, tag, type=failure_type, message=describe(case))
        if case["verdict"] == "regression":
            details = ("index", "lengths", "window", "text_equal", "text")
            element.text = json.dumps({key: case[key] for key in details if key in case}, ensure_ascii=False)
    if report["known_without_case"]:
        suite = ET.SubElement(root, "testsuite", name="known differences")
        for case_id in report["known_without_case"]:
            testcase = ET.SubElement(suite, "testcase", classname="known differences", name=case_id)
            ET.SubElement(testcase, "failure", type="known-without-case", message=WITHOUT_CASE)
    for element in [*root, root]:
        testcases = list(element.iter("testcase"))
        element.set("tests", str(len(testcases)))
        for tag, attribute in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            element.set(attribute, str(sum(testcase.find(tag) is not None for testcase in testcases)))
    tree = ET.ElementTree(root)
    ET.indent(tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def lines(report: dict) -> list[str]:
    """One line per case to look at, then the totals."""
    out = [
        f"{case['verdict']} {case['id']}: {describe(case)}"
        for case in report["cases"]
        if not (case["verdict"] == "match" and case["passed"])
    ]
    out += [f"known {case_id}: {WITHOUT_CASE}" for case_id in report["known_without_case"]]
    summary = report["summary"]
    counts = ", ".join(f"{summary[verdict]} {verdict}" for verdict in VERDICTS)
    tally = f"{summary['passed']} pass, {summary['failed']} fail"
    if report["known_without_case"]:
        tally += f", {len(report['known_without_case'])} listed without a case"
    outcome = "passed" if report["passed"] else "failed"
    out.append(f"{report['kind']}: {summary['cases']} cases ({counts}); {tally}; {outcome}")
    return out


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
        detail = f"HTTP {result['status']}: {result['message']}"
    elif result["verdict"] == "missing":
        detail = "SMG answered, but no capture line carries this case's id"
    elif result["known"] is not None:
        detail = "matches"
    if result["known"] is not None:
        if result["passed"]:
            detail += f"; known: {result['known']}"
        elif result["verdict"] == "match":
            detail += f", but it is listed as a known difference ({result['known']}); remove the entry"
        else:
            detail += "; listed as a known difference, but a missing line is not one"
    return detail
