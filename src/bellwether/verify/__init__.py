"""bellwether verify: replay fixtures through a running SMG and compare what it sends the engine.

Render cases are implemented: each request goes to SMG, and the mock worker behind it records the prompt
token ids SMG sent in its capture file. The other kinds exit with status 2 until their milestone lands,
so a script never mistakes a missing feature for a passing run.

Exit status: 0 when every case passes, 1 when one does not, and 2 when the run gives no verdict at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bellwether.manifest import Manifest, find_manifest, load_manifest

from . import render, report
from .cases import read_cases, render_sets
from .render import CannotVerify

NOT_IMPLEMENTED = 2
CANNOT_RUN = 2  # the stub's status too: no verdict was given, so no script may read the run as passed


def run(args: argparse.Namespace) -> int:
    if args.kind != "render":
        print(
            f"bellwether verify: --kind {args.kind} is not implemented yet (render cases are; parse replay follows).",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED
    if args.chunk_plan:
        print("bellwether verify: --chunk-plan selects parse replays; render cases have none", file=sys.stderr)
        return CANNOT_RUN
    try:
        manifests = select_manifests(args.fixtures, args.models)
        cases, without_cases = render_cases(manifests, named=bool(args.models))
        if not cases:
            models = ", ".join(manifest.model for manifest in manifests)
            raise CannotVerify(f"no render fixtures under {args.fixtures} for {models}")
        known = report.load_known(args.known) if args.known else {}
        results = render.verify(args.smg, args.capture, cases)
    except CannotVerify as err:
        print(f"bellwether verify: {err}", file=sys.stderr)
        return CANNOT_RUN
    report.judge(results, known)
    without_case = report.known_without_case(known, manifests, results)
    written = report.build(
        results,
        without_case,
        url=args.smg,
        capture=args.capture,
        known=args.known,
        manifests=manifests,
        without_cases=without_cases,
    )
    for line in report.lines(written):
        print(line)
    if args.report:
        report.write_json(args.report, written)
    if args.junit:
        report.write_junit(args.junit, written)
    return 0 if written["passed"] else 1


def select_manifests(fixtures: Path, models: list[str] | None) -> list[Manifest]:
    """The manifests of the named models, or every manifest under ``fixtures`` when none is named.

    A manifest that cannot be read or is not valid stops the run, as a model without one does: no verdict is given.
    """
    try:
        if models:
            return [find_manifest(fixtures, model) for model in dict.fromkeys(models)]
        manifests = [load_manifest(path) for path in sorted(fixtures.glob("*/manifest.toml"))]
    except (OSError, ValueError) as err:
        raise CannotVerify(str(err)) from None
    if not manifests:
        raise CannotVerify(f"no manifests under {fixtures}")
    return manifests


def render_cases(manifests: list[Manifest], *, named: bool) -> tuple[list[tuple[Manifest, str, dict]], list[str]]:
    """Every render case of the given models, with its manifest and set name, in file order; and the models with none.

    A set is read in either form, plain or compressed (see ``cases``). A case id is a model's once: the join and the
    known differences go by it. A model named with ``--model`` must have a case; any other model without one is
    named in the report.
    """
    cases, without = [], []
    for manifest in manifests:
        seen: dict[str, str] = {}  # case id -> its set
        for name, path in render_sets(manifest):
            for number, case in read_cases(path):
                if case["id"] in seen:
                    raise CannotVerify(f"{path}:{number}: {case['id']} is already a case of the {seen[case['id']]} set")
                seen[case["id"]] = name
                cases.append((manifest, name, case))
        if not seen:
            if named:
                raise CannotVerify(f"no render fixtures under {manifest.path.parent} for {manifest.model}")
            without.append(manifest.model)
    return cases, without
