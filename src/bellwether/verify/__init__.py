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
from bellwether.record.fixtures import read_fixture_file
from bellwether.unpack import set_files

from . import render, report
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
        cases = render_cases(manifests)
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
        results, without_case, url=args.smg, capture=args.capture, known=args.known, manifests=manifests
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


def render_cases(manifests: list[Manifest]) -> list[tuple[Manifest, str, dict]]:
    """Every render case of the given models, with its manifest and set name, in file order.

    A set is read in either form, plain or compressed. One that cannot be read, such as a set Git LFS has not
    fetched (its error names the command that fetches it), stops the run: passing over it would verify fewer
    cases than the fixtures hold and still pass.
    """
    cases = []
    for manifest in manifests:
        for kind, name, path in set_files(manifest):
            if kind != "render":
                continue
            try:
                found = read_fixture_file(path)
            except ValueError as err:
                raise CannotVerify(str(err)) from None
            cases += [(manifest, name, case) for case in found.values()]
    return cases
