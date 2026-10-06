"""bellwether verify: replay fixtures through a running SMG and compare what it sends the engine.

Render cases are implemented: each request goes to SMG, and the mock worker behind it records the prompt
token ids SMG sent in its capture file. The other kinds exit with status 2 until their milestone lands,
so a script never mistakes a missing feature for a passing run.

A run reads every set once before its first request, to stop on anything it cannot read, keeping only the case ids.
It then sends, judges and writes one case at a time, so its memory does not grow with the number of cases. A failure
after the first request that leaves a case without a verdict (no answer, a capture line verify cannot read) stops
the sending; the cases answered so far are judged and reported, and the cases not sent are named.

Exit status: 0 when every case passes, 1 when one does not, and 2 when the run gives no verdict or stops before its
last case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

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
        sets, without_cases = check_sets(manifests, named=bool(args.models))
        if not sets:
            models = ", ".join(manifest.model for manifest in manifests)
            raise CannotVerify(f"no render fixtures under {args.fixtures} for {models}")
        known = report.load_known(args.known) if args.known else {}
        with (
            render.Capture(args.capture) as capture,
            render.client() as http,
            report.Writer(report=args.report, junit=args.junit) as writer,
        ):
            served = render.served_models(http, args.smg)
            for model in dict.fromkeys(manifest.model for manifest, _, _ in sets):
                if model not in served:
                    raise CannotVerify(f"SMG serves no model {model}; it serves {', '.join(sorted(served)) or 'none'}")
            listed = send(sets, http, args.smg, capture, known, writer)
            written = writer.finish(
                provenance=report.provenance(url=args.smg, capture=args.capture, known=args.known, manifests=manifests),
                known_without_case=report.known_without_case(known, manifests, listed),
                models_without_cases=without_cases,
            )
    except CannotVerify as err:
        print(f"bellwether verify: {err}", file=sys.stderr)
        return CANNOT_RUN
    if written["stopped"] is not None:
        print(f"bellwether verify: {written['stopped']['error']}", file=sys.stderr)
        return CANNOT_RUN
    return 0 if written["passed"] else 1


def send(
    sets: list[tuple[Manifest, str, Path]],
    http: httpx.Client,
    url: str,
    capture: render.Capture,
    known: dict[str, str],
    writer: report.Writer,
) -> set[str]:
    """Send, judge and write every case, one at a time, and return the listed ids among them.

    A failure that leaves a case without a verdict stops the sending. Each case after it is still read, to be named
    as not sent; a set that cannot be read again stops the run where it is, its error naming the set.
    """
    listed: set[str] = set()
    for manifest, set_name, path in sets:
        try:
            for _, case in read_cases(path):
                if case["id"] in known:
                    listed.add(case["id"])
                if writer.stopped is not None:
                    writer.not_sent(manifest.model, set_name, case["id"])
                    continue
                try:
                    result = render.verify_case(http, url, capture, manifest, set_name, case)
                except CannotVerify as err:
                    writer.stop(manifest.model, set_name, case["id"], str(err))
                    continue
                report.judge(result, known)
                writer.add(result)
        except CannotVerify as err:
            if writer.stopped is None:
                writer.stop(manifest.model, set_name, None, str(err))
    return listed


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


def check_sets(manifests: list[Manifest], *, named: bool) -> tuple[list[tuple[Manifest, str, Path]], list[str]]:
    """``(manifest, set, file)`` for each render set with a case, every one read whole; and the models with none.

    A set is read in either form, plain or compressed (see ``cases``), and only the case ids are kept. A case id is a
    model's once: the join and the known differences go by it. A model named with ``--model`` must have a case; any
    other model without one is named in the report.
    """
    sets, without = [], []
    for manifest in manifests:
        seen: dict[str, str] = {}  # case id -> its set
        for name, path in render_sets(manifest):
            before = len(seen)
            for number, case in read_cases(path):
                if case["id"] in seen:
                    raise CannotVerify(f"{path}:{number}: {case['id']} is already a case of the {seen[case['id']]} set")
                seen[case["id"]] = name
            if len(seen) > before:
                sets.append((manifest, name, path))
        if not seen:
            if named:
                raise CannotVerify(f"no render fixtures under {manifest.path.parent} for {manifest.model}")
            without.append(manifest.model)
    return sets, without
