"""bellwether record: run a corpus through one oracle and write or update fixtures.

The reference oracle for ``render`` is implemented (M2). Engine oracles and the other kinds exit
with status 2 until their milestone lands, so a script never mistakes a missing oracle for a
recorded one.
"""

from __future__ import annotations

import argparse
import sys

from bellwether import __version__
from bellwether.manifest import find_manifest

from .corpus import load_corpus
from .fixtures import read_fixture_file, write_fixture_file
from .reference import SOURCE, HfTemplateOracle

NOT_IMPLEMENTED = 2


def run(args: argparse.Namespace) -> int:
    if args.kind != "render" or args.oracle != "reference":
        print(
            f"bellwether record: kind={args.kind} oracle={args.oracle} is not implemented yet "
            "(render/reference landed in M2; engine oracles and the other kinds follow).",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED
    manifest = find_manifest(args.fixtures, args.model)
    sets = load_corpus(args.corpus, args.kind, manifest.slug)
    if not sets:
        print(f"bellwether record: no corpus under {args.corpus / args.kind}", file=sys.stderr)
        return NOT_IMPLEMENTED
    oracle = HfTemplateOracle(manifest.model, manifest.revision)
    provenance = {**oracle.provenance(), "revision": manifest.revision, "bellwether": __version__}
    skipped: list[tuple[str, str]] = []
    for set_name, cases in sets.items():
        out = args.fixtures / manifest.slug / args.kind / f"{set_name}.jsonl"
        recorded = read_fixture_file(out) if out.is_file() else {}
        written = 0
        for case in cases:
            case_id = f"{manifest.slug}/{args.kind}/{case.name}"
            try:
                rendered = oracle.render(case.request)
            except Exception as err:  # the reference cannot render this case: report it, record nothing
                skipped.append((case_id, f"{type(err).__name__}: {err}"))
                continue
            line = recorded.get(case_id, {})
            line.update(
                {
                    "id": case_id,
                    "kind": args.kind,
                    "model": manifest.model,
                    "request": case.request,
                    "reference": {
                        "source": SOURCE,
                        "input_ids": rendered.input_ids,
                        "text": rendered.text,
                        "provenance": provenance,
                    },
                }
            )
            recorded[case_id] = line
            written += 1
        write_fixture_file(out, recorded)
        print(f"{out}: {written} cases recorded, {len(recorded)} in file")
    for case_id, reason in skipped:
        print(f"not recorded {case_id}: {reason}", file=sys.stderr)
    return 1 if skipped else 0
