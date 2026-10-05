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
    not_recorded: list[tuple[str, str]] = []
    for set_name, cases in sets.items():
        out = args.fixtures / manifest.slug / args.kind / f"{set_name}.jsonl"
        previous = read_fixture_file(out) if out.is_file() else {}
        # The file is rebuilt from the cases rendered in this run: a case the corpus no longer has,
        # or that the template now rejects, leaves the file. Witnesses are carried over by id, but
        # only while the request they were recorded for is unchanged.
        lines: dict[str, dict] = {}
        witnesses_kept = witnesses_dropped = 0
        for case in cases:
            case_id = f"{manifest.slug}/{args.kind}/{case.name}"
            try:
                rendered = oracle.render(case.request)
            except Exception as err:  # the reference cannot render this case: report it, record nothing
                not_recorded.append((case_id, f"{type(err).__name__}: {err}"))
                continue
            line = {
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
            old = previous.get(case_id)
            if old is not None and "witnesses" in old:
                if old.get("request") == case.request:
                    line["witnesses"] = old["witnesses"]
                    witnesses_kept += 1
                else:
                    witnesses_dropped += 1
            lines[case_id] = line
        removed = len(set(previous) - set(lines))
        if lines:
            write_fixture_file(out, lines)
        elif out.is_file():
            out.unlink()
        summary = [f"{len(lines)} cases recorded"]
        if witnesses_kept:
            summary.append(f"{witnesses_kept} with witnesses kept")
        if witnesses_dropped:
            summary.append(f"{witnesses_dropped} witnesses dropped because the request changed")
        if removed:
            summary.append(f"{removed} old cases removed")
        print(f"{out}: {', '.join(summary)}")
    # The fixture directory mirrors the corpus: a set file the corpus no longer has goes too.
    kind_dir = args.fixtures / manifest.slug / args.kind
    if kind_dir.is_dir():
        for stale in sorted(p for p in kind_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"):
            if stale.stem not in sets:
                stale.unlink()
                print(f"{stale}: removed, the corpus has no set of that name")
    for case_id, reason in not_recorded:
        print(f"not recorded {case_id}: {reason}", file=sys.stderr)
    return 1 if not_recorded else 0
