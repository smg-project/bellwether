"""bellwether record: run a corpus through one oracle and write or update fixtures.

The reference oracles for ``render`` (the checkpoint's template) and ``parse`` (the round trip through
that template) are implemented. Engine oracles and the other kinds exit with status 2 until their
milestone lands, so a script never mistakes a missing oracle for a recorded one.
"""

from __future__ import annotations

import argparse
import sys

from bellwether import __version__
from bellwether.manifest import find_manifest

from .chunks import chunk_plans
from .corpus import Case, load_corpus
from .fixtures import read_fixture_file, write_fixture_file
from .reference import SOURCE as RENDER_SOURCE
from .reference import HfTemplateOracle
from .roundtrip import SOURCE as PARSE_SOURCE
from .roundtrip import RoundtripOracle

NOT_IMPLEMENTED = 2


def run(args: argparse.Namespace) -> int:
    if args.oracle != "reference" or args.kind not in ("render", "parse"):
        print(
            f"bellwether record: kind={args.kind} oracle={args.oracle} is not implemented yet "
            "(render and parse with the reference oracle landed in M2 and M4; engine oracles and the other "
            "kinds follow).",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED
    manifest = find_manifest(args.fixtures, args.model)
    sets = load_corpus(args.corpus, args.kind, manifest.slug)
    if not sets:
        print(f"bellwether record: no corpus under {args.corpus / args.kind}", file=sys.stderr)
        return NOT_IMPLEMENTED
    wanted = list(getattr(args, "sets", None) or [])
    unknown = sorted(set(wanted) - set(sets))
    if unknown:
        print(
            f"bellwether record: no corpus set named {', '.join(unknown)} under {args.corpus / args.kind}",
            file=sys.stderr,
        )
        return 1
    if wanted:
        sets = {name: cases for name, cases in sets.items() if name in wanted}
    oracle = (
        HfTemplateOracle(manifest.model, manifest.revision)
        if args.kind == "render"
        else RoundtripOracle(manifest.model, manifest.revision)
    )
    provenance = {**oracle.provenance(), "revision": manifest.revision, "bellwether": __version__}
    not_recorded: list[tuple[str, str]] = []
    for set_name, cases in sets.items():
        out = args.fixtures / manifest.slug / args.kind / f"{set_name}.jsonl"
        previous = read_fixture_file(out) if out.is_file() else {}
        # The file is rebuilt from the cases rendered in this run: a case the corpus no longer has,
        # or that the oracle now rejects, leaves the file. Witnesses are carried over by id, but
        # only while the request they were recorded for is unchanged.
        lines: dict[str, dict] = {}
        witnesses_kept = witnesses_dropped = 0
        for case in cases:
            case_id = f"{manifest.slug}/{args.kind}/{case.name}"
            try:
                line = _record(args.kind, oracle, case, provenance)
            except Exception as err:  # the reference cannot answer this case: report it, record nothing
                not_recorded.append((case_id, f"{type(err).__name__}: {err}"))
                continue
            line = {"id": case_id, "kind": args.kind, "model": manifest.model, **line}
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
    if kind_dir.is_dir() and not wanted:
        for stale in sorted(p for p in kind_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"):
            if stale.stem not in sets:
                stale.unlink()
                print(f"{stale}: removed, the corpus has no set of that name")
    for case_id, reason in not_recorded:
        print(f"not recorded {case_id}: {reason}", file=sys.stderr)
    return 1 if not_recorded else 0


def _record(kind: str, oracle: HfTemplateOracle | RoundtripOracle, case: Case, provenance: dict) -> dict:
    """The fields of one fixture line below id, kind and model, for the case's kind."""
    if kind == "render":
        assert isinstance(oracle, HfTemplateOracle)
        rendered = oracle.render(case.request)
        return {
            "request": case.request,
            "reference": {
                "source": RENDER_SOURCE,
                "input_ids": rendered.input_ids,
                "text": rendered.text,
                "provenance": provenance,
            },
        }
    assert isinstance(oracle, RoundtripOracle)
    if case.message is None:
        raise ValueError("a parse case needs `message`, the assistant message the output must parse to")
    output = oracle.render_output(case.request, case.message)
    return {
        "request": case.request,
        "tools": list(case.request.get("tools") or []),
        "output_ids": output.output_ids,
        "output_pieces": output.output_pieces,
        "malformed": False,
        "chunk_plans": chunk_plans(len(output.output_ids)),
        "reference": {
            "source": PARSE_SOURCE,
            "message": {"role": "assistant", **case.message},
            "finish_reason": output.finish_reason,
            "text": output.text,
            "provenance": provenance,
        },
    }
