"""bellwether record: run a corpus through one oracle and write or update fixtures.

The reference oracles for ``render`` (the checkpoint's template) and ``parse`` (the round trip through
that template) are implemented. Engine oracles and the other kinds exit with status 2 until their
milestone lands, so a script never mistakes a missing oracle for a recorded one.

A checkpoint group is recorded once, by its primary: a member's manifest makes ``record`` name the group to record
instead and exit 1. Before recording, the checkpoint's oracle inputs at the pinned revision are compared with the ones
its manifest lists, since the group's members were matched on that list; on a difference ``record`` names the files and
exits 1, recording nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bellwether import __version__
from bellwether.inputs import oracle_inputs
from bellwether.manifest import Manifest, find_manifest, load_manifest
from bellwether.storage import COMPRESSED_SUFFIX, plain_text, stem

from . import sets as set_tables
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
    try:
        manifest = find_manifest(args.fixtures, args.model)
    except (OSError, ValueError) as err:
        print(f"bellwether record: {err}", file=sys.stderr)
        return 1
    refusal = _refusal(manifest, args.fixtures)
    if refusal is not None:
        print(f"bellwether record: {refusal}", file=sys.stderr)
        return 1
    sets = load_corpus(args.corpus, args.kind, manifest.slug)
    if not sets:
        print(f"bellwether record: no corpus under {args.corpus / args.kind}", file=sys.stderr)
        return NOT_IMPLEMENTED
    in_corpus = set(sets)
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
    stop_ids: list[int] = []  # the stop id each parse output recorded in this run ends on
    kind_dir = args.fixtures / manifest.slug / args.kind
    # This run's sets.toml tables, by set; None drops the set's table. sets.toml is read when they are put in, at the
    # end of the run, so a run of the other kind for this model that wrote it in the meantime keeps its tables.
    tables: dict[str, dict | None] = {}
    for set_name, cases in sets.items():
        # A set imported from a public dataset is a benchmark set, stored compressed in Git LFS; the rest are plain.
        form = "zstd" if any(case.origin for case in cases) else "plain"
        out = kind_dir / f"{set_name}{COMPRESSED_SUFFIX if form == 'zstd' else '.jsonl'}"
        other = kind_dir / f"{set_name}{'.jsonl' if form == 'zstd' else COMPRESSED_SUFFIX}"
        previous = read_fixture_file(out) if out.is_file() else read_fixture_file(other) if other.is_file() else {}
        rejected = 0
        # The file is rebuilt from the cases rendered in this run: a case the corpus no longer has,
        # or that the oracle now rejects, leaves the file. Witnesses are carried over by id, but
        # only while the request they were recorded for is unchanged.
        lines: dict[str, dict] = {}
        witnesses_kept = witnesses_dropped = without_unicode_normalization = 0
        for case in cases:
            case_id = f"{manifest.slug}/{args.kind}/{case.name}"
            try:
                line, ids_without_unicode_normalization = _record(args.kind, oracle, case, provenance)
            except Exception as err:  # the reference cannot answer this case: report it, record nothing
                not_recorded.append((case_id, f"{type(err).__name__}: {err}"))
                rejected += 1
                continue
            without_unicode_normalization += ids_without_unicode_normalization
            line = {"id": case_id, "kind": args.kind, "model": manifest.model, **line}
            if isinstance(oracle, RoundtripOracle):
                stop_ids.append(line["reference"]["end_of_turn"]["stop_id"])
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
            generate = oracle.generate_stop() if isinstance(oracle, RoundtripOracle) else {}
            tables[set_name] = set_tables.entry(form, plain_text(out), len(lines), rejected, **generate)
        else:
            out.unlink(missing_ok=True)
            tables[set_name] = None
        # Only once the new file is written, so a failed write keeps the set in its old form.
        other.unlink(missing_ok=True)
        summary = [f"{len(lines)} cases recorded"]
        if without_unicode_normalization:
            # Reported here only: a count in sets.toml would change the format of every table in it.
            summary.append(
                f"{without_unicode_normalization} with output ids built without the tokenizer's Unicode normalization"
            )
        if witnesses_kept:
            summary.append(f"{witnesses_kept} with witnesses kept")
        if witnesses_dropped:
            summary.append(f"{witnesses_dropped} witnesses dropped because the request changed")
        if removed:
            summary.append(f"{removed} old cases removed")
        print(f"{out}: {', '.join(summary)}")
    # The fixture directory mirrors the corpus: a set file the corpus no longer has goes too.
    if kind_dir.is_dir() and not wanted:
        for stale in sorted(p for p in kind_dir.iterdir() if p.is_file()):
            name = stem(stale)
            if name is not None and name not in in_corpus:
                stale.unlink()
                tables[name] = None
                print(f"{stale}: removed, the corpus has no set of that name")
    set_tables.update(args.fixtures / manifest.slug / set_tables.FILE, args.kind, tables)
    differ = oracle.stop_sets_differ(manifest.model, stop_ids) if isinstance(oracle, RoundtripOracle) else None
    if differ is not None:
        print(differ, file=sys.stderr)
    for case_id, reason in not_recorded:
        print(f"not recorded {case_id}: {reason}", file=sys.stderr)
    return 1 if not_recorded else 0


def _refusal(manifest: Manifest, fixtures: Path) -> str | None:
    """Why the checkpoint is not recorded, or None when it may be."""
    if manifest.group is not None:
        # find_manifest has checked that the group names its primary
        primary = load_manifest(fixtures / manifest.group / "manifest.toml")
        return (
            f"{manifest.model} is a member of checkpoint group {manifest.group}, which is recorded once for all its "
            f"members; record the group instead: bellwether record --model {primary.model}"
        )
    if not manifest.inputs:
        return f"{manifest.path} lists no oracle inputs; `bellwether manifests` writes them"
    try:
        found = oracle_inputs(manifest.model, manifest.revision)
    except (OSError, ValueError) as err:
        return f"cannot read the oracle inputs of {manifest.model} at {manifest.revision}: {err}"
    listed = manifest.inputs
    differ = [
        f"{name} (listed {_short(listed.get(name))}, found {_short(found.get(name))})"
        for name in sorted(listed.keys() | found.keys())
        if listed.get(name) != found.get(name)
    ]
    if differ:
        return (
            f"refusing to record {manifest.model}: its oracle inputs at {manifest.revision} differ from "
            f"{manifest.path}: {', '.join(differ)}; the group was matched on the listed files, so run "
            "`bellwether manifests` again"
        )
    return None


def _short(digest: str | None) -> str:
    return digest[:12] if digest else "none"


def _record(kind: str, oracle: HfTemplateOracle | RoundtripOracle, case: Case, provenance: dict) -> tuple[dict, bool]:
    """The fields of one fixture line below id, kind and model, for the case's kind.

    Also whether the line's output ids leave out the tokenizer's Unicode normalization
    (``RoundtripOracle.encode_output``); a render line has no output ids.
    """
    if kind == "render":
        assert isinstance(oracle, HfTemplateOracle)
        rendered = oracle.render(case.request)
        line = {
            "request": case.request,
            "reference": {
                "source": RENDER_SOURCE,
                "input_ids": rendered.input_ids,
                "text": rendered.text,
                "provenance": provenance,
            },
        }
        return line, False
    assert isinstance(oracle, RoundtripOracle)
    if case.message is None:
        raise ValueError("a parse case needs `message`, the assistant message the output must parse to")
    output = oracle.render_output(case.request, case.message)
    line = {
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
            "end_of_turn": output.end_of_turn,
            "text": output.text,
            "provenance": provenance,
        },
    }
    return line, output.ids_without_unicode_normalization
