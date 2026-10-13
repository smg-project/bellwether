"""``bellwether multimodal``: the image parity suite under ``multimodal/``.

Does SMG's media pipeline turn an image into the same pixels the serving engine's own image path gives, bit for bit:
the decoded RGB bytes, the u8 patches, the bf16 and f32 tensors the engine receives? The suite holds a public-benchmark
corpus of images in Git LFS, one reference set per processor family recorded inside the engine's image on CPU, a
cross-check from transformers' processor, and the commands that keep them consistent and compare SMG with them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bellwether.multimodal import check, count, crosscheck, fetch, references, suite, verify


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "multimodal", help="the image parity suite: check, count, fetch, verify, cross-check, import-references"
    )
    parser.set_defaults(func=lambda args: parser.print_help() or 2)
    commands = parser.add_subparsers(dest="multimodal_command", metavar="command")

    def root_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=suite.ROOT, help="the suite's directory (default: multimodal)")

    checker = commands.add_parser(
        "check", help="the manifest, tiers, sources, families and reference tables agree with the files; no fetch"
    )
    root_arg(checker)
    checker.set_defaults(func=check.run)

    counter = commands.add_parser("count", help="images per dataset and class, tiers, recorded references")
    root_arg(counter)
    counter.add_argument("--readme", type=Path, help="write the tables into this file, between its markers")
    counter.add_argument("--check", action="store_true", help="with --readme: compare the file's tables, write nothing")
    counter.set_defaults(func=count.run)

    fetcher = commands.add_parser("fetch", help="git lfs pull of one tier's images, raw arrays and reference sets")
    root_arg(fetcher)
    fetcher.add_argument("--tier", required=True, choices=suite.TIERS)
    fetcher.add_argument(
        "--family", dest="families", action="append", metavar="SLUG", help="only this family's sets (repeat for more)"
    )
    fetcher.add_argument("--no-raw", action="store_true", help="leave the raw arrays as pointers")
    fetcher.add_argument("--dry-run", action="store_true", help="print the git lfs pull command instead of running it")
    fetcher.set_defaults(func=fetch.run)

    verifier = commands.add_parser("verify", help="compare an SMG dump of one tier with each family's reference")
    root_arg(verifier)
    verifier.add_argument("--tier", required=True, choices=suite.TIERS)
    verifier.add_argument(
        "--smg-dump",
        metavar="BIN",
        help="the SMG dump binary (multimodal/tools/smg_dump.rs), run per family over the tier",
    )
    verifier.add_argument(
        "--dump",
        dest="dumps",
        action="append",
        metavar="FAMILY=PATH",
        help="a dump already made for a family (repeat for more)",
    )
    verifier.add_argument(
        "--family", dest="families", action="append", metavar="SLUG", help="only this family (repeat for more)"
    )
    verifier.add_argument(
        "--source",
        choices=suite.SOURCES,
        default="vllm",
        help="the reference to compare with (default: the engine's own path)",
    )
    verifier.add_argument("--known", type=Path, help="known differences (default: <root>/known.toml when it exists)")
    verifier.add_argument("--jobs", type=int, default=1, help="dump processes per family")
    verifier.add_argument("--work", type=Path, help="keep the dumps and logs here instead of a temporary directory")
    verifier.add_argument("--show", type=int, default=20, help="differences printed per family")
    verifier.add_argument("--strict-missing", action="store_true", help="exit 2 when an image got no verdict")
    verifier.add_argument("--report", type=Path, help="write the JSON report here")
    verifier.add_argument("--junit", type=Path, help="write a JUnit XML report here")
    verifier.set_defaults(func=verify.run)

    crosser = commands.add_parser("cross-check", help="the engine's own path against the cross-check, per family")
    root_arg(crosser)
    crosser.add_argument("--family", dest="families", action="append", metavar="SLUG")
    crosser.add_argument("--show", type=int, default=20)
    crosser.add_argument("--strict", action="store_true", help="exit 1 when the two differ on any image")
    crosser.add_argument("--report", type=Path)
    crosser.set_defaults(func=crosscheck.run)

    importer = commands.add_parser(
        "import-references", help="a recorder's JSON Lines -> the family's canonical set and its sets.toml table"
    )
    root_arg(importer)
    importer.add_argument("--family", required=True, metavar="SLUG")
    importer.add_argument("--source", required=True, choices=suite.SOURCES)
    importer.add_argument("--from", dest="recorder_output", required=True, type=Path, metavar="JSONL")
    importer.add_argument(
        "--raw-from", type=Path, metavar="DIR", help="the recorder's raw arrays; those of --raw-tier are copied"
    )
    importer.add_argument("--raw-tier", default="corpus-50", choices=suite.TIERS)
    importer.set_defaults(func=_import)


def _import(args: argparse.Namespace) -> int:
    import sys

    try:
        table = references.import_references(
            args.root, args.family, args.source, args.recorder_output, args.raw_from, args.raw_tier
        )
    except (OSError, ValueError) as err:
        print(f"bellwether multimodal import-references: {err}", file=sys.stderr)
        return 1
    print(
        f"{args.family}/{args.source}: {table['cases']} cases, {table['errors']} errors, "
        f"{table['plain_bytes']} plain bytes, {table['stored_bytes']} stored"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    return args.func(args)
