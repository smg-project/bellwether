"""Command line entry point: ``bellwether gaps | record | verify | report``.

``gaps`` (M1) and ``record --kind render --oracle reference`` (M2) are implemented. The other
subcommands are stubs until their milestone lands (see the milestone table in README.md); stubs exit
with status 2 so that scripts never mistake a missing feature for a passing run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bellwether import __version__
from bellwether.gaps import run as gaps_run
from bellwether.record import run as record_run

NOT_IMPLEMENTED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bellwether",
        description=(
            "Does SMG render the same prompt tokens and parse the same response as the model "
            "vendor's reference, vLLM and SGLang? Record references and witnesses, replay against "
            "SMG, classify every difference. No GPU."
        ),
    )
    parser.add_argument("--version", action="version", version=f"bellwether {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")

    gaps = sub.add_parser("gaps", help="parser, renderer and tokenizer coverage matrix: vLLM, SGLang, SMG")
    gaps.add_argument("--vllm-src", type=Path, help="a vLLM checkout, or a copy with the same layout")
    gaps.add_argument("--vllm-ref", help="fetch vLLM's registry files from GitHub at this commit or tag")
    gaps.add_argument("--sglang-src", type=Path, help="an SGLang checkout, or a copy with the same layout")
    gaps.add_argument("--sglang-ref", help="fetch SGLang's registry files from GitHub at this commit or tag")
    gaps.add_argument("--smg-src", type=Path, required=True, help="an smg checkout; the factories are read from source")
    gaps.add_argument("--aliases", type=Path, help="name mapping to use instead of the packaged aliases.toml")
    gaps.add_argument(
        "--fixtures", type=Path, default=Path("fixtures"), help="fixture root, to mark rows that have cases"
    )
    gaps.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "bellwether" / "registries")
    gaps.add_argument("--format", choices=["markdown", "json"], default="markdown")
    gaps.add_argument("--out", type=Path, help="write here instead of stdout")
    gaps.set_defaults(func=gaps_run)

    record = sub.add_parser("record", help="run the corpus through one oracle and write fixtures")
    record.add_argument("--model", required=True, help="Hugging Face model id, e.g. moonshotai/Kimi-K3")
    record.add_argument("--kind", required=True, choices=["render", "parse", "tokenize", "detokenize"])
    record.add_argument("--oracle", required=True, choices=["reference", "vllm", "sglang"])
    record.add_argument("--fixtures", type=Path, default=Path("fixtures"), help="fixture root holding the manifests")
    record.add_argument("--corpus", type=Path, default=Path("corpus"), help="corpus root: <kind>/<set>.jsonl")
    record.set_defaults(func=record_run)

    verify = sub.add_parser("verify", help="replay fixtures against SMG fronting the scripted mock engine")
    verify.add_argument("--smg", required=True, help="SMG base URL, e.g. http://127.0.0.1:30000")
    verify.add_argument("--fixtures", default="fixtures")
    verify.add_argument("--chunk-plan", action="append", default=None, help="restrict to named chunk plans")
    verify.add_argument("--junit", help="write a JUnit XML report here")
    verify.set_defaults(func=_stub("verify", "M3"))

    report = sub.add_parser("report", help="summarize verify reports as markdown for the artifacts repo")
    report.add_argument("reports", nargs="+")
    report.set_defaults(func=_stub("report", "M5"))
    return parser


def _stub(name: str, milestone: str):
    def run(_: argparse.Namespace) -> int:
        print(f"bellwether {name}: not implemented yet (lands in {milestone}).", file=sys.stderr)
        return NOT_IMPLEMENTED

    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return NOT_IMPLEMENTED
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
