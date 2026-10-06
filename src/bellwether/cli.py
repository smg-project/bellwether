"""Command line entry point: ``bellwether gaps | models | record | import | count | unpack | verify | report``.

``gaps`` (M1), ``models``, ``record --oracle reference`` for render and parse (M2, M4), ``import``, ``count``,
``unpack`` and ``verify --kind render`` against a running SMG (M3) are implemented. The other subcommands and
kinds are stubs until their milestone lands (see the milestone table in README.md); stubs exit with status 2
so that scripts never mistake a missing feature for a passing run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from bellwether import __version__
from bellwether.count import run as count_run
from bellwether.gaps import run as gaps_run
from bellwether.importers import run as import_run
from bellwether.manifest import KINDS
from bellwether.models import run as models_run
from bellwether.record import run as record_run
from bellwether.unpack import run as unpack_run
from bellwether.verify import run as verify_run

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

    models = sub.add_parser("models", help="the list of checkpoints to record: engine registries and the Hub")
    models.add_argument(
        "--out",
        type=Path,
        help="where the list goes, as JSON Lines; models.jsonl with --registry-only, else runs/models-<date>.jsonl",
    )
    models.add_argument(
        "--registry-only",
        action="store_true",
        help="list the registries' checkpoints without asking the Hugging Face Hub; offline once the files are cached",
    )
    models.add_argument(
        "--check",
        action="store_true",
        help="with --registry-only: compare a fresh list with the committed one, writing nothing; exit 1 on a change",
    )
    models.add_argument("--vllm-src", type=Path, help="a vLLM checkout to read the pinned registry from")
    models.add_argument("--sglang-src", type=Path, help="an SGLang checkout to read the pinned docs from")
    models.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "bellwether" / "registries")
    models.set_defaults(func=models_run)

    record = sub.add_parser("record", help="run the corpus through one oracle and write fixtures")
    record.add_argument("--model", required=True, help="Hugging Face model id, e.g. moonshotai/Kimi-K3")
    record.add_argument("--kind", required=True, choices=["render", "parse", "tokenize", "detokenize"])
    record.add_argument("--oracle", required=True, choices=["reference", "vllm", "sglang"])
    record.add_argument("--fixtures", type=Path, default=Path("fixtures"), help="fixture root holding the manifests")
    record.add_argument("--corpus", type=Path, default=Path("corpus"), help="corpus root: <kind>/<set>.jsonl")
    record.add_argument(
        "--set",
        dest="sets",
        action="append",
        metavar="NAME",
        help="record only this corpus set (repeat for more); other sets' fixture files are left as they are",
    )
    record.set_defaults(func=record_run)

    importer = sub.add_parser("import", help="write corpus sets from a public dataset at a pinned revision")
    importer.add_argument("dataset", choices=["bfcl", "gsm8k"], help="the dataset to import")
    importer.add_argument("--corpus", type=Path, default=Path("corpus"), help="corpus root: <kind>/<set>.jsonl")
    importer.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "bellwether" / "datasets",
        help="where pinned files are kept",
    )
    importer.add_argument(
        "--check",
        action="store_true",
        help="compare a fresh import with the corpus instead of writing it; exit 1 on a difference",
    )
    importer.set_defaults(func=import_run)

    count = sub.add_parser("count", help="cases per model, kind and source")
    count.add_argument("--fixtures", type=Path, default=Path("fixtures"), help="fixture root holding the manifests")
    count.add_argument("--corpus", type=Path, default=Path("corpus"), help="corpus root: <kind>/<set>.jsonl")
    count.add_argument("--format", choices=["markdown", "json"], default="markdown")
    count.set_defaults(func=count_run)

    unpack = sub.add_parser("unpack", help="write every fixture set as plain JSON Lines into one tree, for consumers")
    unpack.add_argument("--fixtures", type=Path, default=Path("fixtures"), help="fixture root holding the manifests")
    unpack.add_argument("--out", type=Path, default=Path("fixtures-plain"), help="where the plain tree goes")
    unpack.add_argument("--model", help="only this Hugging Face model id")
    unpack.set_defaults(func=unpack_run)

    verify = sub.add_parser(
        "verify",
        help="replay fixtures against SMG fronting the scripted mock engine",
        description=(
            "Send each render fixture's request to a running SMG, read the prompt token ids SMG sent from the "
            "capture file of the mock worker behind it, and compare them with the reference. Exit 0 when every "
            "case matches or is a listed known difference, 1 otherwise, 2 when the run gives no verdict."
        ),
    )
    verify.add_argument("--smg", required=True, metavar="URL", help="SMG base URL, e.g. http://127.0.0.1:30000")
    verify.add_argument(
        "--capture",
        type=Path,
        required=True,
        metavar="PATH",
        help="the file the mock worker behind SMG appends each request it receives to",
    )
    verify.add_argument(
        "--kind", choices=KINDS, default="render", help="fixtures to replay; only render is implemented (default)"
    )
    verify.add_argument(
        "--fixtures", type=Path, default=Path("fixtures"), metavar="DIR", help="fixture root holding the manifests"
    )
    verify.add_argument(
        "--model",
        dest="models",
        action="append",
        metavar="ID",
        help="verify only this model's fixtures (repeat for more); default: every manifest",
    )
    verify.add_argument(
        "--set",
        dest="sets",
        action="append",
        metavar="NAME",
        help="verify only this render set (repeat for more); default: every set of the selected models",
    )
    verify.add_argument(
        "--chunk-plan",
        action="append",
        default=None,
        metavar="NAME",
        help="restrict parse replays to named chunk plans",
    )
    verify.add_argument(
        "--known",
        type=Path,
        metavar="PATH",
        help="known differences: a TOML table per fixture id with the verdict SMG is known to give (regression, or "
        "rejected with SMG's error code), a reason and an issue link; a listed case passes while it has exactly "
        "that outcome and fails on any other",
    )
    verify.add_argument("--report", type=Path, metavar="PATH", help="write the JSON report here")
    verify.add_argument("--junit", type=Path, metavar="PATH", help="write a JUnit XML report here")
    verify.set_defaults(func=verify_run)

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
