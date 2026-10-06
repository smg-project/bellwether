"""bellwether import: corpus sets from public datasets at pinned revisions, never edited by hand."""

from __future__ import annotations

import argparse


def run(args: argparse.Namespace) -> int:
    if args.dataset == "bfcl":
        from .bfcl import run as run_bfcl

        return run_bfcl(args)
    if args.dataset == "gsm8k":
        from .gsm8k import run as run_gsm8k

        return run_gsm8k(args)
    if args.dataset == "mgsm":
        from .mgsm import run as run_mgsm

        return run_mgsm(args)
    if args.dataset == "shapes":
        from .shapes import run as run_shapes

        return run_shapes(args)
    if args.dataset == "hermes":
        from .hermes import run as run_hermes

        return run_hermes(args)
    raise ValueError(f"no importer for {args.dataset!r}")
