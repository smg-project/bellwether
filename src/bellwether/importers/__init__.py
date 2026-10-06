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
    if args.dataset == "outlines":
        from .outlines import run as run_outlines

        return run_outlines(args)
    raise ValueError(f"no importer for {args.dataset!r}")
