"""bellwether gaps: the parser, renderer and tokenizer coverage matrix for vLLM, SGLang and SMG.

Registries are read from source (a checkout, an offline copy, or files fetched at a pinned ref),
never by importing an engine. Names are normalized and merged through ``aliases.toml``; the
matrix is written as markdown or canonical JSON. See ``registries.py`` for what is read where.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fetch import fetch_registry_files
from .manifests import read_manifests
from .matrix import build_matrix, render_json, render_markdown
from .names import Aliases, default_aliases_path
from .registries import read_sglang, read_smg, read_vllm

USAGE_ERROR = 2


def _roots(src: Path | None, ref: str | None, engine: str, cache: Path) -> list[Path]:
    roots: list[Path] = []
    if src is not None:
        roots.append(src)
    if ref:
        roots.append(fetch_registry_files(engine, ref, cache))
    return roots


def run(args: argparse.Namespace) -> int:
    if args.vllm_src is None and not args.vllm_ref:
        print("bellwether gaps: give --vllm-src or --vllm-ref", file=sys.stderr)
        return USAGE_ERROR
    if args.sglang_src is None and not args.sglang_ref:
        print("bellwether gaps: give --sglang-src or --sglang-ref", file=sys.stderr)
        return USAGE_ERROR
    registries = {
        "vllm": read_vllm(_roots(args.vllm_src, args.vllm_ref, "vllm", args.cache)),
        "sglang": read_sglang(_roots(args.sglang_src, args.sglang_ref, "sglang", args.cache)),
        "smg": read_smg([args.smg_src]),
    }
    aliases = Aliases.load(args.aliases or default_aliases_path())
    manifests = read_manifests(args.fixtures) if args.fixtures.is_dir() else []
    matrix = build_matrix(registries, aliases, manifests)
    text = render_json(matrix) if args.format == "json" else render_markdown(matrix)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0
