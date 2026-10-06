"""bellwether unpack: one plain tree of every fixture set, in the layout of ``fixtures/``, for consumers.

Benchmark sets are stored compressed; a consumer (Symphony's fixture test, smg's consumer test) points at the
unpacked root instead, where every set is plain JSON Lines beside the hand-written ones, and each model's
``manifest.toml`` and ``sets.toml`` come along.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from bellwether.manifest import KINDS, load_manifest
from bellwether.record import sets as set_tables
from bellwether.record.fixtures import COMPRESSED_SUFFIX, plain_text


def unpack(fixtures: Path, out: Path, model: str | None = None) -> list[Path]:
    written: list[Path] = []
    for manifest_path in sorted(fixtures.glob("*/manifest.toml")):
        manifest = load_manifest(manifest_path)
        if model is not None and manifest.model != model:
            continue
        target_dir = out / manifest.slug
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.toml", set_tables.FILE):
            if (manifest_path.parent / name).is_file():
                shutil.copyfile(manifest_path.parent / name, target_dir / name)
        for kind in KINDS:
            kind_dir = manifest_path.parent / kind
            if not kind_dir.is_dir():
                continue
            for path in sorted(kind_dir.iterdir()):
                name = path.name.removesuffix(COMPRESSED_SUFFIX).removesuffix(".jsonl")
                if name == path.name:
                    continue
                target = target_dir / kind / f"{name}.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(plain_text(path).encode("utf-8"))
                written.append(target)
    return written


def run(args: argparse.Namespace) -> int:
    written = unpack(args.fixtures, args.out, args.model)
    print(f"{args.out}: {len(written)} plain fixture sets", file=sys.stdout)
    return 0
