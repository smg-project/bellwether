"""Reference sets: what the engine's own image path gives for every image of a family, one JSON line per image, and the
cross-check beside it.

A recorder (``multimodal/tools/record_vllm.py`` inside the engine image, ``multimodal/tools/oracle_hf.py`` in a torch
environment) writes one JSON line per image with its own key names and timings; ``import_references`` turns such a file
into the suite's canonical set: sorted keys, one line per image in file-name order, the path from the suite root, the
hashes under their ``_sha256`` names, no timings, stored as ``references/<family>/<source>.jsonl.zst`` in Git LFS with
its table in ``sets.toml`` (form, cases, errors, plain size and sha256, the stored object's sha256 and size, and the
recorder's provenance), so a reviewer reads the table without fetching the set and ``check`` compares the pointer.

A line is ``{"name", "path", "w", "h", "mode", "transparent", "grid", "shape", "rgb_sha256", "u8_sha256",
"bf16_sha256", "f32_sha256"}``: the decoded size, the file's mode as opened, whether it had transparency (null where the
recorder did not say), the image grid as the processor reports it (flattened), the pixel tensor's shape, and the sha256
of the loader's RGB bytes, of the un-normalized u8 patches (null where the path exposes none), of the f32 pixel values
and of their bf16 round-to-nearest-even bits. An image the recorder could not process is ``{"name", "path", "error"}``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from bellwether import jsonl, storage
from bellwether.multimodal import suite

FIELDS = (
    "name",
    "path",
    "w",
    "h",
    "mode",
    "transparent",
    "grid",
    "shape",
    "rgb_sha256",
    "u8_sha256",
    "bf16_sha256",
    "f32_sha256",
)
HASHES = ("rgb_sha256", "u8_sha256", "bf16_sha256", "f32_sha256")
RAW_KINDS = ("rgb", "u8", "norm")
# What the two recorders and the SMG dump call each hash; the canonical name is the first.
ALIASES = {
    "rgb_sha256": ("rgb_sha256", "rgb_sha"),
    "u8_sha256": ("u8_sha256", "u8_sha"),
    "bf16_sha256": ("bf16_sha256", "bf16_sha"),
    "f32_sha256": ("f32_sha256", "f32_sha"),
}


def set_path(root: Path, family: str, source: str) -> Path:
    return root / "references" / family / f"{source}.jsonl.zst"


def raw_dir(root: Path, family: str) -> Path:
    return root / "references" / family / "raw"


def hashes_of(record: dict) -> dict[str, str | None]:
    """The four hashes of a recorder's or a dump's line under their canonical names; a hash the line lacks is None."""
    found: dict[str, str | None] = {}
    for canonical, names in ALIASES.items():
        value = next((record[name] for name in names if record.get(name) is not None), None)
        found[canonical] = str(value) if value is not None else None
    return found


def canonical(record: dict, image: suite.Image) -> dict:
    """A recorder's line as the set stores it; the manifest's image supplies the path."""
    if "error" in record:
        return {"error": str(record["error"]), "name": image.name, "path": image.path}
    grid = record.get("grid")
    if isinstance(grid, list):
        flat: list[int] = []
        stack = [grid]
        while stack:
            item = stack.pop(0)
            if isinstance(item, list):
                stack[0:0] = item
            else:
                flat.append(int(item))
        grid = flat
    line = {
        "name": image.name,
        "path": image.path,
        "w": int(record["w"]),
        "h": int(record["h"]),
        "mode": record.get("mode"),
        "transparent": record.get("transparent"),
        "grid": grid,
        "shape": [int(x) for x in record["shape"]],
        **hashes_of(record),
    }
    return {key: line[key] for key in sorted(line)}


def read_recorder_output(path: Path) -> tuple[dict | None, dict[str, dict]]:
    """A recorder's JSON Lines: the ``meta`` line when it has one, and the records by file name (the last wins)."""
    meta: dict | None = None
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="\n") as handle:
        for _, record in jsonl.load(handle, path):
            if not isinstance(record, dict):
                raise ValueError(f"{path}: a line is not a JSON object")
            if "meta" in record and "name" not in record:
                meta = record["meta"]
                continue
            records[str(record["name"])] = record
    return meta, records


def import_references(
    root: Path,
    family: str,
    source: str,
    recorder_output: Path,
    raw_from: Path | None = None,
    raw_tier: str = "corpus-50",
) -> dict:
    """Write ``references/<family>/<source>.jsonl.zst`` from a recorder's output and its table in ``sets.toml``;
    with ``raw_from``, copy the raw arrays of the images of ``raw_tier`` beside it. Returns the table written."""
    if source not in suite.SOURCES:
        raise ValueError(f"unknown reference source {source!r}; one of {', '.join(suite.SOURCES)}")
    families = suite.read_families(root)
    if family not in families:
        raise ValueError(f"no family {family!r} under {root / 'families'}")
    images = suite.by_name(suite.read_manifest(root))
    meta, records = read_recorder_output(recorder_output)
    lines: list[str] = []
    errors = 0
    unknown = sorted(name for name in records if name not in images)
    if unknown:
        raise ValueError(f"{recorder_output} names {len(unknown)} files the manifest does not list, first {unknown[0]}")
    for name in sorted(records):
        line = canonical(records[name], images[name])
        errors += "error" in line
        lines.append(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    plain = "".join(lines).encode("utf-8")
    target = set_path(root, family, source)
    storage.write(target, plain)
    stored = target.read_bytes()
    table = {
        "form": "zstd",
        "cases": len(lines) - errors,
        "errors": errors,
        "plain_bytes": len(plain),
        "plain_sha256": hashlib.sha256(plain).hexdigest(),
        "stored_bytes": len(stored),
        "stored_sha256": hashlib.sha256(stored).hexdigest(),
    }
    if meta:
        table["provenance"] = {k: v for k, v in sorted(meta.items()) if isinstance(v, str | int | float | bool)}
    if raw_from is not None:
        copied = import_raw(root, family, raw_from, suite.tier_images(root, raw_tier, list(images.values())))
        table["raw_tier"] = raw_tier
        table["raw_files"] = copied
    tables = suite.read_sets(root / "references" / family / suite.SETS_FILE)
    tables[(source, family)] = table
    suite.write_sets(root / "references" / family / suite.SETS_FILE, tables)
    return table


def import_raw(root: Path, family: str, raw_from: Path, images: list[suite.Image]) -> int:
    """Copy ``<name>.{rgb,u8,norm}.npy`` of each image from a recorder's raw directory; the kinds it lacks are skipped
    (the engine path exposes no u8). Returns how many files were copied."""
    target = raw_dir(root, family)
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image in images:
        for kind in RAW_KINDS:
            source = raw_from / f"{image.name}.{kind}.npy"
            if source.is_file():
                shutil.copyfile(source, target / source.name)
                copied += 1
    return copied


def read_references(root: Path, family: str, source: str) -> dict[str, dict]:
    """The set's lines by file name. A set Git LFS has not fetched is refused with the command that fetches it."""
    path = set_path(root, family, source)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: no {source} reference recorded for {family}")
    found: dict[str, dict] = {}
    for _, record in jsonl.loads(storage.plain_text(path), path):
        found[str(record["name"])] = record
    return found
