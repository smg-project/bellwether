"""``bellwether multimodal verify``: SMG's media pipeline against the engine's own image path, image by image.

The SMG side is a dump: one JSON line per image with the decoded size, the grid, the pixel tensor's shape and the sha256
of the decoded RGB bytes, of the u8 patches, of the bf16 wire bytes and of the f32 wire bytes, the protocol
``multimodal/tools/smg_dump.rs`` implements (an example in smg's multimodal crate, driven by ``SMG_DUMP_LIST``,
``SMG_DUMP_OUT``, ``SMG_DUMP_PROCESSOR`` and ``SMG_DUMP_CONFIG``). ``--smg-dump BIN`` runs that binary over the tier
for each family; ``--dump FAMILY=PATH`` reads a dump already made. Each image is compared with the family's reference
(``--source``, the engine's own path by default): the decoded RGB bytes, the patch shape, the u8 patches where the
reference recorded them, the bf16 and the f32 wire bytes. A difference on an image a known difference covers
(``multimodal/known.toml``) is counted apart and does not fail the run.

The verdicts are per image and family: ``match``, ``difference`` (which artifacts differ, in the order rgb, shape, u8,
bf16, f32), ``known``, ``error`` (one side could not process the image), ``missing`` (the dump has no line for it).
The report (``--report``, JSON) carries the per-class table and every difference; ``--junit`` writes one test case per
family and class. Exit 0 when every image matches or is known, 1 when one differs, 2 when the run gives no verdict (no
reference set, a dump that cannot be read or made, an empty tier).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from bellwether.multimodal import references, suite

ARTIFACTS = ("rgb", "shape", "u8", "bf16", "f32")
NO_VERDICT = 2


def read_known(path: Path | None) -> list[dict]:
    """The known differences: ``[[known]]`` tables with ``family``, a ``reason``, an ``issue`` and the images they
    cover, ``names`` or ``short_side_under`` (every image whose shorter decoded side is under that many pixels)."""
    if path is None or not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("known", [])
    for entry in entries:
        for key in ("family", "reason"):
            if not isinstance(entry.get(key), str):
                raise ValueError(f"{path}: every [[known]] table has a `{key}` string")
        if "names" not in entry and "short_side_under" not in entry:
            raise ValueError(f"{path}: a [[known]] table names its images (`names`) or a `short_side_under` size")
    return entries


def known_for(entries: list[dict], family: str, reference: dict) -> dict | None:
    for entry in entries:
        if entry["family"] != family:
            continue
        if reference["name"] in entry.get("names", []):
            return entry
        limit = entry.get("short_side_under")
        if limit is not None and min(int(reference["w"]), int(reference["h"])) < int(limit):
            return entry
    return None


def compare(reference: dict, dumped: dict) -> list[str]:
    """The artifacts that differ between a reference line and a dump line, in ``ARTIFACTS`` order; empty on a match."""
    differing: list[str] = []
    ref_hashes = references.hashes_of(reference)
    dump_hashes = references.hashes_of(dumped)
    same_size = (int(reference["w"]), int(reference["h"])) == (int(dumped["w"]), int(dumped["h"]))
    if not same_size or ref_hashes["rgb_sha256"] != dump_hashes["rgb_sha256"]:
        differing.append("rgb")
    same_shape = [int(x) for x in reference["shape"]] == [int(x) for x in dumped["shape"]]
    if not same_shape:
        differing.append("shape")
    for artifact in ("u8", "bf16", "f32"):
        expected = ref_hashes[f"{artifact}_sha256"]
        if expected is None:
            continue  # the reference path exposes no such artifact
        if not same_shape or expected != dump_hashes[f"{artifact}_sha256"]:
            differing.append(artifact)
    return differing


def judge(
    family: str, images: list[suite.Image], refs: dict[str, dict], dump: dict[str, dict], known: list[dict]
) -> dict:
    """The family's verdicts over the tier's images: per class counts, every difference, the known ones apart."""
    classes: dict[str, Counter] = {}
    differences: list[dict] = []
    known_hits: list[dict] = []
    errors: list[dict] = []
    missing: list[str] = []
    for image in images:
        counter = classes.setdefault(image.cls, Counter())
        counter["images"] += 1
        reference = refs.get(image.name)
        dumped = dump.get(image.name)
        if reference is None:
            counter["no_reference"] += 1
            missing.append(image.name)
            continue
        if dumped is None:
            counter["missing"] += 1
            missing.append(image.name)
            continue
        if "error" in reference or "error" in dumped:
            counter["error"] += 1
            errors.append(
                {
                    "name": image.name,
                    "class": image.cls,
                    "reference": reference.get("error"),
                    "smg": dumped.get("error"),
                }
            )
            continue
        differing = compare(reference, dumped)
        if not differing:
            counter["match"] += 1
            continue
        entry = known_for(known, family, reference)
        record = {
            "name": image.name,
            "path": image.path,
            "class": image.cls,
            "differs": differing,
            "reference": {k: reference.get(k) for k in ("w", "h", "grid", "shape")},
            "smg": {k: dumped.get(k) for k in ("w", "h", "grid", "shape")},
        }
        if entry is not None:
            counter["known"] += 1
            record["reason"] = entry["reason"]
            record["issue"] = entry.get("issue", "")
            known_hits.append(record)
        else:
            counter["difference"] += 1
            for artifact in differing:
                counter[f"differs_{artifact}"] += 1
            differences.append(record)
    total = Counter()
    for counter in classes.values():
        total.update(counter)
    return {
        "classes": {cls: dict(counter) for cls, counter in sorted(classes.items())},
        "total": dict(total),
        "differences": differences,
        "known": known_hits,
        "errors": errors,
        "missing": missing,
    }


def run_dump(
    binary: Path, family: suite.Family, images: list[suite.Image], root: Path, work: Path, jobs: int
) -> dict[str, dict]:
    """Run the SMG dump binary over the images in ``jobs`` shards and read its lines by name."""
    shards = [images[i::jobs] for i in range(jobs) if images[i::jobs]]
    procs = []
    for index, shard in enumerate(shards):
        listing = work / f"{family.slug}-{index}.txt"
        listing.write_text("".join(f"{(root / image.path).resolve()}\n" for image in shard), encoding="utf-8")
        out = work / f"{family.slug}-{index}.jsonl"
        env = dict(
            os.environ,
            SMG_DUMP_MODE="hash",
            SMG_DUMP_LIST=str(listing),
            SMG_DUMP_OUT=str(out),
            SMG_DUMP_PROCESSOR=family.smg_processor,
            SMG_DUMP_CONFIG=str(family.config.resolve()),
            SMG_MM_THREADS=os.environ.get("SMG_MM_THREADS", "1"),
        )
        log = (work / f"{family.slug}-{index}.log").open("w")
        procs.append((subprocess.Popen([str(binary)], env=env, stdout=log, stderr=subprocess.STDOUT), log, out))
    lines: dict[str, dict] = {}
    for proc, log, out in procs:
        status = proc.wait()
        log.close()
        if status != 0:
            raise RuntimeError(f"{binary} exited with {status} for {family.slug}; its output is in {log.name}")
        _, records = references.read_recorder_output(out)
        lines.update(records)
    return lines


def junit(report: dict, path: Path) -> None:
    cases = []
    failures = 0
    for family, result in report["families"].items():
        for cls, counter in result["classes"].items():
            failing = [d for d in result["differences"] if d["class"] == cls]
            missing = counter.get("missing", 0) + counter.get("no_reference", 0) + counter.get("error", 0)
            body = ""
            if failing or missing:
                failures += 1
                text = "\n".join(f"{d['name']}: {', '.join(d['differs'])}" for d in failing)
                if missing:
                    text += f"\n{missing} image(s) without a verdict"
                body = f'<failure message="{escape(str(len(failing)))} difference(s)">{escape(text)}</failure>'
            cases.append(f'<testcase classname="multimodal.{escape(family)}" name="{escape(cls)}">{body}</testcase>')
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="bellwether multimodal {escape(report["tier"])}" '
        f'tests="{len(cases)}" failures="{failures}">\n' + "\n".join(cases) + "\n</testsuite>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root: Path = args.root
    try:
        images = suite.tier_images(root, args.tier)
        families = suite.read_families(root)
        known = read_known(args.known if args.known else root / "known.toml")
    except (OSError, ValueError) as err:
        print(f"bellwether multimodal verify: {err}", file=sys.stderr)
        return NO_VERDICT
    if not images:
        print(f"bellwether multimodal verify: tier {args.tier} names no image", file=sys.stderr)
        return NO_VERDICT
    dumps = {}
    for item in args.dumps or []:
        slug, _, path = item.partition("=")
        if not path:
            print(f"bellwether multimodal verify: --dump takes FAMILY=PATH, got {item!r}", file=sys.stderr)
            return NO_VERDICT
        dumps[slug] = Path(path)
    selected = args.families or (sorted(dumps) if dumps else sorted(families))
    if not args.smg_dump and not dumps:
        print("bellwether multimodal verify: give --smg-dump BIN or --dump FAMILY=PATH", file=sys.stderr)
        return NO_VERDICT
    report = {
        "tier": args.tier,
        "source": args.source,
        "images": len(images),
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": {},
    }
    status = 0
    with tempfile.TemporaryDirectory(prefix="bellwether-multimodal-") as tmp:
        work = Path(args.work) if args.work else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        for slug in selected:
            if slug not in families:
                print(f"bellwether multimodal verify: no family {slug}", file=sys.stderr)
                return NO_VERDICT
            try:
                refs = references.read_references(root, slug, args.source)
                if slug in dumps:
                    _, dump = references.read_recorder_output(dumps[slug])
                else:
                    dump = run_dump(Path(args.smg_dump), families[slug], images, root, work, args.jobs)
            except (OSError, ValueError, RuntimeError) as err:
                print(f"bellwether multimodal verify: {err}", file=sys.stderr)
                return NO_VERDICT
            result = judge(slug, images, refs, dump, known)
            result["reference"] = (
                suite.read_sets(root / "references" / slug / suite.SETS_FILE)
                .get((args.source, slug), {})
                .get("provenance", {})
            )
            report["families"][slug] = result
            total = result["total"]
            print(
                f"{slug}: {total.get('images', 0)} images: match {total.get('match', 0)}, "
                f"difference {total.get('difference', 0)}, known {total.get('known', 0)}, "
                f"error {total.get('error', 0)}, "
                f"missing {total.get('missing', 0) + total.get('no_reference', 0)}"
            )
            for difference in result["differences"][: args.show]:
                print(f"  {difference['name']} ({difference['class']}): differs in {', '.join(difference['differs'])}")
            if result["differences"]:
                status = max(status, 1)
            if result["missing"] or result["errors"]:
                status = NO_VERDICT if not result["differences"] and status == 0 and args.strict_missing else status
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.junit:
        junit(report, args.junit)
    return status
