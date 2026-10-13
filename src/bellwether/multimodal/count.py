"""``bellwether multimodal count``: what the suite holds, as tables, from the manifest, the tiers and each family's
``sets.toml``; no image or set is read. With ``--readme`` the tables go into ``multimodal/README.md`` between its
markers, and ``--check`` compares them with the file as CI runs it, so they say what main holds.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, OrderedDict
from pathlib import Path

from bellwether.multimodal import suite

README_BEGIN = (
    "<!-- bellwether multimodal count --readme writes everything from here to the end marker: "
    "edit none of it by hand -->"
)
README_END = "<!-- end of what bellwether multimodal count --readme writes -->"


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def tables(root: Path) -> str:
    images = suite.read_manifest(root)
    tiers = {
        tier: set(suite.read_tier(root, tier)) for tier in suite.TIERS if (root / "tiers" / f"{tier}.txt").is_file()
    }
    lines = [
        "### Images by dataset",
        "",
        "| Dataset | Images | Bytes | Classes | In corpus-50 | In corpus-500 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    per_source: OrderedDict[str, Counter] = OrderedDict()
    classes_of: dict[str, set[str]] = {}
    for image in images:
        c = per_source.setdefault(image.source, Counter())
        c["images"] += 1
        c["bytes"] += image.bytes
        c["c50"] += image.path in tiers.get("corpus-50", set())
        c["c500"] += image.path in tiers.get("corpus-500", set())
        classes_of.setdefault(image.source, set()).add(image.cls)
    total = Counter()
    for source in sorted(per_source):
        c = per_source[source]
        total.update(c)
        lines.append(
            f"| {source} | {c['images']} | {_mb(c['bytes'])} | {len(classes_of[source])} | {c['c50']} | {c['c500']} |"
        )
    lines.append(
        f"| all | {total['images']} | {_mb(total['bytes'])} | {len({i.cls for i in images})} | "
        f"{total['c50']} | {total['c500']} |"
    )
    lines += ["", "### Classes", "", "| Class | Images | In corpus-50 | In corpus-500 |", "|---|---:|---:|---:|"]
    per_class: dict[str, Counter] = {}
    for image in images:
        c = per_class.setdefault(image.cls, Counter())
        c["images"] += 1
        c["c50"] += image.path in tiers.get("corpus-50", set())
        c["c500"] += image.path in tiers.get("corpus-500", set())
    for cls in sorted(per_class):
        c = per_class[cls]
        lines.append(f"| {cls} | {c['images']} | {c['c50']} | {c['c500']} |")
    lines += [
        "",
        "### References",
        "",
        "Each family's recorded sets, from its `sets.toml`; `cases` counts the images with a result, `errors` those "
        "the recorder could not process.",
        "",
        "| Family | Source | Cases | Errors | Plain | Stored | Raw arrays | Recorder |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for slug, family in sorted(suite.read_families(root).items()):
        sets = suite.read_sets(family.path.parent.parent / "references" / slug / suite.SETS_FILE)
        if not sets:
            lines.append(f"| {slug} | | 0 | 0 | | | | nothing recorded yet |")
        for (source, _), table in sorted(sets.items()):
            prov = table.get("provenance", {})
            who = ", ".join(f"{k} {prov[k]}" for k in ("vllm", "transformers", "torch", "pillow") if k in prov)
            raw = table.get("raw_files", "")
            lines.append(
                f"| {slug} | {source} | {table.get('cases', 0)} | {table.get('errors', 0)} | "
                f"{_mb(int(table.get('plain_bytes', 0)))} | {_mb(int(table.get('stored_bytes', 0)))} | {raw} | {who} |"
            )
    return "\n".join(lines) + "\n"


def replace_between(text: str, body: str) -> str:
    begin = text.index(README_BEGIN) + len(README_BEGIN)
    end = text.index(README_END)
    return text[:begin] + "\n\n" + body + "\n" + text[end:]


def run(args: argparse.Namespace) -> int:
    try:
        body = tables(args.root)
    except (OSError, ValueError) as err:
        print(f"bellwether multimodal count: {err}", file=sys.stderr)
        return 1
    if args.readme is None:
        print(body, end="")
        return 0
    text = args.readme.read_text(encoding="utf-8")
    if README_BEGIN not in text or README_END not in text:
        print(f"bellwether multimodal count: {args.readme} lacks the markers", file=sys.stderr)
        return 1
    updated = replace_between(text, body)
    if args.check:
        if updated != text:
            print(
                f"bellwether multimodal count: the tables in {args.readme} are not what the suite holds; "
                "run without --check",
                file=sys.stderr,
            )
            return 1
        print(f"{args.readme}: the tables are what the suite holds")
        return 0
    args.readme.write_text(updated, encoding="utf-8")
    print(f"{args.readme}: tables written")
    return 0
