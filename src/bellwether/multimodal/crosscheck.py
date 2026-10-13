"""``bellwether multimodal cross-check``: the engine's own image path against the cross-check, per family.

The reference is what the engine's image gives (its media loader, then its multimodal processor); the cross-check is
transformers' processor behind the same loader semantics, run outside the engine. Where the two differ on an image, the
engine's path overrides or extends the processor, and the difference is recorded here as a finding, never averaged
away: the decoded RGB bytes, the shape, the bf16 and the f32 wire bytes are compared, and the u8 patches where both
recorded them. Exit 0 with the report; ``--strict`` exits 1 when the two differ on any image.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bellwether.multimodal import references, suite
from bellwether.multimodal.verify import compare


def run(args: argparse.Namespace) -> int:
    root: Path = args.root
    try:
        images = suite.by_name(suite.read_manifest(root))
        families = suite.read_families(root)
    except (OSError, ValueError) as err:
        print(f"bellwether multimodal cross-check: {err}", file=sys.stderr)
        return 2
    report: dict[str, dict] = {}
    differ_any = False
    for slug in args.families or sorted(families):
        try:
            engine = references.read_references(root, slug, "vllm")
            check = references.read_references(root, slug, "hf")
        except (OSError, ValueError) as err:
            print(f"{slug}: {err}", file=sys.stderr)
            continue
        classes: dict[str, Counter] = {}
        differences = []
        for name in sorted(set(engine) & set(check)):
            image = images.get(name)
            cls = image.cls if image else "?"
            counter = classes.setdefault(cls, Counter())
            counter["both"] += 1
            if "error" in engine[name] or "error" in check[name]:
                counter["error"] += 1
                continue
            differing = compare(engine[name], check[name])
            if differing:
                counter["differ"] += 1
                differences.append(
                    {
                        "name": name,
                        "class": cls,
                        "differs": differing,
                        "engine": {k: engine[name].get(k) for k in ("w", "h", "grid", "shape")},
                        "cross_check": {k: check[name].get(k) for k in ("w", "h", "grid", "shape")},
                    }
                )
            else:
                counter["agree"] += 1
        total = Counter()
        for counter in classes.values():
            total.update(counter)
        only_engine = len(set(engine) - set(check))
        only_check = len(set(check) - set(engine))
        report[slug] = {
            "classes": {c: dict(v) for c, v in sorted(classes.items())},
            "total": dict(total),
            "only_engine": only_engine,
            "only_cross_check": only_check,
            "differences": differences,
        }
        differ_any = differ_any or bool(differences)
        print(
            f"{slug}: {total.get('both', 0)} images in both sets: agree {total.get('agree', 0)}, "
            f"differ {total.get('differ', 0)}, error {total.get('error', 0)}; only the engine {only_engine}, "
            f"only the cross-check {only_check}"
        )
        for difference in differences[: args.show]:
            print(f"  {difference['name']} ({difference['class']}): {', '.join(difference['differs'])}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if args.strict and differ_any else 0
