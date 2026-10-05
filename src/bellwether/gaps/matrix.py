"""Merge the registries into one matrix per kind and render it as markdown or canonical JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .manifests import Manifest
from .names import Aliases
from .registries import KINDS, SYSTEMS, Registry

SCHEMA_VERSION = 1
_TITLES = {
    "tool": "Tool-call parsers",
    "reasoning": "Reasoning parsers",
    "renderer": "Renderers",
    "tokenizer": "Tokenizer modes",
}
_SYSTEM_LABELS = {"vllm": "vLLM", "sglang": "SGLang", "smg": "SMG"}


@dataclass
class Row:
    kind: str
    key: str
    label: str = ""
    cells: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: {s: [] for s in SYSTEMS})
    fixtures: list[str] = field(default_factory=list)


@dataclass
class KindMatrix:
    kind: str
    rows: list[Row]
    counts: dict[str, int]
    engine_only: list[str]
    smg_only: list[str]
    engines_differ: list[str]
    shared_impl: list[tuple[str, str, list[str]]]


@dataclass
class Matrix:
    sources: dict[str, Registry]
    aliases_path: str
    kinds: dict[str, KindMatrix]


def build_matrix(registries: dict[str, Registry], aliases: Aliases, manifests: list[Manifest]) -> Matrix:
    kinds: dict[str, KindMatrix] = {}
    for kind in KINDS:
        rows: dict[str, Row] = {}
        for system in SYSTEMS:
            reg = registries.get(system)
            if reg is None:
                continue
            for entry in sorted(e for e in reg.entries if e.kind == kind):
                key = aliases.canonical(kind, system, entry.name)
                rows.setdefault(key, Row(kind, key)).cells[system].append((entry.name, entry.impl))
        for row in rows.values():
            row.label = next((row.cells[s][0][0] for s in SYSTEMS if row.cells[s]), row.key)
            for manifest in manifests:
                name = manifest.smg.get(kind)
                if name and any(n == name for n, _ in row.cells["smg"]):
                    row.fixtures.append(manifest.slug)
        ordered = [rows[k] for k in sorted(rows)]
        kinds[kind] = KindMatrix(
            kind=kind,
            rows=ordered,
            counts={s: len({n for r in ordered for n, _ in r.cells[s]}) for s in SYSTEMS},
            engine_only=[r.key for r in ordered if (r.cells["vllm"] or r.cells["sglang"]) and not r.cells["smg"]],
            smg_only=[r.key for r in ordered if r.cells["smg"] and not (r.cells["vllm"] or r.cells["sglang"])],
            engines_differ=[r.key for r in ordered if bool(r.cells["vllm"]) != bool(r.cells["sglang"])],
            shared_impl=_shared_impl(ordered),
        )
    return Matrix(registries, aliases.path, kinds)


def _shared_impl(rows: list[Row]) -> list[tuple[str, str, list[str]]]:
    """One implementation behind several rows in one system: evidence for the alias table."""
    by_impl: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        for system in SYSTEMS:
            for _, impl in row.cells[system]:
                if impl:
                    by_impl.setdefault((system, impl), set()).add(row.key)
    return sorted((s, impl, sorted(keys)) for (s, impl), keys in by_impl.items() if len(keys) > 1)


def to_dict(matrix: Matrix) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "aliases": matrix.aliases_path,
        "sources": {
            system: {
                "commit": reg.commit,
                "roots": [str(r) for r in reg.roots],
                "notes": list(reg.notes),
                "model_patterns": reg.model_patterns,
            }
            for system, reg in matrix.sources.items()
        },
        "kinds": {
            kind: {
                "counts": km.counts,
                "rows": [
                    {
                        "key": r.key,
                        "label": r.label,
                        "cells": {s: [{"name": n, "impl": i} for n, i in r.cells[s]] for s in SYSTEMS},
                        "fixtures": r.fixtures,
                    }
                    for r in km.rows
                ],
                "engine_only": km.engine_only,
                "smg_only": km.smg_only,
                "engines_differ": km.engines_differ,
                "shared_impl": [{"system": s, "impl": i, "rows": rows} for s, i, rows in km.shared_impl],
            }
            for kind, km in matrix.kinds.items()
        },
    }


def render_json(matrix: Matrix) -> str:
    return json.dumps(to_dict(matrix), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _cell(names: list[tuple[str, str]]) -> str:
    return ", ".join(f"{n} ({i})" if i else n for n, i in names)


def render_markdown(matrix: Matrix) -> str:
    out = ["# Coverage matrix: vLLM, SGLang, SMG", ""]
    for system, reg in matrix.sources.items():
        out.append(f"- {_SYSTEM_LABELS[system]} at `{reg.commit}`")
        out.extend(f"  - note: {note}" for note in reg.notes)
    out.append(f"- aliases: `{matrix.aliases_path}`")
    out.append("")
    for kind, km in matrix.kinds.items():
        counts = ", ".join(f"{_SYSTEM_LABELS[s]} {km.counts[s]}" for s in SYSTEMS)
        out.append(f"## {_TITLES[kind]} ({counts} names; {len(km.rows)} rows)")
        out.append("")
        out.append("| row | vLLM | SGLang | SMG | fixtures |")
        out.append("|---|---|---|---|---|")
        for r in km.rows:
            cells = " | ".join(_cell(r.cells[s]) for s in SYSTEMS)
            out.append(f"| {r.label} | {cells} | {', '.join(r.fixtures)} |")
        out.append("")
        out.append(
            f"Engine names with no SMG counterpart ({len(km.engine_only)}): {', '.join(km.engine_only) or 'none'}"
        )
        out.append("")
        out.append(f"SMG-only names ({len(km.smg_only)}): {', '.join(km.smg_only) or 'none'}")
        out.append("")
        out.append(f"Only one engine has it ({len(km.engines_differ)}): {', '.join(km.engines_differ) or 'none'}")
        out.append("")
        if km.shared_impl:
            out.append("One implementation behind several rows (alias candidates):")
            out.extend(f"- {_SYSTEM_LABELS[s]} `{impl}`: {', '.join(rows)}" for s, impl, rows in km.shared_impl)
            out.append("")
    return "\n".join(out)
