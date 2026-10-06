"""Build the list: one row per checkpoint, from the registries and, unless told otherwise, the Hub.

The registries are pinned and the Hub is not, so a rebuild can change rows; each row pins the
Hub's sha of the day, and the committed list's diff shows what a rebuild changed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

from .hub import Details, Hub
from .registry import Entry
from .rules import (
    POOLING_HEADS,
    TIER1,
    admits_details,
    admits_listing,
    is_current_chat,
    is_gpt_oss,
    is_hub_id,
    order_key,
    status_of,
    tier_of,
)


@dataclass(frozen=True)
class Row:
    model: str
    revision: str | None  # the Hub's sha when the list was built
    tier: int
    status: str
    created: date | None
    downloads: int | None  # over the 30 days before the build
    modality: str  # "text" or "multimodal"
    sources: tuple[str, ...]  # the engines whose registries name it, or "hub"


@dataclass
class _Named:
    """What the registries say of one checkpoint.

    Which engines name it, whether one calls it multimodal, and whether one gives it as an
    architecture's example (vLLM's default, an id in SGLang's docs) rather than as one of vLLM's
    extra test models.
    """

    sources: set[str] = field(default_factory=set)
    multimodal: bool = False
    example: bool = False

    def add(self, sources: Iterable[str], multimodal: bool, example: bool) -> None:
        self.sources.update(sources)
        self.multimodal = self.multimodal or multimodal
        self.example = self.example or example


def registry_checkpoints(entries: Iterable[Entry]) -> dict[str, _Named]:
    """Every checkpoint a generative entry names; gpt-oss and names that are no checkpoint are left out."""
    named: dict[str, _Named] = {}
    for entry in entries:
        if entry.generative:
            for index, model in enumerate(entry.checkpoints):
                if is_hub_id(model) and not is_gpt_oss(model):
                    example = entry.engine != "vllm" or index == 0
                    named.setdefault(model, _Named()).add([entry.engine], entry.multimodal, example)
    return named


def unnamed(entries: Iterable[Entry]) -> list[str]:
    """Generative entries that name no checkpoint, or name something that is not one: a run's notes."""
    notes = []
    for entry in entries:
        if not entry.generative:
            continue
        if not entry.checkpoints:
            notes.append(f"{entry.engine} {entry.name}: no checkpoint named")
        notes.extend(
            f"{entry.engine} {entry.name}: {name!r} is not a Hugging Face id"
            for name in entry.checkpoints
            if not is_hub_id(name)
        )
    return notes


def registry_only_rows(entries: Iterable[Entry], built: date) -> list[Row]:
    """The registries' checkpoints alone: no sha, date or downloads, so no row can be shown to be tier 2."""
    named = registry_checkpoints(entries)
    return ordered(_row(model, None, n.sources, n.multimodal, built, checked=False) for model, n in named.items())


def hub_rows(
    entries: list[Entry], hub: Hub, built: date, log: Callable[[str], None] = lambda message: None
) -> list[Row]:
    """The registries' checkpoints as the Hub has them, then each of their organizations' chat checkpoints."""
    found = _resolve(registry_checkpoints(entries), hub)
    log(f"registries: {len(found)} checkpoints, {sum(d is None for d, _ in found.values())} not on the Hub")
    text, multimodal = _registered(entries, found)
    rows = {
        model: _row(model, details, named.sources, named.multimodal, built, checked=True)
        for model, (details, named) in found.items()
    }
    # The organizations that publish a registered architecture: those of the engines' examples. vLLM's
    # extras add tiny, random and quantized test models from namespaces that publish none.
    for org in sorted({model.split("/")[0] for model, (_, named) in found.items() if named.example}):
        listing = hub.list_models(org)
        added = 0
        for listed in listing:
            if listed.id in rows or not admits_listing(listed):
                continue
            details = hub.model(listed.id)
            if details is None or details.id in rows or not admits_details(details, text | multimodal):
                continue
            is_multimodal = any(arch in multimodal for arch in details.architectures)
            rows[details.id] = _row(details.id, details, {"hub"}, is_multimodal, built, checked=True)
            added += 1
        log(f"{org}: {len(listing)} listed, {added} added")
    return ordered(rows.values())


def _resolve(named: dict[str, _Named], hub: Hub) -> dict[str, tuple[Details | None, _Named]]:
    """Each registry checkpoint under the Hub's own id; two names that reach one repository merge."""
    found: dict[str, tuple[Details | None, _Named]] = {}
    for model in sorted(named):
        details = hub.model(model)
        if details is not None and is_gpt_oss(details.id, details.architectures):
            continue
        _, merged = found.setdefault(details.id if details else model, (details, _Named()))
        merged.add(named[model].sources, named[model].multimodal, named[model].example)
    return found


def _registered(entries: list[Entry], found: dict[str, tuple[Details | None, _Named]]) -> tuple[set[str], set[str]]:
    """Text and multimodal architectures: vLLM's generative tables, and what registry checkpoints' configs name.

    SGLang's docs name checkpoints, not architectures, so the configs of those checkpoints are what
    says which architectures SGLang serves.
    """
    text: set[str] = set()
    multimodal: set[str] = set()
    for entry in entries:
        if entry.engine == "vllm" and entry.generative:
            (multimodal if entry.multimodal else text).add(entry.name)
    for details, named in found.values():
        if details is not None:
            architectures = {arch for arch in details.architectures if not arch.endswith(POOLING_HEADS)}
            (multimodal if named.multimodal else text).update(architectures)
    return text, multimodal


def _row(
    model: str, details: Details | None, sources: Iterable[str], multimodal: bool, built: date, checked: bool
) -> Row:
    created = details.created if details else None
    return Row(
        model=model,
        revision=details.sha if details else None,
        tier=tier_of(model, created, built, is_current_chat(details)),
        status=status_of(details, checked),
        created=created,
        downloads=details.downloads if details else None,
        modality="multimodal" if multimodal else "text",
        sources=tuple(sorted(sources)),
    )


def missing_from_tier1(rows: Iterable[Row]) -> list[str]:
    """Simo's named models the list lacks; tier 1 ranks rows and adds none, so a missing one is said aloud."""
    models = {row.model for row in rows}
    return [entry for entry in TIER1 if not entry.endswith("/") and entry not in models]


def ordered(rows: Iterable[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: order_key(row.model, row.tier, row.downloads))


def to_jsonl(rows: Iterable[Row]) -> str:
    """One compact line per row, keys in a fixed order, so a rebuild diffs line by line."""
    return "".join(f"{_line(row)}\n" for row in rows)


def _line(row: Row) -> str:
    values = {
        "model": row.model,
        "revision": row.revision,
        "tier": row.tier,
        "status": row.status,
        "created": row.created.isoformat() if row.created else None,
        "downloads": row.downloads,
        "modality": row.modality,
        "sources": list(row.sources),
    }
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
