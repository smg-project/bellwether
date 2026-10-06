"""Build the list: one row per checkpoint, from the registries and, unless told otherwise, the Hub.

The registries are pinned and the Hub is not, so a rebuild can change rows; each row pins the
Hub's sha of the day, and the committed list's diff shows what a rebuild changed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

from .hub import Details, Hub, HubUnavailable
from .registry import Entry
from .rules import (
    NO_CHECKPOINT,
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


@dataclass
class _Found:
    """A registry checkpoint as the Hub answered for it, under the Hub's own id."""

    details: Details | None  # None: the Hub has no such model, or gave no answer (``unavailable``)
    named: _Named = field(default_factory=_Named)
    unavailable: str | None = None  # the hub-error status, when the Hub gave no answer


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


def without_checkpoint(entries: Iterable[Entry]) -> dict[str, _Named]:
    """Generative entries that name no checkpoint a row could be keyed by, under the entry's own name.

    vLLM's entry is named by its architecture, an SGLang docs row by its model family; either is the
    row's key, so the entry is in the list though no checkpoint stands for it.
    """
    named: dict[str, _Named] = {}
    for entry in entries:
        if entry.generative and not any(is_hub_id(model) for model in entry.checkpoints):
            named.setdefault(entry.name, _Named()).add([entry.engine], entry.multimodal, example=False)
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
    entries = list(entries)
    named = registry_checkpoints(entries)
    rows = [_row(model, None, n.sources, n.multimodal, built, checked=False) for model, n in named.items()]
    return ordered([*rows, *_rows_without_checkpoint(entries)])


def hub_rows(
    entries: list[Entry], hub: Hub, built: date, log: Callable[[str], None] = lambda message: None
) -> list[Row]:
    """The registries' checkpoints as the Hub has them, then each of their organizations' chat checkpoints."""
    found = _resolve(registry_checkpoints(entries), hub)
    missing = sum(f.details is None and f.unavailable is None for f in found.values())
    unanswered = [model for model, f in found.items() if f.unavailable is not None]
    log(f"registries: {len(found)} checkpoints, {missing} not on the Hub")
    if unanswered:
        log(f"registries: no answer from the Hub for {', '.join(unanswered)}; their rows say why")
    text, multimodal = _registered(entries, found)
    rows = {
        model: _row(model, f.details, f.named.sources, f.named.multimodal, built, checked=True, status=f.unavailable)
        for model, f in found.items()
    }
    # The organizations that publish a registered architecture: those of the engines' examples. vLLM's
    # extras add tiny, random and quantized test models from namespaces that publish none.
    for org in sorted({model.split("/")[0] for model, f in found.items() if f.named.example}):
        try:
            listing = hub.list_models(org)
        except HubUnavailable as err:
            log(f"{org}: not listed, {err.status}")
            continue
        added = 0
        for listed in listing:
            if listed.id in rows or not admits_listing(listed):
                continue
            try:
                details = hub.model(listed.id)
            except HubUnavailable as err:
                log(f"{listed.id}: left out, {err.status}")
                continue
            if details is None or details.id in rows or not admits_details(details, text | multimodal):
                continue
            is_multimodal = any(arch in multimodal for arch in details.architectures)
            rows[details.id] = _row(details.id, details, {"hub"}, is_multimodal, built, checked=True)
            added += 1
        log(f"{org}: {len(listing)} listed, {added} added")
    return ordered([*rows.values(), *_rows_without_checkpoint(entries)])


def _resolve(named: dict[str, _Named], hub: Hub) -> dict[str, _Found]:
    """Each registry checkpoint under the Hub's own id; two names that reach one repository merge.

    A checkpoint the Hub gives no answer for keeps its row, under the registry's name, with the error.
    """
    found: dict[str, _Found] = {}
    for model in sorted(named):
        try:
            details, unavailable = hub.model(model), None
        except HubUnavailable as err:
            details, unavailable = None, err.status
        if details is not None and is_gpt_oss(details.id, details.architectures):
            continue
        merged = found.setdefault(details.id if details else model, _Found(details, unavailable=unavailable))
        merged.named.add(named[model].sources, named[model].multimodal, named[model].example)
    return found


def _registered(entries: list[Entry], found: dict[str, _Found]) -> tuple[set[str], set[str]]:
    """Text and multimodal architectures: vLLM's generative tables, and what registry checkpoints' configs name.

    SGLang's docs name checkpoints, not architectures, so the configs of those checkpoints are what
    says which architectures SGLang serves.
    """
    text: set[str] = set()
    multimodal: set[str] = set()
    for entry in entries:
        if entry.engine == "vllm" and entry.generative:
            (multimodal if entry.multimodal else text).add(entry.name)
    for f in found.values():
        if f.details is not None:
            architectures = {arch for arch in f.details.architectures if not arch.endswith(POOLING_HEADS)}
            (multimodal if f.named.multimodal else text).update(architectures)
    return text, multimodal


def _row(
    model: str,
    details: Details | None,
    sources: Iterable[str],
    multimodal: bool,
    built: date,
    checked: bool,
    status: str | None = None,
) -> Row:
    """One checkpoint's row; ``status``, when given, is the Hub's error in place of what its details say."""
    created = details.created if details else None
    return Row(
        model=model,
        revision=details.sha if details else None,
        tier=tier_of(model, created, built, is_current_chat(details)),
        status=status or status_of(details, checked),
        created=created,
        downloads=details.downloads if details else None,
        modality="multimodal" if multimodal else "text",
        sources=tuple(sorted(sources)),
    )


def _rows_without_checkpoint(entries: Iterable[Entry]) -> list[Row]:
    """A row for each entry that names no checkpoint; the Hub has nothing to say about a name that is none."""
    return [
        Row(
            name, None, 3, NO_CHECKPOINT, None, None, "multimodal" if n.multimodal else "text", tuple(sorted(n.sources))
        )
        for name, n in without_checkpoint(entries).items()
    ]


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
