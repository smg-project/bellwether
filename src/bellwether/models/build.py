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
from .pins import Pin
from .registry import NOTHING_SERVED, Entry, Served
from .rules import (
    NO_CHECKPOINT,
    TIER1,
    admits_details,
    admits_listing,
    is_current_chat,
    is_generative_in_code,
    is_gpt_oss,
    is_hub_id,
    is_quantized_copy,
    is_test_model,
    order_key,
    status_of,
    tier_of,
    tier_without_hub,
)


@dataclass(frozen=True)
class Row:
    model: str
    revision: str | None  # the Hub's sha of the revision vLLM loads, else of the day; without the Hub, vLLM's own
    tier: int | None  # None where the Hub decides it and was not asked, or gave no answer
    status: str
    created: date | None
    downloads: int | None  # over the 30 days before the build
    modality: str | None  # "multimodal" when an architecture of it is, "text"; None when none is known
    sources: tuple[str, ...]  # the engines whose registries name it, or "hub"


@dataclass
class _Named:
    """What the registries say of one checkpoint.

    Which engines name it, the architectures vLLM lists it under (SGLang's docs give none), and
    whether one gives it as an architecture's example (vLLM's default, an id in SGLang's docs) rather
    than as one of vLLM's extra test models; and how vLLM loads it: with the vendor's code, at a
    revision of its own, with the tokenizer of another repository.
    """

    sources: set[str] = field(default_factory=set)
    architectures: set[str] = field(default_factory=set)
    example: bool = False
    vendor_code: bool = False
    revision: str | None = None
    tokenizer: str | None = None

    def add(
        self,
        sources: Iterable[str],
        architectures: Iterable[str],
        example: bool,
        vendor_code: bool = False,
        revision: str | None = None,
        tokenizer: str | None = None,
    ) -> None:
        self.sources.update(sources)
        self.architectures.update(architectures)
        self.example = self.example or example
        self.vendor_code = self.vendor_code or vendor_code
        self.revision = self.revision or revision
        self.tokenizer = self.tokenizer or tokenizer

    def merge(self, other: _Named) -> None:
        self.add(other.sources, other.architectures, other.example, other.vendor_code, other.revision, other.tokenizer)


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
                if is_hub_id(model) and not _is_set_aside(entry, model):
                    example = entry.engine != "vllm" or index == 0
                    # vLLM's revision and tokenizer are those of the default; vendor code is the architecture's.
                    own = (entry.revision, entry.tokenizer) if index == 0 else (None, None)
                    named.setdefault(model, _Named()).add(
                        [entry.engine], _architectures(entry), example, entry.vendor_code, *own
                    )
    return named


def set_aside(entries: Iterable[Entry]) -> list[str]:
    """The registries' gpt-oss checkpoints, which the maintainer set aside: left out of the list, and said aloud."""
    return sorted(
        {
            model
            for entry in entries
            if entry.generative
            for model in entry.checkpoints
            if is_hub_id(model) and _is_set_aside(entry, model)
        }
    )


def _is_set_aside(entry: Entry, model: str) -> bool:
    """gpt-oss by the checkpoint's name, or by the architecture vLLM lists it under."""
    return is_gpt_oss(model, [entry.name] if entry.engine == "vllm" else ())


def without_checkpoint(entries: Iterable[Entry]) -> dict[str, _Named]:
    """Generative entries that name no checkpoint a row could be keyed by, under the entry's own name.

    vLLM's entry is named by its architecture, an SGLang docs row by its model family; either is the
    row's key, so the entry is in the list though no checkpoint stands for it.
    """
    named: dict[str, _Named] = {}
    for entry in entries:
        if entry.generative and not any(is_hub_id(model) for model in entry.checkpoints):
            named.setdefault(entry.name, _Named()).add([entry.engine], _architectures(entry), example=False)
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


def registry_only_rows(entries: Iterable[Entry], built: date, served: Served = NOTHING_SERVED) -> list[Row]:
    """The registries' checkpoints alone: no sha, date or downloads, so no row can be shown to be tier 2."""
    entries = list(entries)
    _, multimodal = _registered(entries, served)
    rows = [_row(model, None, n, built, False, multimodal) for model, n in registry_checkpoints(entries).items()]
    return ordered([*rows, *_rows_without_checkpoint(entries, multimodal)])


def hub_rows(
    entries: list[Entry],
    hub: Hub,
    built: date,
    log: Callable[[str], None] = lambda message: None,
    served: Served = NOTHING_SERVED,
) -> list[Row]:
    """The registries' checkpoints as the Hub has them, then each of their organizations' chat checkpoints."""
    found = _resolve(registry_checkpoints(entries), hub)
    missing = sum(f.details is None and f.unavailable is None for f in found.values())
    unanswered = [model for model, f in found.items() if f.unavailable is not None]
    log(f"registries: {len(found)} checkpoints, {missing} not on the Hub")
    if unanswered:
        log(f"registries: no answer from the Hub for {', '.join(unanswered)}; their rows say why")
    text, multimodal = _registered(entries, served)
    rows = {
        model: _row(model, f.details, f.named, built, True, multimodal, f.unavailable) for model, f in found.items()
    }
    # The organizations that publish a registered architecture: those of the engines' examples, and those
    # of vLLM's extras that are real checkpoints (NousResearch's Hermes 3, mistral-community's Pixtral).
    # The other extras are tiny or random test models and quantized copies, whose namespaces would add
    # nothing to record.
    publishers = {
        model.split("/")[0]
        for model, f in found.items()
        if f.named.example or not (is_test_model(model) or is_quantized_copy(model))
    }
    for org in sorted(publishers):
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
            rows[details.id] = _row(details.id, details, _Named({"hub"}), built, True, multimodal)
            added += 1
        log(f"{org}: {len(listing)} listed, {added} added")
    return ordered([*rows.values(), *_rows_without_checkpoint(entries, multimodal)])


def _resolve(named: dict[str, _Named], hub: Hub) -> dict[str, _Found]:
    """Each registry checkpoint under the Hub's own id; two names that reach one repository merge.

    A checkpoint the Hub gives no answer for keeps its row, under the registry's name, with the error.
    """
    found: dict[str, _Found] = {}
    for model in sorted(named):
        try:
            details, unavailable = hub.model(model, named[model].revision, named[model].tokenizer), None
        except HubUnavailable as err:
            details, unavailable = None, err.status
        if details is not None and is_gpt_oss(details.id, details.architectures):
            continue
        merged = found.setdefault(details.id if details else model, _Found(details, unavailable=unavailable))
        merged.named.merge(named[model])
    return found


def _registered(entries: list[Entry], served: Served) -> tuple[set[str], set[str]]:
    """Text and multimodal architectures: vLLM's generative tables, and what SGLang's code serves.

    SGLang's code also serves pooling and draft heads, which are left out, and so is a name vLLM
    files only among its pooling, draft or backend models. An architecture SGLang serves through a
    multimodal processor is multimodal.
    """
    text: set[str] = set()
    multimodal: set[str] = set()
    elsewhere: set[str] = set()
    for entry in entries:
        if entry.engine == "vllm":
            if entry.generative:
                (multimodal if entry.multimodal else text).add(entry.name)
            else:
                elsewhere.add(entry.name)
    only_elsewhere = elsewhere - text - multimodal
    for architecture in served.architectures:
        if is_generative_in_code(architecture) and architecture not in only_elsewhere:
            (multimodal if architecture in served.multimodal else text).add(architecture)
    return text, multimodal


def _row(
    model: str,
    details: Details | None,
    named: _Named,
    built: date,
    checked: bool,
    multimodal: set[str],
    status: str | None = None,
) -> Row:
    """One checkpoint's row; ``status``, when given, is the Hub's error in place of what its details say."""
    created = details.created if details else None
    answered = checked and status is None
    return Row(
        model=model,
        revision=details.sha if details else named.revision,
        tier=tier_of(model, created, built, is_current_chat(details)) if answered else tier_without_hub(model),
        status=status or status_of(details, checked, named.vendor_code),
        created=created,
        downloads=details.downloads if details else None,
        modality=_modality(named.architectures | set(details.architectures if details else ()), multimodal),
        sources=tuple(sorted(named.sources)),
    )


def _rows_without_checkpoint(entries: Iterable[Entry], multimodal: set[str]) -> list[Row]:
    """A row for each entry that names no checkpoint; the Hub has nothing to say about a name that is none."""
    return [
        Row(name, None, 3, NO_CHECKPOINT, None, None, _modality(n.architectures, multimodal), tuple(sorted(n.sources)))
        for name, n in without_checkpoint(entries).items()
    ]


def _architectures(entry: Entry) -> set[str]:
    """The architectures an entry gives its checkpoints under: vLLM's key; SGLang's docs give a family."""
    return {entry.name} if entry.engine == "vllm" else set()


def _modality(architectures: set[str], multimodal: set[str]) -> str | None:
    """Multimodal when one of its architectures is, in either engine's code; unknown without an architecture."""
    if not architectures:
        return None
    return "multimodal" if architectures & multimodal else "text"


def missing_from_tier1(rows: Iterable[Row]) -> list[str]:
    """The maintainer's named models the list lacks; tier 1 ranks rows and adds none, so a missing one is said aloud."""
    models = {row.model for row in rows}
    return [entry for entry in TIER1 if not entry.endswith("/") and entry not in models]


def ordered(rows: Iterable[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: order_key(row.model, row.tier, row.downloads))


def header(pins: Iterable[Pin], hub: date | None) -> str:
    """The list's first line: the registries it was read from, and the day the Hub was read, if it was.

    Tier 2 and downloads depend on that day; without the Hub nothing in the list depends on a date.
    """
    values = {
        "registries": {pin.engine: {"repo": pin.repo, "ref": pin.ref, "commit": pin.commit} for pin in pins},
        "hub": hub.isoformat() if hub else None,
    }
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")) + "\n"


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
