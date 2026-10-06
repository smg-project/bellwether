"""The rules that decide which checkpoints are on the list, and in which order.

Each rule is a pure function of what the registries and the Hub say, so it can be checked against
the design (``docs/benchmark-sets.md``, "Which models") without a network.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from datetime import date

from .hub import Details, Listed

# A word of the model name that marks a quantized or converted copy. A copy carries its source's
# tokenizer and template, so recording it adds nothing. ``bnb`` is bitsandbytes' usual short form.
QUANTIZED_MARKERS = frozenset(
    {"gguf", "awq", "gptq", "mlx", "onnx", "fp8", "nvfp4", "mxfp4", "mxfp8", "int4", "int8", "bnb", "bitsandbytes"}
)
# The Hub's own word for the same thing, from the card's ``base_model_relation: quantized``. Tags
# are read for this one statement only: a format tag such as ``fp8`` also marks native releases.
QUANTIZED_TAG = "base_model:quantized:"

# vLLM's registry tables for generative architectures; the others hold pooling models, draft
# models for speculative decoding, and the Transformers backend's wrappers.
GENERATIVE_TABLES = frozenset({"_TEXT_GENERATION_EXAMPLE_MODELS", "_MULTIMODAL_EXAMPLE_MODELS"})
# Heads that score or classify instead of generating; a generative table holds a few of them.
POOLING_HEADS = (
    "ForRanking",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForRetrieval",
    "ForRewardModel",
    "ForProcessRewardModel",
    "ForEmbedding",
)
# Heads that draft tokens for speculative decoding. SGLang's code serves them beside the models they
# draft for; vLLM keeps them in a table of their own.
DRAFT_HEADS = ("Eagle", "Eagle3", "NextN", "MTP", "DSpark")
# Pipeline tags of embedding, reranking and classification models, which the list leaves out.
NON_GENERATIVE_PIPELINES = frozenset(
    {
        "feature-extraction",
        "sentence-similarity",
        "text-ranking",
        "text-classification",
        "token-classification",
        "zero-shot-classification",
        "fill-mask",
    }
)

CUTOFF = date(2025, 1, 1)  # older checkpoints are listed only when a registry names them
# Simo's tier, in his order. An entry ending in "/" is an organization: all its current chat
# checkpoints, in order of downloads.
TIER1 = (
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "MiniMaxAI/MiniMax-M3",
    "zai-org/GLM-5.3-Flash",
    "Qwen/",
    "tencent/Hy4-preview",
)

UNCHECKED = "unchecked"  # --registry-only: the Hub was not asked
NOT_ON_HUB = "not-on-hub"  # a registry names it, the Hub has no such model
NO_CHECKPOINT = "no-checkpoint-named"  # a registry entry names none; its row is keyed by the entry's name
GATED = "gated"  # the files need an accepted license and a token
NEEDS_VENDOR_CODE = "needs-vendor-code"  # vLLM loads it with trust_remote_code; the oracle would need that code
NO_CHAT_TEMPLATE = "no-chat-template"  # the hf-template oracle has nothing to render with
# The template is only in the processor's files, which the oracle (AutoTokenizer) does not read yet.
PROCESSOR_CHAT_TEMPLATE = "processor-chat-template"
PENDING = "pending"  # nothing recorded yet, nothing in the way
# A tokenizer or processor config that could not be read gives its own status (``Details.unread``):
# hub-error- and the Hub's HTTP status or the error's name, invalid-tokenizer-config, invalid-processor-config.

_HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")


def is_quantized_copy(model_id: str, tags: Iterable[str] = ()) -> bool:
    """Only whole words of the name count, so neither an organization's name nor a longer word trips it."""
    words = re.split(r"[^a-z0-9]+", model_id.rsplit("/", 1)[-1].lower())
    return any(word in QUANTIZED_MARKERS for word in words) or any(tag.startswith(QUANTIZED_TAG) for tag in tags)


def is_gpt_oss(model_id: str, architectures: Iterable[str] = ()) -> bool:
    """gpt-oss is set aside (Simo); a checkpoint built on it says so in its name or its architecture."""
    return "gpt-oss" in model_id.lower() or "GptOssForCausalLM" in architectures


def is_hub_id(name: str) -> bool:
    """Registries also hold placeholders (``""``) and local names; a Hub checkpoint is ``org/name``."""
    return _HUB_ID.fullmatch(name) is not None


def is_generative_architecture(table: str, architecture: str) -> bool:
    return table in GENERATIVE_TABLES and not architecture.endswith(POOLING_HEADS)


def is_generative_in_code(architecture: str) -> bool:
    """An architecture an engine's code serves, unless it scores or classifies, or drafts for another."""
    return not architecture.endswith(POOLING_HEADS + DRAFT_HEADS)


def is_generative_pipeline(pipeline_tag: str | None) -> bool:
    """A model without a pipeline tag gets the benefit of the doubt; its architecture decides."""
    return pipeline_tag not in NON_GENERATIVE_PIPELINES


def passes_date_rule(created: date | None, registry_named: bool) -> bool:
    return registry_named or (created is not None and created >= CUTOFF)


def admits_listing(listed: Listed) -> bool:
    """The rules a listing settles alone, so a model they rule out costs no second call."""
    return (
        not is_gpt_oss(listed.id)
        and not is_quantized_copy(listed.id, listed.tags)
        and passes_date_rule(listed.created, registry_named=False)
        and is_generative_pipeline(listed.pipeline_tag)
    )


def admits_details(details: Details, registered: Collection[str]) -> bool:
    """A Hub model joins the list when its config names a registered architecture and it ships a template.

    A tokenizer config that could not be read may hold one, so its model joins with a status that says why.
    """
    return (
        (details.chat_template or details.unread is not None)
        and any(arch in registered for arch in details.architectures)
        and not is_gpt_oss(details.id, details.architectures)
    )


def is_current_chat(details: Details | None) -> bool:
    """What tier 1 means by a current chat checkpoint: the Hub rule without the registry exception."""
    return (
        details is not None
        and details.chat_template
        and details.created is not None
        and details.created >= CUTOFF
        and not is_quantized_copy(details.id, details.tags)
    )


def tier_of(model_id: str, created: date | None, built: date, current_chat: bool) -> int:
    if _tier1_rank(model_id, current_chat) is not None:
        return 1
    if created is not None and created >= _twelve_months_before(built):
        return 2
    return 3


def tier_without_hub(model_id: str) -> int | None:
    """Without the Hub's answer only Simo's named models have a tier; the rest wait on what the Hub says.

    Whether a checkpoint is a current chat checkpoint, and when it was created, are the Hub's to tell.
    """
    return 1 if model_id in TIER1 else None


# A tier left to the Hub sorts after the tiers the Hub could have given before it (1 and 2), before 3.
_TIER_ORDER = {1: 1, 2: 2, None: 3, 3: 4}


def order_key(model_id: str, tier: int | None, downloads: int | None) -> tuple[int, int, int, str]:
    """Tier, then Simo's order inside tier 1, then 30-day downloads, then the id so ties are stable."""
    rank = _tier1_rank(model_id, current_chat=True) if tier == 1 else None
    return (_TIER_ORDER[tier], rank or 0, -(downloads or 0), model_id)


def status_of(details: Details | None, checked: bool, vendor_code: bool = False) -> str:
    """What stands in the way first: the Hub's answer, then what vLLM needs to load it, then the template."""
    if not checked:
        return NEEDS_VENDOR_CODE if vendor_code else UNCHECKED
    if details is None:
        return NOT_ON_HUB
    if details.gated:
        return GATED
    if details.unread:
        return details.unread
    if vendor_code:
        return NEEDS_VENDOR_CODE
    if not details.chat_template:
        return NO_CHAT_TEMPLATE
    if details.processor_only:
        return PROCESSOR_CHAT_TEMPLATE
    return PENDING


def _tier1_rank(model_id: str, current_chat: bool) -> int | None:
    for rank, entry in enumerate(TIER1):
        if model_id == entry or (entry.endswith("/") and current_chat and model_id.startswith(entry)):
            return rank
    return None


def _twelve_months_before(built: date) -> date:
    try:
        return built.replace(year=built.year - 1)
    except ValueError:  # built on 29 February
        return built.replace(year=built.year - 1, day=28)
