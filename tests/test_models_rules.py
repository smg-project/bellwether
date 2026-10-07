"""Tests for the rules that decide which checkpoints ``bellwether models`` lists and in which order.

The Qwen data is the Hub's listing of the Qwen organization on 2026-10-06 (id, created date, 30-day
downloads, pipeline tag, and the one kind of tag the rules read), with the 64 chat checkpoints a
first pass over that listing chose by hand. That pass is evidence, not the rule: it also left out
base checkpoints that ship a chat template, which the rule keeps.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from bellwether.models import rules
from bellwether.models.hub import Details, Listed

DATA = Path(__file__).parent / "data" / "models"
BUILT = date(2026, 10, 6)


def qwen_listing() -> list[Listed]:
    lines = (DATA / "qwen-listing-2026-10-06.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return [
        Listed(r["id"], date.fromisoformat(r["created"]), r["downloads"], tuple(r["tags"]), r["pipeline_tag"])
        for r in rows
    ]


def qwen_chat() -> set[str]:
    return set((DATA / "qwen-chat-2026-10-06.txt").read_text().split())


def details(**overrides) -> Details:
    fields = {
        "id": "Qwen/Qwen3-8B",
        "sha": "b968826d",
        "created": date(2025, 4, 27),
        "downloads": 1000,
        "tags": (),
        "architectures": ("Qwen3ForCausalLM",),
        "chat_template": True,
        "gated": False,
    }
    return Details(**{**fields, **overrides})


@pytest.mark.parametrize(
    "model",
    [
        "Qwen/Qwen3-8B-GGUF",
        "Qwen/Qwen3-8B-AWQ",
        "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        "Qwen/Qwen3-14B-MLX-bf16",
        "microsoft/Phi-3-mini-4k-instruct-onnx",
        "Qwen/Qwen3-8B-FP8",
        "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4",
        "example/Model-MXFP4",
        "example/Model-mxfp8",
        "example/Model-7B-int4-ov",
        "Qwen/Qwen3-0.6B-GPTQ-Int8",
        "example/Model-7B-bnb-4bit",
        "example/Model-7B-bitsandbytes",
    ],
)
def test_a_quantization_or_conversion_marker_in_the_name_marks_a_copy(model: str) -> None:
    assert rules.is_quantized_copy(model)


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-V3", "FP8-Lab/Chat-7B", "example/Gguface-7B", "example/Mlxtral-8x7B"],
)
def test_a_marker_counts_only_as_a_word_of_the_model_name(model: str) -> None:
    assert not rules.is_quantized_copy(model)


def test_the_hubs_quantized_base_model_tag_marks_a_copy_whatever_its_name() -> None:
    assert rules.is_quantized_copy("example/Qwen3-8B-quantized.w4a16", ("base_model:quantized:Qwen/Qwen3-8B",))
    assert not rules.is_quantized_copy("example/Qwen3-8B-quantized.w4a16")
    assert not rules.is_quantized_copy("example/Qwen3-8B-tuned", ("base_model:finetune:Qwen/Qwen3-8B",))


def test_no_qwen_chat_checkpoint_is_a_copy_and_every_tagged_copy_is_caught() -> None:
    listing = qwen_listing()
    copies = {m.id for m in listing if rules.is_quantized_copy(m.id, m.tags)}
    assert not copies & qwen_chat()
    tagged = {m.id for m in listing if any(t.startswith("base_model:quantized:") for t in m.tags)}
    assert tagged <= copies
    marked = {m.id for m in listing if any(k in m.id.lower() for k in ("gguf", "awq", "gptq", "mlx", "fp8", "int4"))}
    assert marked <= copies
    assert len(copies) == 243


def test_gpt_oss_is_left_out_by_name_or_by_architecture() -> None:
    assert rules.is_gpt_oss("openai/gpt-oss-20b")
    assert rules.is_gpt_oss("lmsys/gpt-oss-20b-bf16")
    assert rules.is_gpt_oss("OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview")
    assert rules.is_gpt_oss("example/renamed-checkpoint", ("GptOssForCausalLM",))
    assert not rules.is_gpt_oss("Qwen/Qwen3-8B", ("Qwen3ForCausalLM",))


def test_only_an_organization_and_a_name_make_a_hub_id() -> None:
    assert rules.is_hub_id("Qwen/Qwen3-8B")
    assert rules.is_hub_id("rednote-hilab/dots.vlm1.inst")
    assert not rules.is_hub_id("")
    assert not rules.is_hub_id("funaudiochat")
    assert not rules.is_hub_id("a/b/c")


def test_only_the_text_generation_and_multimodal_tables_are_generative() -> None:
    assert rules.is_generative_architecture("_TEXT_GENERATION_EXAMPLE_MODELS", "Qwen3ForCausalLM")
    assert rules.is_generative_architecture("_MULTIMODAL_EXAMPLE_MODELS", "Qwen2VLForConditionalGeneration")
    assert not rules.is_generative_architecture("_EMBEDDING_EXAMPLE_MODELS", "Qwen2VLForConditionalGeneration")
    assert not rules.is_generative_architecture("_SPECULATIVE_DECODING_EXAMPLE_MODELS", "DFlashDraftModel")
    assert not rules.is_generative_architecture("_TRANSFORMERS_BACKEND_MODELS", "TransformersForCausalLM")


def test_a_pooling_head_is_not_generative_even_in_a_generative_table() -> None:
    assert not rules.is_generative_architecture("_MULTIMODAL_EXAMPLE_MODELS", "JinaVLForRanking")
    tagger = "Qwen3ASRForcedAlignerForTokenClassification"
    assert not rules.is_generative_architecture("_MULTIMODAL_EXAMPLE_MODELS", tagger)
    assert rules.is_generative_architecture("_TEXT_GENERATION_EXAMPLE_MODELS", "PanguEmbeddedForCausalLM")


def test_embedding_ranking_and_classification_pipelines_are_not_generative() -> None:
    for tag in ("feature-extraction", "sentence-similarity", "text-ranking", "text-classification"):
        assert not rules.is_generative_pipeline(tag)
    assert not rules.is_generative_pipeline("token-classification")
    for tag in ("text-generation", "image-text-to-text", "any-to-any", "automatic-speech-recognition", None):
        assert rules.is_generative_pipeline(tag)


def test_checkpoints_created_before_2025_are_left_out_unless_a_registry_names_them() -> None:
    assert rules.passes_date_rule(date(2025, 1, 1), registry_named=False)
    assert not rules.passes_date_rule(date(2024, 12, 31), registry_named=False)
    assert rules.passes_date_rule(date(2024, 12, 31), registry_named=True)
    assert not rules.passes_date_rule(None, registry_named=False)


def test_a_listing_is_kept_when_it_is_no_copy_no_older_than_2025_and_generative() -> None:
    kept = Listed("Qwen/Qwen3-8B", date(2025, 4, 27), 10, (), "text-generation")
    assert rules.admits_listing(kept)
    assert not rules.admits_listing(Listed("Qwen/Qwen3-8B-FP8", date(2025, 4, 28), 10, (), "text-generation"))
    assert not rules.admits_listing(Listed("Qwen/Qwen2.5-7B-Instruct", date(2024, 9, 16), 10, (), "text-generation"))
    assert not rules.admits_listing(Listed("Qwen/Qwen3-Embedding-8B", date(2025, 6, 3), 10, (), "feature-extraction"))
    assert not rules.admits_listing(Listed("openai/gpt-oss-120b", date(2025, 8, 4), 10, (), "text-generation"))


def test_the_listing_rules_keep_every_qwen_chat_checkpoint_of_2026_10_06() -> None:
    kept = {m.id for m in qwen_listing() if rules.admits_listing(m)}
    assert qwen_chat() <= kept
    assert len(kept) == 112  # 48 more for the details to decide: base checkpoints, speech, image models
    assert {"Qwen/Qwen3-8B-Base", "Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen-Image-2512"} <= kept - qwen_chat()


def test_details_are_kept_for_a_registered_architecture_with_a_chat_template() -> None:
    registered = {"Qwen3ForCausalLM", "Qwen3VLForConditionalGeneration"}
    assert rules.admits_details(details(), registered)
    assert not rules.admits_details(details(architectures=("QwenImagePipeline",)), registered)
    assert not rules.admits_details(details(architectures=()), registered)
    assert not rules.admits_details(details(chat_template=False), registered)
    renamed = details(id="example/renamed", architectures=("GptOssForCausalLM",))
    assert not rules.admits_details(renamed, registered | {"GptOssForCausalLM"})


def test_a_current_chat_checkpoint_ships_a_template_dates_from_2025_and_is_no_copy() -> None:
    assert rules.is_current_chat(details())
    assert not rules.is_current_chat(details(chat_template=False))
    assert not rules.is_current_chat(details(created=date(2024, 9, 16)))
    assert not rules.is_current_chat(details(id="Qwen/Qwen2.5-Omni-7B-AWQ", created=date(2025, 5, 14)))
    assert not rules.is_current_chat(None)


def test_the_maintainers_models_are_tier_one_whatever_their_date() -> None:
    for model in (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "MiniMaxAI/MiniMax-M3",
        "zai-org/GLM-5.3-Flash",
        "moonshotai/Kimi-K3",
    ):
        assert rules.tier_of(model, None, BUILT, current_chat=False) == 1
    assert rules.tier_of("tencent/Hy4-preview", date(2020, 1, 1), BUILT, current_chat=False) == 1


def test_a_qwen_checkpoint_is_tier_one_only_when_it_is_a_current_chat_checkpoint() -> None:
    assert rules.tier_of("Qwen/Qwen3-8B", date(2025, 4, 27), BUILT, current_chat=True) == 1
    assert rules.tier_of("Qwen/Qwen2.5-0.5B-Instruct", date(2024, 9, 16), BUILT, current_chat=False) == 3
    assert rules.tier_of("Qwen/Qwen3.8-27B-FP8", date(2026, 8, 13), BUILT, current_chat=False) == 2
    assert rules.tier_of("QwenLM/Qwen3-8B", date(2025, 4, 27), BUILT, current_chat=True) == 3


def test_tier_two_is_the_twelve_months_before_the_list_is_built() -> None:
    assert rules.tier_of("example/a", date(2025, 10, 6), BUILT, current_chat=True) == 2
    assert rules.tier_of("example/a", date(2025, 10, 5), BUILT, current_chat=True) == 3
    assert rules.tier_of("example/a", None, BUILT, current_chat=False) == 3
    assert rules.tier_of("example/a", date(2027, 2, 28), date(2028, 2, 29), current_chat=False) == 2


def test_tier_one_follows_the_maintainers_order_and_the_other_tiers_follow_downloads() -> None:
    rows = [
        ("tencent/Hy4-preview", 1, 5),
        ("Qwen/Qwen3-8B", 1, 900),
        ("Qwen/Qwen3.5-27B", 1, 950),
        ("zai-org/GLM-5.3-Flash", 1, 1),
        ("MiniMaxAI/MiniMax-M3", 1, 2),
        ("deepseek-ai/DeepSeek-V4.1-Flash", 1, 3),
        ("example/b", 2, 10),
        ("example/a", 2, 10),
        ("example/c", 2, 99),
        ("example/unknown", 3, None),
        ("example/zero", 3, 0),
        ("example/some", 3, 7),
    ]
    ordered = [model for model, tier, downloads in sorted(rows, key=lambda r: rules.order_key(*r))]
    assert ordered == [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "MiniMaxAI/MiniMax-M3",
        "zai-org/GLM-5.3-Flash",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3-8B",
        "tencent/Hy4-preview",
        "example/c",
        "example/a",
        "example/b",
        "example/some",
        "example/unknown",
        "example/zero",
    ]


def test_the_status_says_why_nothing_can_be_recorded_yet() -> None:
    assert rules.status_of(None, checked=False) == "unchecked"
    assert rules.status_of(None, checked=True) == "not-on-hub"
    assert rules.status_of(details(gated=True, chat_template=False), checked=True) == "gated"
    assert rules.status_of(details(chat_template=False), checked=True) == "no-chat-template"
    assert rules.status_of(details(), checked=True) == "pending"


def test_a_tokenizer_config_that_could_not_be_read_gives_its_own_status_not_no_chat_template() -> None:
    assert rules.status_of(details(chat_template=False, unread="hub-error-503"), checked=True) == "hub-error-503"
    invalid = details(chat_template=False, unread="invalid-tokenizer-config")
    assert rules.status_of(invalid, checked=True) == "invalid-tokenizer-config"
    out_of_reach = details(gated=True, chat_template=False, unread="hub-error-403")
    assert rules.status_of(out_of_reach, checked=True) == "gated"
