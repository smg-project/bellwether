"""Tests for building the list from registry entries and a fake Hub, and for its JSON Lines form."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from bellwether.models.build import Row, hub_rows, missing_from_tier1, registry_only_rows, set_aside, to_jsonl, unnamed
from bellwether.models.hub import Details, HubUnavailable, Listed
from bellwether.models.registry import Entry, read_sglang, read_vllm

DATA = Path(__file__).parent / "data" / "models"
BUILT = date(2026, 10, 6)
TEXT = "_TEXT_GENERATION_EXAMPLE_MODELS"
MULTIMODAL = "_MULTIMODAL_EXAMPLE_MODELS"
PAGE = "docs/docs/supported-models/generative_models.mdx"
MULTIMODAL_PAGE = "docs/docs/supported-models/multimodal_language_models.mdx"


def excerpt_entries() -> list[Entry]:
    entries = read_vllm((DATA / "vllm-registry-excerpt.py.txt").read_text())
    for name in ("generative_models", "multimodal_language_models", "diffusion_language_models"):
        source = (DATA / f"sglang-{name}-excerpt.mdx").read_text()
        entries.extend(read_sglang(source, f"docs/docs/supported-models/{name}.mdx"))
    return entries


def test_registry_only_lists_every_generative_checkpoint_once_without_asking_the_hub() -> None:
    rows = registry_only_rows(excerpt_entries(), BUILT)
    models = [row.model for row in rows]
    assert len(models) == len(set(models)) == 28  # 25 checkpoints, and 3 entries that name none
    assert not {"lmsys/gpt-oss-20b-bf16", "openai/gpt-oss-20b", "openai/gpt-oss-120b"} & set(models)
    assert not {"jinaai/jina-reranker-m0", "Qwen/Qwen3-ForcedAligner-0.6B", "MrLight/dse-qwen2-2b-mrl-v1"} & set(models)
    assert "Qwen/Qwen3-4B" not in models  # a speculative-decoding target, not a generative entry
    assert {row.status for row in rows} == {"unchecked", "no-checkpoint-named"}
    assert {(row.revision, row.created, row.downloads) for row in rows} == {(None, None, None)}
    assert models[:2] == ["deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3-Flash"]
    # Without the Hub only Simo's named models have a tier; a row with no checkpoint is the rest.
    assert [row.tier for row in rows] == [1, 1] + [None] * 23 + [3] * 3
    assert models[2:25] == sorted(models[2:25])  # without downloads, the id orders the rows


def test_a_checkpoint_any_registry_calls_multimodal_is_multimodal() -> None:
    rows = {row.model: row for row in registry_only_rows(excerpt_entries(), BUILT)}
    assert rows["zai-org/GLM-5.3-Flash"].modality == "multimodal"  # vLLM names it in both tables
    assert rows["Qwen/Qwen3-ASR-1.7B"].modality == "multimodal"
    assert rows["inclusionAI/LLaDA2.0-flash"].modality == "text"
    assert sum(row.modality == "multimodal" for row in rows.values()) == 11  # JetVLM and FunAudioChat among them


def test_one_checkpoint_named_by_both_engines_is_one_row_with_both_sources() -> None:
    entries = [
        Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-0.6B",)),
        Entry("sglang", "Qwen", PAGE, True, False, ("Qwen/Qwen3-0.6B",)),
    ]
    [row] = registry_only_rows(entries, BUILT)
    assert row.sources == ("sglang", "vllm")


def test_names_that_are_no_checkpoint_are_reported() -> None:
    assert unnamed(excerpt_entries()) == [
        "vllm Qwen4ExpForCausalLM: '' is not a Hugging Face id",
        "vllm FunAudioChatForConditionalGeneration: 'funaudiochat' is not a Hugging Face id",
        "sglang JetVLM: no checkpoint named",
    ]


def test_the_gpt_oss_checkpoints_set_aside_are_named_whether_the_name_or_the_architecture_says_so() -> None:
    entries = [
        Entry("vllm", "GptOssForCausalLM", TEXT, True, False, ("lmsys/gpt-oss-20b-bf16", "example/oss-renamed-120b")),
        Entry("vllm", "InternVLChatModel", MULTIMODAL, True, True, ("OpenGVLab/InternVL3_5-1B",)),
        Entry("vllm", "InternVLChatModel", MULTIMODAL, True, True, ("OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview",)),
        Entry("sglang", "GPT-OSS", PAGE, True, False, ("openai/gpt-oss-20b", "openai/gpt-oss-120b")),
    ]
    assert set_aside(entries) == [
        "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview",
        "example/oss-renamed-120b",  # only its architecture says so
        "lmsys/gpt-oss-20b-bf16",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    assert [row.model for row in registry_only_rows(entries, BUILT)] == ["OpenGVLab/InternVL3_5-1B"]


def test_simos_models_the_list_lacks_are_named() -> None:
    rows = registry_only_rows(excerpt_entries(), BUILT)
    assert missing_from_tier1(rows) == ["MiniMaxAI/MiniMax-M3", "tencent/Hy4-preview"]


def details(model: str, created: date, downloads: int, architectures: tuple[str, ...], **flags) -> Details:
    flags = {"chat_template": True, "gated": False, **flags}
    return Details(model, f"sha-of-{model}", created, downloads, (), architectures, **flags)


def listed(model: str, created: date, pipeline_tag: str = "text-generation") -> Listed:
    return Listed(model, created, 0, (), pipeline_tag)


class FakeHub:
    """The Hub as two dicts; it records what was asked so the tests can see which calls were saved.

    ``unavailable`` maps a model or an organization to the error the Hub answers for it.
    """

    def __init__(
        self, listings: dict[str, list[Listed]], models: dict[str, Details], unavailable: dict[str, str] | None = None
    ) -> None:
        self.listings, self.models, self.unavailable = listings, models, unavailable or {}
        self.listed: list[str] = []
        self.asked: list[str] = []
        self.loads: dict[str, tuple[str | None, str | None]] = {}  # the revision and tokenizer each was asked with

    def list_models(self, org: str) -> list[Listed]:
        self.listed.append(org)
        if org in self.unavailable:
            raise HubUnavailable(org, self.unavailable[org])
        return self.listings.get(org, [])

    def model(self, model_id: str, revision: str | None = None, tokenizer: str | None = None) -> Details | None:
        self.asked.append(model_id)
        self.loads[model_id] = (revision, tokenizer)
        if model_id in self.unavailable:
            raise HubUnavailable(model_id, self.unavailable[model_id])
        return self.models.get(model_id)


ENTRIES = [
    Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-8B",)),
    Entry("vllm", "Qwen2ForCausalLM", TEXT, True, False, ("Qwen/Qwen2.5-0.5B-Instruct",)),
    Entry("vllm", "Qwen3_5ForConditionalGeneration", MULTIMODAL, True, True, ("Qwen/Qwen3.5-0.8B",)),
    Entry("vllm", "Glm5NextForCausalLM", TEXT, True, False, ("zai-org/GLM-5.3-Flash",)),
    Entry("vllm", "Glm5NextForConditionalGeneration", MULTIMODAL, True, True, ("zai-org/GLM-5.3-Flash",)),
    Entry("vllm", "ChatGLMModel", TEXT, True, False, ("THUDM/chatglm3-6b",)),
    Entry("vllm", "GPT2LMHeadModel", TEXT, True, False, ("openai-community/gpt2",)),
    Entry("vllm", "Gemma3ForCausalLM", TEXT, True, False, ("google/gemma-3-1b-it",)),
    Entry("vllm", "GptOssForCausalLM", TEXT, True, False, ("lmsys/gpt-oss-20b-bf16",)),
    Entry("vllm", "Qwen2VLForConditionalGeneration", "_EMBEDDING_EXAMPLE_MODELS", False, False, ("MrLight/dse",)),
    Entry("vllm", "GoneForCausalLM", TEXT, True, False, ("example/gone",)),
    Entry("sglang", "ChatGLM", PAGE, True, False, ("zai-org/chatglm3-6b",)),
    Entry("sglang", "XVERSE (MoE)", PAGE, True, False, ("xverse/XVERSE-MoE-A36B",)),
    Entry("sglang", "GPT-OSS", PAGE, True, False, ("openai/gpt-oss-20b",)),
    Entry("sglang", "OSS (renamed)", PAGE, True, False, ("example/oss-named-120b",)),
]

CHATGLM = details("zai-org/chatglm3-6b", date(2023, 10, 25), 40, ("ChatGLMModel",))
MODELS = {
    "Qwen/Qwen3-8B": details("Qwen/Qwen3-8B", date(2025, 4, 27), 900, ("Qwen3ForCausalLM",)),
    "Qwen/Qwen2.5-0.5B-Instruct": details("Qwen/Qwen2.5-0.5B-Instruct", date(2024, 9, 16), 300, ("Qwen2ForCausalLM",)),
    "Qwen/Qwen3.5-0.8B": details("Qwen/Qwen3.5-0.8B", date(2026, 2, 28), 500, ("Qwen3_5ForConditionalGeneration",)),
    "zai-org/GLM-5.3-Flash": details(
        "zai-org/GLM-5.3-Flash", date(2026, 9, 1), 50, ("Glm5NextForConditionalGeneration",)
    ),
    "THUDM/chatglm3-6b": CHATGLM,  # renamed: the Hub answers with the new id
    "zai-org/chatglm3-6b": CHATGLM,
    "openai-community/gpt2": details(
        "openai-community/gpt2", date(2022, 3, 2), 9000, ("GPT2LMHeadModel",), chat_template=False
    ),
    "google/gemma-3-1b-it": details("google/gemma-3-1b-it", date(2025, 3, 10), 700, ("Gemma3ForCausalLM",), gated=True),
    "xverse/XVERSE-MoE-A36B": details("xverse/XVERSE-MoE-A36B", date(2024, 4, 1), 10, ("XverseMoeForCausalLM",)),
    "Qwen/Qwen3.5-27B": details("Qwen/Qwen3.5-27B", date(2026, 2, 24), 950, ("Qwen3_5ForConditionalGeneration",)),
    "Qwen/Qwen3-8B-Base": details(
        "Qwen/Qwen3-8B-Base", date(2025, 4, 28), 100, ("Qwen3ForCausalLM",), chat_template=False
    ),
    "Qwen/Qwen-Image": details("Qwen/Qwen-Image", date(2025, 8, 2), 800, ()),
    "zai-org/GLM-4.7-Flash": details("zai-org/GLM-4.7-Flash", date(2026, 1, 19), 400, ("Glm4MoeLiteForCausalLM",)),
    "xverse/XVERSE-MoE-A4.2B-Chat": details(
        "xverse/XVERSE-MoE-A4.2B-Chat", date(2026, 3, 1), 5, ("XverseMoeForCausalLM",)
    ),
    "example/oss-tune-20b": details("example/oss-tune-20b", date(2026, 1, 1), 1, ("GptOssForCausalLM",)),
    "example/oss-named-120b": details("example/oss-named-120b", date(2026, 1, 1), 1, ("GptOssForCausalLM",)),
}
LISTINGS = {
    "Qwen": [
        listed("Qwen/Qwen3-8B", date(2025, 4, 27)),
        listed("Qwen/Qwen3-8B-FP8", date(2025, 4, 28)),
        listed("Qwen/Qwen3.5-27B", date(2026, 2, 24), "image-text-to-text"),
        listed("Qwen/Qwen2.5-7B-Instruct", date(2024, 9, 16)),
        listed("Qwen/Qwen3-Embedding-8B", date(2025, 6, 3), "feature-extraction"),
        listed("Qwen/Qwen3-8B-Base", date(2025, 4, 28)),
        listed("Qwen/Qwen-Image", date(2025, 8, 2), "text-to-image"),
    ],
    "zai-org": [
        listed("zai-org/GLM-5.3-Flash", date(2026, 9, 1)),
        listed("zai-org/chatglm3-6b", date(2023, 10, 25)),
        listed("zai-org/GLM-4.7-Flash", date(2026, 1, 19)),
    ],
    "xverse": [
        listed("xverse/XVERSE-MoE-A36B", date(2024, 4, 1)),
        listed("xverse/XVERSE-MoE-A4.2B-Chat", date(2026, 3, 1)),
    ],
    "example": [listed("example/oss-tune-20b", date(2026, 1, 1))],
}


def build() -> tuple[list[Row], FakeHub, list[str]]:
    hub, log = FakeHub(LISTINGS, MODELS), []
    return hub_rows(ENTRIES, hub, BUILT, log.append), hub, log


def test_the_hub_list_holds_the_registries_checkpoints_and_the_organizations_chat_checkpoints() -> None:
    rows, _, _ = build()
    assert [(row.model, row.tier) for row in rows] == [
        ("zai-org/GLM-5.3-Flash", 1),
        ("Qwen/Qwen3.5-27B", 1),
        ("Qwen/Qwen3-8B", 1),
        ("Qwen/Qwen3.5-0.8B", 1),
        ("xverse/XVERSE-MoE-A4.2B-Chat", 2),
        ("openai-community/gpt2", 3),
        ("google/gemma-3-1b-it", 3),
        ("Qwen/Qwen2.5-0.5B-Instruct", 3),
        ("zai-org/chatglm3-6b", 3),
        ("xverse/XVERSE-MoE-A36B", 3),
        ("example/gone", 3),
    ]


def test_each_row_pins_the_hubs_sha_and_says_why_nothing_can_be_recorded_yet() -> None:
    rows = {row.model: row for row in build()[0]}
    assert rows["Qwen/Qwen3-8B"] == Row(
        "Qwen/Qwen3-8B", "sha-of-Qwen/Qwen3-8B", 1, "pending", date(2025, 4, 27), 900, "text", ("vllm",)
    )
    assert rows["Qwen/Qwen3.5-27B"].sources == ("hub",)
    assert rows["Qwen/Qwen3.5-27B"].modality == "multimodal"
    assert rows["xverse/XVERSE-MoE-A4.2B-Chat"].modality == "text"
    assert rows["zai-org/GLM-5.3-Flash"].modality == "multimodal"
    assert (rows["example/gone"].status, rows["example/gone"].revision) == ("not-on-hub", None)
    assert rows["google/gemma-3-1b-it"].status == "gated"
    assert rows["openai-community/gpt2"].status == "no-chat-template"


def test_a_renamed_checkpoint_is_one_row_under_the_hubs_id() -> None:
    rows = {row.model: row for row in build()[0]}
    assert "THUDM/chatglm3-6b" not in rows
    assert rows["zai-org/chatglm3-6b"].sources == ("sglang", "vllm")


def test_only_the_registries_organizations_are_listed_and_ruled_out_listings_cost_no_call() -> None:
    _, hub, log = build()
    assert hub.listed == ["Qwen", "example", "google", "openai-community", "xverse", "zai-org"]
    never_asked = {"Qwen/Qwen3-8B-FP8", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-Embedding-8B", "lmsys/gpt-oss-20b-bf16"}
    assert not never_asked & set(hub.asked)
    assert "openai/gpt-oss-20b" not in hub.asked
    assert log[0] == "registries: 9 checkpoints, 1 not on the Hub"  # merged, gpt-oss left out
    assert "Qwen: 7 listed, 1 added" in log
    assert hub.asked.count("Qwen/Qwen3-8B") == hub.asked.count("zai-org/GLM-5.3-Flash") == 1


def test_an_sglang_only_architecture_is_registered_through_its_checkpoints_config() -> None:
    rows = [row.model for row in build()[0]]
    assert "xverse/XVERSE-MoE-A4.2B-Chat" in rows  # XverseMoeForCausalLM is in no vLLM table here
    assert "zai-org/GLM-4.7-Flash" not in rows  # no registry names its architecture
    assert "Qwen/Qwen3-8B-Base" not in rows  # ships no chat template
    assert "Qwen/Qwen-Image" not in rows  # its config names no architecture
    assert "example/oss-tune-20b" not in rows  # gpt-oss, as only its architecture says
    assert "example/oss-named-120b" not in rows  # the same, named by a registry


def test_a_hub_model_whose_tokenizer_config_could_not_be_read_is_a_row_that_says_why() -> None:
    def qwen(name: str, **flags) -> Details:
        return details(f"Qwen/{name}", date(2025, 4, 28), 10, ("Qwen3ForCausalLM",), **flags)

    models = {
        "Qwen/Qwen3-8B": qwen("Qwen3-8B"),
        "Qwen/Qwen3-14B": qwen("Qwen3-14B", chat_template=False, unread="hub-error-503"),
        "Qwen/Qwen3-32B": qwen("Qwen3-32B", chat_template=False, unread="invalid-tokenizer-config"),
        "Qwen/Qwen3-4B": qwen("Qwen3-4B", chat_template=False, gated=True, unread="hub-error-401"),
        "Qwen/Qwen3-1.7B": qwen("Qwen3-1.7B", chat_template=False, gated=True),  # read with access: it ships none
    }
    entries = [Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-8B",))]
    hub = FakeHub({"Qwen": [listed(model, date(2025, 4, 28)) for model in models]}, models)
    statuses = {row.model: row.status for row in hub_rows(entries, hub, BUILT)}
    assert statuses == {
        "Qwen/Qwen3-8B": "pending",
        "Qwen/Qwen3-14B": "hub-error-503",
        "Qwen/Qwen3-32B": "invalid-tokenizer-config",
        "Qwen/Qwen3-4B": "gated",
    }


def test_a_hub_model_whose_template_only_the_processor_reads_is_kept_with_a_status_that_says_so() -> None:
    def qwen(name: str, **flags) -> Details:
        return details(f"Qwen/{name}", date(2025, 4, 28), 10, ("Qwen3VLForConditionalGeneration",), **flags)

    models = {
        "Qwen/Qwen3-VL-8B": qwen("Qwen3-VL-8B"),
        "Qwen/Qwen3-VL-4B": qwen("Qwen3-VL-4B", processor_only=True),
        "Qwen/Qwen3-VL-2B": qwen("Qwen3-VL-2B", processor_only=True, gated=True),
    }
    entries = [Entry("vllm", "Qwen3VLForConditionalGeneration", MULTIMODAL, True, True, ("Qwen/Qwen3-VL-8B",))]
    hub = FakeHub({"Qwen": [listed(model, date(2025, 4, 28)) for model in models]}, models)
    statuses = {row.model: row.status for row in hub_rows(entries, hub, BUILT)}
    assert statuses == {
        "Qwen/Qwen3-VL-8B": "pending",
        "Qwen/Qwen3-VL-4B": "processor-chat-template",
        "Qwen/Qwen3-VL-2B": "gated",
    }


def test_a_registry_entry_that_names_no_checkpoint_is_a_row_under_its_name() -> None:
    entries = [
        Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-8B",)),
        Entry("vllm", "Qwen4ExpForCausalLM", TEXT, True, False, ("",)),
        Entry("vllm", "FunAudioChatForConditionalGeneration", MULTIMODAL, True, True, ("funaudiochat",)),
        Entry("sglang", "JetVLM", MULTIMODAL_PAGE, True, True, ()),
        Entry("sglang", "JetVLM", MULTIMODAL_PAGE, True, True, ()),  # the page has two such rows
        Entry("vllm", "JinaVLForRanking", MULTIMODAL, False, True, ("",)),  # not generative: no row
    ]
    expected = [
        Row(
            "FunAudioChatForConditionalGeneration", None, 3, "no-checkpoint-named", None, None, "multimodal", ("vllm",)
        ),
        Row("JetVLM", None, 3, "no-checkpoint-named", None, None, "multimodal", ("sglang",)),
        Row("Qwen4ExpForCausalLM", None, 3, "no-checkpoint-named", None, None, "text", ("vllm",)),
    ]
    offline = registry_only_rows(entries, BUILT)
    assert [row for row in offline if row.status == "no-checkpoint-named"] == expected
    hub = FakeHub({}, {"Qwen/Qwen3-8B": details("Qwen/Qwen3-8B", date(2025, 4, 27), 900, ("Qwen3ForCausalLM",))})
    online = hub_rows(entries, hub, BUILT)
    assert [row for row in online if row.status == "no-checkpoint-named"] == expected
    assert hub.asked == ["Qwen/Qwen3-8B"]


def test_without_the_hub_a_checkpoint_vllm_loads_with_vendor_code_says_so_and_keeps_vllms_revision() -> None:
    kimi = ("moonshotai/Kimi-K3", "moonshotai/Kimi-K3-Base")
    ernie = ("baidu/ERNIE-4.5-VL-28B-A3B-PT",)
    entries = [
        Entry("vllm", "KimiK3ForCausalLM", TEXT, True, False, kimi, vendor_code=True),
        Entry("vllm", "Ernie4_5_VLMoeForConditionalGeneration", MULTIMODAL, True, True, ernie, True, "refs/pr/17"),
        Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-8B",)),
    ]
    rows = {row.model: (row.status, row.revision) for row in registry_only_rows(entries, BUILT)}
    assert rows == {
        "moonshotai/Kimi-K3": ("needs-vendor-code", None),
        "moonshotai/Kimi-K3-Base": ("needs-vendor-code", None),  # an extra is loaded the same way
        "baidu/ERNIE-4.5-VL-28B-A3B-PT": ("needs-vendor-code", "refs/pr/17"),  # the revision vLLM loads
        "Qwen/Qwen3-8B": ("unchecked", None),
    }


def test_with_the_hub_a_row_is_read_at_vllms_revision_with_vllms_tokenizer_and_vendor_code_is_said() -> None:
    moondream, hcx, gemma = (
        "moondream/moondream3-preview",
        "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        "google/gemma-3-1b-it",
    )
    pinned = "a6cdfd3464d1b767259cad23e164eaf39d3e3960"
    entries = [
        Entry(
            "vllm", "Moondream3ForCausalLM", MULTIMODAL, True, True, (moondream,), True, None, "moondream/starmie-v1"
        ),
        Entry("vllm", "HCXVisionV2ForCausalLM", MULTIMODAL, True, True, (hcx,), True, pinned),
        Entry("vllm", "Gemma3ForCausalLM", TEXT, True, False, (gemma,), vendor_code=True),
        Entry("vllm", "Phi3SmallForCausalLM", TEXT, True, False, ("microsoft/Phi-3-small-8k-instruct",), True),
    ]
    models = {
        moondream: details(moondream, date(2025, 9, 1), 5, ("Moondream3ForCausalLM",)),
        hcx: details(hcx, date(2025, 12, 1), 5, ("HCXVisionV2ForCausalLM",), chat_template=False),
        gemma: details(gemma, date(2025, 3, 10), 5, ("Gemma3ForCausalLM",), gated=True),
    }
    hub = FakeHub({}, models)
    rows = {row.model: row.status for row in hub_rows(entries, hub, BUILT)}
    assert rows == {
        moondream: "needs-vendor-code",
        hcx: "needs-vendor-code",  # before what its template says
        gemma: "gated",  # out of reach comes first
        "microsoft/Phi-3-small-8k-instruct": "not-on-hub",
    }
    assert hub.loads[moondream] == (None, "moondream/starmie-v1")
    assert hub.loads[hcx] == (pinned, None)


def test_a_registry_checkpoint_the_hub_could_not_be_asked_about_keeps_its_row_with_the_error() -> None:
    entries = [Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-8B",))]
    hub = FakeHub({}, {}, unavailable={"Qwen/Qwen3-8B": "hub-error-503"})
    [row] = hub_rows(entries, hub, BUILT)
    assert row == Row("Qwen/Qwen3-8B", None, None, "hub-error-503", None, None, "text", ("vllm",))  # no tier to give


def test_a_listed_checkpoint_or_organization_the_hub_fails_on_is_logged_and_the_run_goes_on() -> None:
    entries = [
        Entry("vllm", "Qwen3ForCausalLM", TEXT, True, False, ("Qwen/Qwen3-8B",)),
        Entry("vllm", "XverseMoeForCausalLM", TEXT, True, False, ("xverse/XVERSE-MoE-A36B",)),
    ]
    qwen3 = details("Qwen/Qwen3-8B", date(2025, 4, 27), 900, ("Qwen3ForCausalLM",))
    xverse = details("xverse/XVERSE-MoE-A36B", date(2024, 4, 1), 10, ("XverseMoeForCausalLM",))
    later = details("Qwen/Qwen3-14B", date(2025, 4, 28), 800, ("Qwen3ForCausalLM",))
    listings = {"Qwen": [listed("Qwen/Qwen3-14B", date(2025, 4, 28)), listed("Qwen/Qwen3-32B", date(2025, 4, 28))]}
    models = {"Qwen/Qwen3-8B": qwen3, "xverse/XVERSE-MoE-A36B": xverse, "Qwen/Qwen3-14B": later}
    hub = FakeHub(listings, models, unavailable={"Qwen/Qwen3-32B": "hub-error-502", "xverse": "hub-error-500"})
    log: list[str] = []
    rows = hub_rows(entries, hub, BUILT, log.append)
    assert [row.model for row in rows] == ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "xverse/XVERSE-MoE-A36B"]
    assert "Qwen/Qwen3-32B: left out, hub-error-502" in log
    assert "xverse: not listed, hub-error-500" in log
    assert hub.listed == ["Qwen", "xverse"]


def test_the_list_is_canonical_json_lines() -> None:
    rows = [
        Row("Qwen/Qwen3-8B", "b968826d", 1, "pending", date(2025, 4, 27), 900, "text", ("vllm",)),
        Row("example/gone", None, 3, "not-on-hub", None, None, "text", ("sglang", "vllm")),
    ]
    text = to_jsonl(rows)
    assert text == (
        '{"model":"Qwen/Qwen3-8B","revision":"b968826d","tier":1,"status":"pending","created":"2025-04-27",'
        '"downloads":900,"modality":"text","sources":["vllm"]}\n'
        '{"model":"example/gone","revision":null,"tier":3,"status":"not-on-hub","created":null,'
        '"downloads":null,"modality":"text","sources":["sglang","vllm"]}\n'
    )
    assert [json.loads(line)["model"] for line in text.splitlines()] == ["Qwen/Qwen3-8B", "example/gone"]


def test_the_organizations_listed_are_those_of_each_architectures_example_not_of_vllms_test_extras() -> None:
    entries = [
        Entry("vllm", "MixtralForCausalLM", TEXT, True, False, ("mistralai/Mixtral-8x7B-v0.1", "TitanML/tiny-mixtral")),
        Entry("vllm", "InklingForCausalLM", TEXT, True, False, ("thinkingmachines/Inkling-NVFP4",)),
    ]
    names = ("mistralai/Mixtral-8x7B-v0.1", "TitanML/tiny-mixtral", "thinkingmachines/Inkling-NVFP4")
    hub = FakeHub({}, {name: details(name, date(2025, 1, 1), 1, ("MixtralForCausalLM",)) for name in names})
    rows = hub_rows(entries, hub, BUILT)
    assert {row.model for row in rows} == set(names)  # an extra is still a row
    assert hub.listed == ["mistralai", "thinkingmachines"]  # a default that is a copy still names its publisher
