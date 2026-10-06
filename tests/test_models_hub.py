"""Tests for the Hub client over a stub of ``HfApi``: the Hub's JSON goes in, nothing touches a network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from huggingface_hub import ModelInfo
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from bellwether.models.hub import Details, HfHub, Listed


def http_error(kind: type[HfHubHTTPError], status: int, **headers: str) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/api/models/example")
    return kind(f"{status} from the Hub", response=httpx.Response(status, headers=headers, request=request))


class StubApi:
    """Answers ``list_models`` and ``model_info`` from the Hub's JSON, failing first where told to."""

    def __init__(self, models=None, listings=None, failures=None) -> None:
        self.models = models or {}
        self.listings = listings or {}
        self.failures = list(failures or [])
        self.calls: list[tuple] = []

    def _fail_first(self) -> None:
        if self.failures:
            raise self.failures.pop(0)

    def list_models(self, **kwargs):
        self.calls.append(("list_models", kwargs))
        self._fail_first()
        return iter(ModelInfo(**m) for m in self.listings.get(kwargs["author"], []))

    def model_info(self, repo_id: str, **kwargs):
        self.calls.append(("model_info", repo_id))
        self._fail_first()
        if repo_id not in self.models:
            raise http_error(RepositoryNotFoundError, 404)
        return ModelInfo(**self.models[repo_id])


class Downloads:
    """Stands in for ``hf_hub_download``: writes the given tokenizer config and records each call."""

    def __init__(self, tmp_path: Path, tokenizer_config: dict | Exception) -> None:
        self.tmp_path = tmp_path
        self.tokenizer_config = tokenizer_config
        self.calls: list[tuple] = []

    def __call__(self, repo_id: str, filename: str, revision: str) -> str:
        self.calls.append((repo_id, filename, revision))
        if isinstance(self.tokenizer_config, Exception):
            raise self.tokenizer_config
        path = self.tmp_path / filename
        path.write_text(json.dumps(self.tokenizer_config))
        return str(path)


def no_download(*args, **kwargs) -> str:
    raise AssertionError(f"downloaded {args} {kwargs}")


def no_wait(seconds: float) -> None:
    raise AssertionError(f"waited {seconds}s for an answer that was not a rate limit")


def client(api: StubApi, download=no_download) -> HfHub:
    """A client that fails at once where it would wait or download unasked, instead of hanging."""
    return HfHub(api=api, download=download, sleep=no_wait)


def info(model_id: str = "Qwen/Qwen3-8B", **fields) -> dict:
    """``model_info``'s JSON for a model whose template sits in its tokenizer config."""
    return {
        "id": model_id,
        "sha": "b968826d9c46dd6066d109eabc6255188de91218",
        "createdAt": "2025-04-27T03:40:08.000Z",
        "downloads": 6512345,
        "tags": ["transformers", "safetensors", "qwen3", "text-generation", "conversational"],
        "pipeline_tag": "text-generation",
        "gated": False,
        "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
        "siblings": [{"rfilename": "config.json"}, {"rfilename": "tokenizer_config.json"}],
        **fields,
    }


def test_a_listing_gives_created_date_downloads_tags_and_pipeline_tag() -> None:
    listed = info("Qwen/Qwen3.5-27B", createdAt="2026-02-24T08:00:00.000Z", downloads=42)
    api = StubApi(listings={"Qwen": [listed]})
    assert client(api).list_models("Qwen") == [
        Listed(
            "Qwen/Qwen3.5-27B",
            date(2026, 2, 24),
            42,
            ("transformers", "safetensors", "qwen3", "text-generation", "conversational"),
            "text-generation",
        )
    ]
    [(_, kwargs)] = api.calls
    assert kwargs["author"] == "Qwen"
    assert set(kwargs["expand"]) == {"createdAt", "downloads", "tags", "pipeline_tag"}


def test_details_give_the_hubs_own_id_the_sha_and_the_configs_architectures(tmp_path: Path) -> None:
    renamed = info("zai-org/chatglm2-6b", config={"architectures": ["ChatGLMModel"]}, gated="manual")
    hub = client(StubApi(models={"THUDM/chatglm2-6b": renamed}), Downloads(tmp_path, {}))
    assert hub.model("THUDM/chatglm2-6b") == Details(
        id="zai-org/chatglm2-6b",
        sha="b968826d9c46dd6066d109eabc6255188de91218",
        created=date(2025, 4, 27),
        downloads=6512345,
        tags=("transformers", "safetensors", "qwen3", "text-generation", "conversational"),
        architectures=("ChatGLMModel",),
        chat_template=False,
        gated=True,
    )


def test_a_model_the_hub_does_not_have_has_no_details() -> None:
    assert client(StubApi()).model("example/missing") is None


@pytest.mark.parametrize(
    "files", [["chat_template.jinja"], ["chat_template.json"], ["additional_chat_templates/tool_use.jinja"]]
)
def test_a_template_file_is_a_chat_template(files: list[str]) -> None:
    siblings = [{"rfilename": name} for name in ["config.json", "tokenizer_config.json", *files]]
    assert client(StubApi(models={"Qwen/Qwen3-8B": info(siblings=siblings)})).model("Qwen/Qwen3-8B").chat_template


def test_a_template_the_hub_reads_from_the_tokenizer_config_needs_no_download() -> None:
    config = {"architectures": ["Qwen3ForCausalLM"], "tokenizer_config": {"chat_template": "{{ messages }}"}}
    assert client(StubApi(models={"Qwen/Qwen3-8B": info(config=config)})).model("Qwen/Qwen3-8B").chat_template


def test_the_tokenizer_config_is_read_at_the_pinned_sha_when_nothing_else_tells(tmp_path: Path) -> None:
    sha = "b968826d9c46dd6066d109eabc6255188de91218"
    with_template = Downloads(tmp_path, {"chat_template": "{{ messages }}", "eos_token": "<|im_end|>"})
    assert client(StubApi(models={"Qwen/Qwen3-8B": info()}), with_template).model("Qwen/Qwen3-8B").chat_template
    assert with_template.calls == [("Qwen/Qwen3-8B", "tokenizer_config.json", sha)]
    without = client(StubApi(models={"Qwen/Qwen3-8B": info()}), Downloads(tmp_path, {"eos_token": "<|im_end|>"}))
    assert not without.model("Qwen/Qwen3-8B").chat_template


def test_a_model_without_a_tokenizer_config_ships_no_template() -> None:
    diffusers = info("Qwen/Qwen-Image", config={}, siblings=[{"rfilename": "model_index.json"}])
    details = client(StubApi(models={"Qwen/Qwen-Image": diffusers})).model("Qwen/Qwen-Image")
    assert not details.chat_template
    assert details.architectures == ()


def test_a_gated_tokenizer_config_out_of_reach_counts_as_no_template(tmp_path: Path) -> None:
    gated = info("google/gemma-3-4b-it", gated="manual")
    out_of_reach = Downloads(tmp_path, http_error(GatedRepoError, 401))
    details = client(StubApi(models={"google/gemma-3-4b-it": gated}), out_of_reach).model("google/gemma-3-4b-it")
    assert details.gated and not details.chat_template


def test_the_hub_is_asked_again_after_a_rate_limit() -> None:
    waits: list[float] = []
    limited = http_error(HfHubHTTPError, 429, **{"Retry-After": "7"})
    api = StubApi(listings={"Qwen": [info()]}, failures=[limited, http_error(HfHubHTTPError, 429)])
    hub = HfHub(api=api, download=no_download, sleep=waits.append)
    assert [m.id for m in hub.list_models("Qwen")] == ["Qwen/Qwen3-8B"]
    assert waits == [7, 60]  # the Hub's Retry-After when it sends one, else a minute
    assert len(api.calls) == 3


def test_other_hub_errors_are_not_retried() -> None:
    api = StubApi(failures=[http_error(HfHubHTTPError, 500)])
    with pytest.raises(HfHubHTTPError):
        client(api).list_models("Qwen")
    assert len(api.calls) == 1
