"""Tests for the Hub client over a stub of ``HfApi``: the Hub's JSON goes in, nothing touches a network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from huggingface_hub import ModelInfo
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from bellwether.models.hub import Details, HfHub, HubUnavailable, Listed
from bellwether.models.rules import status_of

REQUEST = httpx.Request("GET", "https://huggingface.co/api/models/example")
TEMPLATE = {"chat_template": "{{ messages }}"}


def http_error(kind: type[HfHubHTTPError], status: int, **headers: str) -> HfHubHTTPError:
    return kind(f"{status} from the Hub", response=httpx.Response(status, headers=headers, request=REQUEST))


def unreachable(cause: Exception) -> LocalEntryNotFoundError:
    """What ``hf_hub_download`` raises when its HEAD request fails and the file is not cached."""
    error = LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")
    error.__cause__ = cause
    return error


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
    """Stands in for ``hf_hub_download``: writes each config it is given, as JSON or as raw bytes."""

    def __init__(
        self,
        tmp_path: Path,
        tokenizer_config: dict | bytes | Exception | None,
        processor_config: dict | bytes | Exception | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.files = {"tokenizer_config.json": tokenizer_config, "processor_config.json": processor_config}
        self.calls: list[tuple] = []

    def __call__(self, repo_id: str, filename: str, revision: str) -> str:
        self.calls.append((repo_id, filename, revision))
        content = self.files[filename]
        if content is None:
            raise AssertionError(f"downloaded {filename}, which the repository does not list")
        if isinstance(content, Exception):
            raise content
        path = self.tmp_path / filename
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(json.dumps(content))
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


@pytest.mark.parametrize("files", [["chat_template.jinja"], ["additional_chat_templates/tool_use.jinja"]])
def test_a_template_file_is_a_chat_template(files: list[str]) -> None:
    siblings = [{"rfilename": name} for name in ["config.json", "tokenizer_config.json", *files]]
    assert client(StubApi(models={"Qwen/Qwen3-8B": info(siblings=siblings)})).model("Qwen/Qwen3-8B").chat_template


@pytest.mark.parametrize(
    ("files", "tokenizer_config", "processor_config", "status", "downloaded"),
    [
        (["chat_template.jinja", "chat_template.json"], None, None, "pending", []),
        (["tokenizer_config.json", "chat_template.json"], TEMPLATE, None, "pending", ["tokenizer_config.json"]),
        (
            ["tokenizer_config.json", "chat_template.json"],
            {},
            None,
            "processor-chat-template",
            ["tokenizer_config.json"],
        ),
        (["chat_template.json"], None, None, "processor-chat-template", []),
        (
            ["tokenizer_config.json", "processor_config.json"],
            {},
            TEMPLATE,
            "processor-chat-template",
            ["tokenizer_config.json", "processor_config.json"],
        ),
        (
            ["tokenizer_config.json", "processor_config.json"],
            {},
            {"image_seq_length": 256},
            "no-chat-template",
            ["tokenizer_config.json", "processor_config.json"],
        ),
        (["processor_config.json"], None, b"{", "invalid-processor-config", ["processor_config.json"]),
        (
            ["processor_config.json"],
            None,
            unreachable(http_error(HfHubHTTPError, 503)),
            "hub-error-503",
            ["processor_config.json"],
        ),
        (
            ["tokenizer_config.json", "chat_template.json"],
            unreachable(http_error(HfHubHTTPError, 503)),
            None,
            "hub-error-503",
            ["tokenizer_config.json"],
        ),
    ],
    ids=[
        "both-template-files",
        "tokenizer-config-and-processor-file",
        "processor-file-only",
        "processor-file-and-no-tokenizer-config",
        "processor-config-only",
        "processor-config-without-a-template",
        "processor-config-not-json",
        "processor-config-unread",
        "tokenizer-config-unread-beside-a-processor-file",
    ],
)
def test_a_template_only_the_processor_reads_has_its_own_status_and_the_tokenizers_is_read_first(
    tmp_path: Path,
    files: list[str],
    tokenizer_config: dict | bytes | Exception | None,
    processor_config: dict | bytes | Exception | None,
    status: str,
    downloaded: list[str],
) -> None:
    siblings = [{"rfilename": name} for name in ["config.json", *files]]
    downloads = Downloads(tmp_path, tokenizer_config, processor_config)
    hub = client(StubApi(models={"Qwen/Qwen3-VL-8B": info("Qwen/Qwen3-VL-8B", siblings=siblings)}), downloads)
    assert status_of(hub.model("Qwen/Qwen3-VL-8B"), checked=True) == status
    assert [filename for _, filename, _ in downloads.calls] == downloaded


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


def test_a_gated_tokenizer_config_out_of_reach_is_unread_not_a_missing_template(tmp_path: Path) -> None:
    gated = info("google/gemma-3-4b-it", gated="manual")
    out_of_reach = Downloads(tmp_path, http_error(GatedRepoError, 401))
    details = client(StubApi(models={"google/gemma-3-4b-it": gated}), out_of_reach).model("google/gemma-3-4b-it")
    assert (details.gated, details.chat_template, details.unread) == (True, False, "hub-error-401")


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (http_error(HfHubHTTPError, 403), "hub-error-403"),  # refused on the download itself
        (http_error(RepositoryNotFoundError, 401), "hub-error-401"),  # how huggingface_hub reports a 401 on a file
        (unreachable(http_error(HfHubHTTPError, 403)), "hub-error-403"),  # refused on the HEAD request before it
    ],
)
def test_a_tokenizer_config_answering_401_or_403_is_out_of_reach_as_a_gated_ones_is(
    tmp_path: Path, error: Exception, status: str
) -> None:
    hub = client(StubApi(models={"Qwen/Qwen3-8B": info(gated=False)}), Downloads(tmp_path, error))
    details = hub.model("Qwen/Qwen3-8B")
    assert (details.gated, details.chat_template, details.unread) == (True, False, status)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (http_error(HfHubHTTPError, 500), "hub-error-500"),
        (unreachable(http_error(HfHubHTTPError, 503)), "hub-error-503"),  # what is left after the HEAD's retries
        (unreachable(httpx.ConnectError("connection refused", request=REQUEST)), "hub-error-connect-error"),
        (httpx.ReadTimeout("timed out", request=REQUEST), "hub-error-read-timeout"),
        (http_error(RemoteEntryNotFoundError, 404), "hub-error-404"),  # listed, then not found at the same sha
    ],
)
def test_any_other_error_reading_the_tokenizer_config_is_named_not_counted_as_no_template(
    tmp_path: Path, error: Exception, status: str
) -> None:
    hub = client(StubApi(models={"Qwen/Qwen3-8B": info()}), Downloads(tmp_path, error))
    details = hub.model("Qwen/Qwen3-8B")
    assert (details.gated, details.chat_template, details.unread) == (False, False, status)


@pytest.mark.parametrize(
    "text",
    [b'{"chat_template": "{{ messages }}"', b"[]", b'"{{ messages }}"', b"\xff\xfe{}"],
    ids=["cut-short", "array", "string", "not-utf-8"],
)
def test_a_tokenizer_config_that_is_not_a_json_object_is_invalid_not_without_a_template(
    tmp_path: Path, text: bytes
) -> None:
    hub = client(StubApi(models={"Qwen/Qwen3-8B": info()}), Downloads(tmp_path, text))
    details = hub.model("Qwen/Qwen3-8B")
    assert (details.chat_template, details.unread) == (False, "invalid-tokenizer-config")


def test_the_hub_is_asked_again_after_a_rate_limit() -> None:
    waits: list[float] = []
    limited = http_error(HfHubHTTPError, 429, **{"Retry-After": "7"})
    api = StubApi(listings={"Qwen": [info()]}, failures=[limited, http_error(HfHubHTTPError, 429)])
    hub = HfHub(api=api, download=no_download, sleep=waits.append)
    assert [m.id for m in hub.list_models("Qwen")] == ["Qwen/Qwen3-8B"]
    assert waits == [7, 60]  # the Hub's Retry-After when it sends one, else a minute
    assert len(api.calls) == 3


def test_other_hub_errors_are_not_retried_and_a_failed_listing_names_its_error() -> None:
    api = StubApi(failures=[http_error(HfHubHTTPError, 500)])
    with pytest.raises(HubUnavailable) as raised:
        client(api).list_models("Qwen")
    assert raised.value.status == "hub-error-500"
    assert len(api.calls) == 1


@pytest.mark.parametrize(
    ("failures", "status"),
    [
        ([http_error(HfHubHTTPError, 503)], "hub-error-503"),
        ([httpx.ConnectError("connection refused", request=REQUEST)], "hub-error-connect-error"),
        ([http_error(HfHubHTTPError, 429)] * 8, "hub-error-429"),  # still limited after the seven waits
    ],
)
def test_a_details_request_the_hub_fails_is_unavailable_with_its_error_named(
    failures: list[Exception], status: str
) -> None:
    api = StubApi(models={"Qwen/Qwen3-8B": info()}, failures=failures)
    hub = HfHub(api=api, download=no_download, sleep=lambda seconds: None)
    with pytest.raises(HubUnavailable) as raised:
        hub.model("Qwen/Qwen3-8B")
    assert raised.value.status == status
    assert len(api.calls) == len(failures)
