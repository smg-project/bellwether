"""What the list needs from the Hugging Face Hub, behind a small interface so tests can use a fake."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

LISTING_FIELDS = ["createdAt", "downloads", "tags", "pipeline_tag"]
# Where transformers 5.18 takes a chat template from, under the names the Hub's listing gives the files.
# AutoTokenizer, which the hf-template oracle loads: chat_template.jinja, the *.jinja files directly
# under additional_chat_templates/, and tokenizer_config.json's "chat_template", which those files
# override (tokenization_utils_base.py L1634, L1662-1678, L1768-1771, L1784-1801). AutoProcessor reads
# those two template sources as well (processing_utils.py L1286-1303, L1340-1342) and alone reads
# chat_template.json and processor_config.json's "chat_template" (L1306-1312, L1326-1328, L1399-1402,
# L1445-1451). Mistral's tekken conversion reads chat_template.json too, but only through mistral-common
# (integrations/mistral/tokenizer.py L75-79), which this environment does not install.
TOKENIZER_TEMPLATE_FILE = "chat_template.jinja"
TEMPLATE_DIR = "additional_chat_templates/"
PROCESSOR_TEMPLATE_FILE = "chat_template.json"
PROCESSOR_CONFIG = "processor_config.json"
RATE_LIMITED = 429
ATTEMPTS = 8  # the Hub counts requests in five-minute windows; eight waits outlast one
DEFAULT_WAIT = 60  # seconds, when a rate-limited answer carries no Retry-After
# Why a tokenizer or processor config could not be read, as the row's status gives it (``Details.unread``).
INVALID_TOKENIZER_CONFIG = "invalid-tokenizer-config"  # not JSON, or not a JSON object
INVALID_PROCESSOR_CONFIG = "invalid-processor-config"
HUB_ERROR = "hub-error"  # then the HTTP status the Hub answered, else the error: hub-error-503, hub-error-read-timeout
OUT_OF_REACH = frozenset({f"{HUB_ERROR}-401", f"{HUB_ERROR}-403"})  # without access, as a gated model's files are


@dataclass(frozen=True)
class Listed:
    """One model as its organization's listing shows it: enough to rule it out without a second call."""

    id: str
    created: date | None
    downloads: int  # over the last 30 days, as the Hub counts them
    tags: tuple[str, ...]
    pipeline_tag: str | None


@dataclass(frozen=True)
class Details:
    """One model's present state: the commit a row pins, what its config names, and what it ships."""

    id: str  # the Hub's own spelling; a renamed repository answers with its new id
    sha: str
    created: date | None
    downloads: int
    tags: tuple[str, ...]
    architectures: tuple[str, ...]  # ``config.json``'s ``architectures``
    chat_template: bool  # False also when a config could not be read; ``unread`` then says why
    gated: bool  # the Hub flags it gated, or a config it holds answered 401 or 403
    unread: str | None = None  # why the tokenizer or processor config could not be read, as a status
    processor_only: bool = False  # only the processor's files hold the template, which the oracle does not read yet


class Hub(Protocol):
    def list_models(self, org: str) -> list[Listed]: ...

    def model(self, model_id: str) -> Details | None:
        """The model's details, or ``None`` when the Hub has no such model."""
        ...


class HfHub:
    """The Hub through ``huggingface_hub``, imported only when the Hub is asked.

    A token (``HF_TOKEN`` or a saved login) raises the rate limits and opens gated models' files;
    the list needs none. A call the Hub turns away for its rate limit waits and is made again.
    """

    def __init__(
        self, api: Any = None, download: Callable[..., str] | None = None, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        if download is None:
            from huggingface_hub import hf_hub_download

            download = hf_hub_download
        self.api, self.download, self.sleep = api, download, sleep

    def list_models(self, org: str) -> list[Listed]:
        models = self._patiently(lambda: list(self.api.list_models(author=org, expand=LISTING_FIELDS)))
        return [Listed(m.id, _day(m.created_at), m.downloads or 0, tuple(m.tags or ()), m.pipeline_tag) for m in models]

    def model(self, model_id: str) -> Details | None:
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            info = self._patiently(lambda: self.api.model_info(model_id))
        except RepositoryNotFoundError:
            return None
        config = info.config or {}
        files = {sibling.rfilename for sibling in info.siblings or ()}
        template, processor_only = self._tokenizer_template(info, config, files), False
        if template is False:  # nothing for the oracle; the processor's files may still hold one
            template = self._processor_template(info, files)
            processor_only = template is True
        unread = template if isinstance(template, str) else None
        return Details(
            id=info.id,
            sha=info.sha,
            created=_day(info.created_at),
            downloads=info.downloads or 0,
            tags=tuple(info.tags or ()),
            architectures=tuple(config.get("architectures") or ()),
            chat_template=template is True,
            gated=bool(info.gated) or unread in OUT_OF_REACH,
            unread=unread,
            processor_only=processor_only,
        )

    def _tokenizer_template(self, info: Any, config: dict, files: set[str]) -> bool | str:
        """What AutoTokenizer reads: a template file, the template the Hub shows from the tokenizer config,
        or that config itself; when that config cannot be read, the status that says why instead.
        """
        if TOKENIZER_TEMPLATE_FILE in files or any(name.startswith(TEMPLATE_DIR) for name in files):
            return True
        if (config.get("tokenizer_config") or {}).get("chat_template"):
            return True
        if "tokenizer_config.json" not in files:
            return False
        tokenizer_config = self._json_config(info, "tokenizer_config.json", INVALID_TOKENIZER_CONFIG)
        return tokenizer_config if isinstance(tokenizer_config, str) else bool(tokenizer_config.get("chat_template"))

    def _processor_template(self, info: Any, files: set[str]) -> bool | str:
        """What only AutoProcessor reads: the legacy template file, or the processor config's template."""
        if PROCESSOR_TEMPLATE_FILE in files:
            return True
        if PROCESSOR_CONFIG not in files:
            return False
        processor_config = self._json_config(info, PROCESSOR_CONFIG, INVALID_PROCESSOR_CONFIG)
        return processor_config if isinstance(processor_config, str) else bool(processor_config.get("chat_template"))

    def _json_config(self, info: Any, filename: str, invalid: str) -> dict | str:
        """The JSON object in a config file at the pinned sha, or the status that says why it could not be read.

        A Hub or network error never stops the run, and is never taken for a config without a template.
        """
        import httpx
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        try:
            path = self._patiently(lambda: self.download(info.id, filename, revision=info.sha))
        except (HfHubHTTPError, LocalEntryNotFoundError, httpx.HTTPError) as err:
            return _hub_error(err)
        try:
            content = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError:  # not JSON, or not UTF-8, which transformers reads it as
            return invalid
        return content if isinstance(content, dict) else invalid

    def _patiently(self, call: Callable[[], T]) -> T:
        from huggingface_hub.errors import HfHubHTTPError

        for _ in range(ATTEMPTS - 1):
            try:
                return call()
            except HfHubHTTPError as err:
                if err.response.status_code != RATE_LIMITED:
                    raise
                self.sleep(_retry_after(err.response.headers.get("Retry-After", "")))
        return call()


def _retry_after(header: str) -> int:
    return int(header) if header.isdigit() else DEFAULT_WAIT


def _hub_error(err: Exception) -> str:
    """``hub-error-`` and the HTTP status the Hub answered, else the error's name as status words.

    ``hf_hub_download`` raises ``LocalEntryNotFoundError`` from what its HEAD request met, after
    its own retries; that cause is the error to name.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    cause = err.__cause__ if isinstance(err, LocalEntryNotFoundError) and err.__cause__ else err
    response = getattr(cause, "response", None)
    name = str(response.status_code) if response is not None else type(cause).__name__
    return f"{HUB_ERROR}-{re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '-', name).lower()}"


def _day(moment: datetime | None) -> date | None:
    return moment.date() if moment else None
