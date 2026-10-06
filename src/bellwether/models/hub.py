"""What the list needs from the Hugging Face Hub, behind a small interface so tests can use a fake."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

LISTING_FIELDS = ["createdAt", "downloads", "tags", "pipeline_tag"]
TEMPLATE_FILES = frozenset({"chat_template.jinja", "chat_template.json"})
RATE_LIMITED = 429
ATTEMPTS = 8  # the Hub counts requests in five-minute windows; eight waits outlast one
DEFAULT_WAIT = 60  # seconds, when a rate-limited answer carries no Retry-After


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
    chat_template: bool
    gated: bool


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
        return Details(
            id=info.id,
            sha=info.sha,
            created=_day(info.created_at),
            downloads=info.downloads or 0,
            tags=tuple(info.tags or ()),
            architectures=tuple(config.get("architectures") or ()),
            chat_template=self._ships_template(info, config),
            gated=bool(info.gated),
        )

    def _ships_template(self, info: Any, config: dict) -> bool:
        """A template file, the template the Hub shows from the tokenizer config, or that config itself."""
        from huggingface_hub.errors import GatedRepoError

        files = {sibling.rfilename for sibling in info.siblings or ()}
        if files & TEMPLATE_FILES or any(name.startswith("additional_chat_templates/") for name in files):
            return True
        if (config.get("tokenizer_config") or {}).get("chat_template"):
            return True
        if "tokenizer_config.json" not in files:
            return False
        try:
            path = self._patiently(lambda: self.download(info.id, "tokenizer_config.json", revision=info.sha))
        except GatedRepoError:
            return False  # out of reach without access; the row's status says gated
        return bool(json.loads(Path(path).read_text()).get("chat_template"))

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


def _day(moment: datetime | None) -> date | None:
    return moment.date() if moment else None
