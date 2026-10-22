"""The day a template renders: transformers' ``strftime_now`` reads the clock at each render, so a template that
writes the day into the prompt (Apertus, the Granite line, a Llama 3.2 variant, among others) gives a reference that
changes with the day the recorder ran. With ``SOURCE_DATE_EPOCH`` set, the reproducible-builds convention, the clock
those templates read is that instant, in UTC, and the provenance names it; without it the real day renders, as
before."""

from __future__ import annotations

import os
from datetime import UTC, datetime

ENV = "SOURCE_DATE_EPOCH"


def pinned() -> datetime | None:
    """The instant ``SOURCE_DATE_EPOCH`` names, as a naive UTC datetime (the shape ``datetime.now()`` returns), or
    none when the variable is unset or empty."""
    value = os.environ.get(ENV, "").strip()
    if not value:
        return None
    return datetime.fromtimestamp(int(value), tz=UTC).replace(tzinfo=None)


def pin_template_clock() -> datetime | None:
    """Point the clock transformers' ``strftime_now`` reads at the pinned instant, and return it; none, and nothing
    changed, when ``SOURCE_DATE_EPOCH`` is unset. ``chat_template_utils`` looks ``datetime`` up in its own module at
    each call, so templates compiled before the pin render the pinned day too."""
    instant = pinned()
    if instant is None:
        return None
    from transformers.utils import chat_template_utils

    class Pinned(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return instant if tz is None else instant.replace(tzinfo=UTC).astimezone(tz)

    chat_template_utils.datetime = Pinned
    return instant
