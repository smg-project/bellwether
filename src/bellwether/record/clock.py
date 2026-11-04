"""The day a template renders: transformers' ``strftime_now`` reads the clock at each render, so a template that
writes the day into the prompt (Apertus, the Granite line, a Llama 3.2 variant, among others) would give a reference
that changes with the day the recorder ran. The clock those templates read is one fixed instant, in UTC, and the
provenance names it: ``SOURCE_DATE_EPOCH`` when it is set (the reproducible-builds convention), else the corpus's
recording day, ``RECORDING_DAY``. No reference carries the day the recorder ran."""

from __future__ import annotations

import os
from datetime import UTC, datetime

ENV = "SOURCE_DATE_EPOCH"
# Noon UTC of 2026-10-14, the day the dated groups' references are recorded at; noon, so the day is the same in
# whichever zone a reader formats it. The smg fixture job renders the gateway's side at the same value, so a move of
# this day re-records every dated set and moves the job's value with it.
RECORDING_DAY = "1791979200"


def pinned_value() -> str:
    """``SOURCE_DATE_EPOCH`` as set, stripped; ``RECORDING_DAY`` when the variable is unset or blank."""
    return os.environ.get(ENV, "").strip() or RECORDING_DAY


def pinned() -> datetime:
    """The instant ``pinned_value`` names, as a naive UTC datetime (the shape ``datetime.now()`` returns)."""
    return datetime.fromtimestamp(int(pinned_value()), tz=UTC).replace(tzinfo=None)


def pin_template_clock() -> datetime:
    """Point the clock transformers' ``strftime_now`` reads at the pinned instant, and return it.
    ``chat_template_utils`` looks ``datetime`` up in its own module at each call, so templates compiled before the pin
    render the pinned day too."""
    instant = pinned()
    from transformers.utils import chat_template_utils

    class Pinned(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return instant if tz is None else instant.replace(tzinfo=UTC).astimezone(tz)

    chat_template_utils.datetime = Pinned
    return instant
