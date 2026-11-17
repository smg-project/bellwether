"""The draws a template makes: Jinja's ``random`` filter reads Python's global generator at each render, so a template
that draws (one checkpoint's writes a two-character nonce after every role) renders one conversation differently each
time. The round trip compares two renders of one conversation, the generation prompt and the conversation with the
turn appended, and no turn extends a prompt that was drawn apart from it. So its renders are made under equal draws:
the generator starts from one fixed value for the render and is given back as it was, and nothing else in the process
sees the pin. It is the clock pin's twin (``clock``): no recorded line may carry what the recording process happened
to read. The value itself is any fixed one: the output is the turn after the prompt, and for a template that draws
only where it opens a turn no drawn value reaches a recorded case (two different values record byte-equal sets)."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager

SEED = 0


@contextmanager
def equal_draws() -> Iterator[None]:
    """A render inside this block draws from the generator's fixed start. The generator's own state is restored on
    the way out, whatever the render raises."""
    state = random.getstate()
    random.seed(SEED)
    try:
        yield
    finally:
        random.setstate(state)
