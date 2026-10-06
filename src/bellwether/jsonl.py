"""JSON Lines as every reader here takes them: one JSON value per line, and only "\\n" ends a line.

``json.dumps`` with ``ensure_ascii=False``, as the corpus and fixture writers call it, writes U+2028, U+2029 and U+0085
raw inside a string, and ``str.splitlines()`` would end a line at each of them. The corpus and fixture readers and the
importers' dataset readers all read through ``loads``. It is a top-level module so that both sides can import it:
importing anything under ``bellwether.record`` runs its ``__init__``, which loads the oracles and their third-party
dependencies, and the importers import nothing beyond the standard library. Nor does this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def loads(text: str, where: str | Path) -> Iterator[tuple[int, Any]]:
    """Each line's JSON value with its 1-based line number, in order; a blank line is skipped.

    Only "\\n" ends a line, so the empty piece after a final "\\n" is a blank line too. A line that is not JSON raises
    ``ValueError`` naming ``where`` (the file the text came from) and the line's number.
    """
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"{where}:{number}: not JSON: {err}") from err
        yield number, value
