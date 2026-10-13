"""Add rows to the list of checkpoints, ``fixtures/models.tsv``, keeping those whose files the Hub opens to this run.

    ROWS="$(cat rows.tsv)" python scripts/add_model_rows.py fixtures/models.tsv

The rows come from the ``ROWS`` environment variable, one ``model<TAB>revision<TAB>downloads<TAB>day<TAB>tier`` row a
line, as the list holds them. A row is added when the Hub lets the run's token (``HF_TOKEN``) read the checkpoint's
files, which a gated model's are only once its terms are accepted on the Hub with the token's account; a model the list
holds already keeps its row. The list keeps its comment lines first and its rows in the order of their model ids,
compared without case. Prints a markdown summary for the run: how many rows were added, and each row left out with why.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FIELDS = 5  # model, revision, downloads, day, tier


def why_closed(api, model: str) -> str | None:
    """Why the token cannot read the model's files, or None when it can."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        api.auth_check(model)
    except GatedRepoError:
        return "gated: the token's account has not been given access"
    except RepositoryNotFoundError:
        return "not found, or private"
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else "no answer"
        return f"hub error {status}"
    return None


def add_rows(path: Path, text: str, api) -> tuple[list[str], list[tuple[str, str]]]:
    """Add the rows of ``text`` the Hub opens to the list at ``path``: the models added, and each row left out, with
    why."""
    lines = path.read_text().splitlines()
    comments = [line for line in lines if line.startswith("#")]
    rows = [line.split("\t") for line in lines if line.strip() and not line.startswith("#")]
    listed = {row[0] for row in rows}
    added, left_out = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = line.split("\t")
        if len(row) != FIELDS:
            left_out.append((line, f"{len(row)} fields, not {FIELDS}"))
            continue
        if row[0] in listed:
            left_out.append((row[0], "listed already"))
            continue
        reason = why_closed(api, row[0])
        if reason:
            left_out.append((row[0], reason))
            continue
        rows.append(row)
        listed.add(row[0])
        added.append(row[0])
    rows.sort(key=lambda row: row[0].lower())
    path.write_text("\n".join(comments + ["\t".join(row) for row in rows]) + "\n")
    return added, left_out


def main() -> int:
    from huggingface_hub import HfApi

    added, left_out = add_rows(Path(sys.argv[1]), os.environ.get("ROWS", ""), HfApi())
    print(f"### The list\n\n{len(added)} rows added, {len(left_out)} left out.\n")
    if left_out:
        print("| model | left out because |\n|---|---|")
        print("\n".join(f"| {model} | {reason} |" for model, reason in left_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
