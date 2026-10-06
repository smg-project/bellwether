"""Oracle inputs: every file the reference oracle reads for a checkpoint, each with its sha256.

Checkpoints whose oracle inputs are equal render and parse identically, so they form one checkpoint group and are
recorded once (docs/benchmark-sets.md, "Which models"). The inputs are:

- the tokenizer files, whichever the checkpoint has (``TOKENIZER_FILES``);
- the chat template, inside ``tokenizer_config.json`` or in ``chat_template.jinja`` or ``chat_template.json``;
- from ``generation_config.json``, only the token ids, which the end of a turn depends on, so that sampling defaults do
  not split a group;
- from ``config.json``, only the two fields that choose the tokenizer class.

A whole file is hashed as its bytes. A narrowed file is hashed over the canonical JSON of only its fields,
``json.dumps(..., sort_keys=True)`` with a field the file lacks written as null, so anyone can recompute it.

A Hub checkpoint is read from the Hugging Face cache at the exact revision, through ``huggingface_hub``, which fetches
only these files, and only those the cache lacks; no file from the repository is run. Offline (``HF_HUB_OFFLINE=1``),
the cached snapshot stands for the checkpoint, as it does for the oracle, which can read nothing else offline. When
the cache also holds the commit's file list, a snapshot that lacks one of these files is refused rather than read as
complete. A model given as a directory is read as it is, as the reference oracle reads it, and the revision is unused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")
NARROWED = {
    "generation_config.json": ("bos_token_id", "eos_token_id", "pad_token_id"),
    "config.json": ("model_type", "tokenizer_class"),
}
FILES = (*TOKENIZER_FILES, *TEMPLATE_FILES, *NARROWED)


def oracle_inputs(model: str, revision: str) -> dict[str, str]:
    """``{file: sha256}`` for each oracle input the checkpoint has at ``revision``, sorted by file name."""
    directory = checkpoint_dir(model, revision)
    return {name: _sha256(directory / name) for name in sorted(FILES) if (directory / name).is_file()}


def checkpoint_dir(model: str, revision: str) -> Path:
    """The directory holding the checkpoint's oracle inputs at ``revision``: the model itself when it is a directory,
    else its snapshot in the Hugging Face cache."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import constants, snapshot_download

    try:
        # Offline, the snapshot is read from the cache as it is: asked for a commit, the library would otherwise
        # ask the Hub for the commit's file list whenever the cache does not hold it, and fail.
        found = snapshot_download(
            model, revision=revision, allow_patterns=list(FILES), local_files_only=constants.is_offline_mode()
        )
    except FileNotFoundError as err:
        raise FileNotFoundError(f"{model} at {revision}: {err}") from err
    return Path(found)


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    fields = NARROWED.get(path.name)
    if fields is not None:
        loaded = json.loads(data)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: not a JSON object")
        data = json.dumps({key: loaded.get(key) for key in fields}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
