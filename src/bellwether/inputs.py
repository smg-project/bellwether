"""Oracle inputs: every file the reference oracle reads for a checkpoint, each with its sha256.

Checkpoints whose oracle inputs are equal render and parse identically, so they form one checkpoint group and are
recorded once (docs/benchmark-sets.md, "Which models"). The inputs are:

- the tokenizer files, whichever the checkpoint has (``TOKENIZER_FILES``);
- the chat template, inside ``tokenizer_config.json`` or in ``chat_template.jinja`` or ``chat_template.json``, and the
  named templates in ``additional_chat_templates/``, each ``<name>.jinja`` there, of which transformers takes
  ``tool_use`` when a request has tools;
- from ``chat_template.json``, only its ``chat_template`` value, since no oracle reads the file's bytes: vLLM reads it
  through the processor, parsed, and transformers' tokenizer does not open it;
- the token ids the end of a turn depends on (``STOP_IDS``), so that sampling defaults do not split a group: from
  ``generation_config.json``, or, for a checkpoint that ships none, from ``config.json`` as
  ``GenerationConfig.from_model_config`` reads it, ``text_config`` included;
- from ``config.json``, the two fields that choose the tokenizer class.

A whole file is hashed as its bytes. A narrowed file is hashed over the canonical JSON of only its fields,
``json.dumps(..., sort_keys=True)`` with a field the file lacks written as null, so anyone can recompute it; for a
checkpoint without ``generation_config.json``, ``config.json``'s fields are joined by the three token ids
``from_model_config`` gives.

A Hub checkpoint is read from the Hugging Face cache at the exact revision, through ``huggingface_hub``, which fetches
only these files, and only those the cache lacks; no file from the repository is run. Offline (``HF_HUB_OFFLINE=1``),
the cached snapshot stands for the checkpoint only when the cache also holds the commit's file list, which a download
with network access leaves (``trees/<commit>.json``): with it, a snapshot that lacks a listed file is refused, and a
file the list does not name is one the checkpoint does not ship. Without it, a file the snapshot lacks could be either,
so the checkpoint is refused rather than read with a file missing from its inputs. A model given as a directory is read
as it is, as the reference oracle reads it, and the revision is unused.
"""

from __future__ import annotations

import copy
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
NAMED_TEMPLATES = "additional_chat_templates"  # transformers' CHAT_TEMPLATE_DIR
STOP_IDS = ("bos_token_id", "eos_token_id", "pad_token_id")
NARROWED = {
    "chat_template.json": ("chat_template",),
    "generation_config.json": STOP_IDS,
    "config.json": ("model_type", "tokenizer_class"),
}
FILES = tuple(dict.fromkeys((*TOKENIZER_FILES, *TEMPLATE_FILES, *NARROWED)))
PATTERNS = (*FILES, f"{NAMED_TEMPLATES}/*.jinja")


def oracle_inputs(model: str, revision: str) -> dict[str, str]:
    """``{file: sha256}`` for each oracle input the checkpoint has at ``revision``, sorted by file name; a named
    template is keyed by its path, ``additional_chat_templates/<name>.jinja``."""
    directory = checkpoint_dir(model, revision)
    names = [name for name in FILES if (directory / name).is_file()]
    if (directory / NAMED_TEMPLATES).is_dir():
        names += [
            f"{NAMED_TEMPLATES}/{path.name}"
            for path in (directory / NAMED_TEMPLATES).iterdir()
            if path.is_file() and path.suffix == ".jinja"
        ]
    stop_ids_from_config = "generation_config.json" not in names
    return {name: _sha256(directory / name, stop_ids_from_config) for name in sorted(names)}


def checkpoint_dir(model: str, revision: str) -> Path:
    """The directory holding the checkpoint's oracle inputs at ``revision``: the model itself when it is a directory,
    else its snapshot in the Hugging Face cache."""
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import constants, get_cached_repo_tree, snapshot_download
    from huggingface_hub.errors import CachedRepoTreeNotFoundError

    offline = constants.is_offline_mode()
    try:
        # Offline, the snapshot is read from the cache: asked for a commit, the library would otherwise ask the Hub
        # for the commit's file list whenever the cache does not hold it, and fail. With the list cached, it refuses a
        # snapshot that lacks a listed file (IncompleteSnapshotError, a FileNotFoundError).
        found = snapshot_download(model, revision=revision, allow_patterns=list(PATTERNS), local_files_only=offline)
        if offline:
            get_cached_repo_tree(model, revision=revision)
    except CachedRepoTreeNotFoundError as err:
        raise FileNotFoundError(
            f"{model} at {revision}: the cache holds no list of the commit's files, so a file its snapshot lacks "
            "cannot be told from one the checkpoint does not ship; read it once with network access, which caches "
            "the list and fetches what the snapshot lacks"
        ) from err
    except FileNotFoundError as err:
        raise FileNotFoundError(f"{model} at {revision}: {err}") from err
    return Path(found)


def _sha256(path: Path, stop_ids_from_config: bool) -> str:
    data = path.read_bytes()
    fields = NARROWED.get(path.name)
    if fields is not None:
        try:
            loaded = json.loads(data)
        except ValueError as err:  # JSONDecodeError, and UnicodeDecodeError for bytes that are not text
            raise ValueError(f"{path}: not valid JSON: {err}") from err
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: not a JSON object")
        narrowed = {key: loaded.get(key) for key in fields}
        if path.name == "config.json" and stop_ids_from_config:
            narrowed |= stop_ids_of_model_config(path, loaded)
        data = json.dumps(narrowed, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stop_ids_of_model_config(path: Path, values: dict) -> dict[str, object]:
    """The token ids ``GenerationConfig.from_model_config`` gives for ``config.json``, read without vendor code.

    This is the rule the end of turn follows (#43: ``generation_eos_ids`` and ``model_config`` in
    ``bellwether.record.roundtrip``): the model config it is given carries the defaults of the class transformers
    has for its ``model_type``, as vLLM's does, and ``from_model_config`` takes a value the top level leaves unset from
    ``text_config`` (or ``decoder``, ``generator``). A model type transformers does not know is read as written, since
    its class is the vendor's code, named by ``auto_map``, which bellwether never runs.
    """
    from transformers import CONFIG_MAPPING, GenerationConfig

    try:
        model_type = values.get("model_type")
        if isinstance(model_type, str) and model_type in CONFIG_MAPPING:
            config = CONFIG_MAPPING[model_type].from_dict(copy.deepcopy(values))
        else:
            config = copy.deepcopy(values)
        generation = GenerationConfig.from_model_config(config)
    except Exception as err:  # whatever the file holds, the checkpoint is named and nothing is hashed
        raise ValueError(
            f"{path}: GenerationConfig.from_model_config cannot read it: {type(err).__name__}: {err}"
        ) from err
    return {key: getattr(generation, key) for key in STOP_IDS}
