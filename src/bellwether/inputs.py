"""Oracle inputs: every file the reference oracle reads for a checkpoint, each with its sha256.

Checkpoints whose oracle inputs are equal render and parse identically, so they form one checkpoint group and are
recorded once (docs/benchmark-sets.md, "Which models"). The inputs are:

- the tokenizer files, whichever the checkpoint has (``TOKENIZER_FILES``);
- the chat template, inside ``tokenizer_config.json`` or in ``chat_template.jinja`` or ``chat_template.json``, and the
  named templates in ``additional_chat_templates/``, each ``<name>.jinja`` there, of which transformers takes
  ``tool_use`` when a request has tools;
- from ``chat_template.json``, only its ``chat_template`` value, since no oracle reads the file's bytes: vLLM reads it
  through the processor, parsed, and transformers' tokenizer does not open it; from ``processor_config.json`` the same
  key, which transformers' processor takes before any template file, and ``processor_class`` and ``auto_map``, the two
  keys that decide whether transformers builds the checkpoint's processor, which decides whose template the render
  reference resolves for a request without tools, as the engine's server does (``bellwether.record.reference``); the
  file is an input only when it carries one of the three;
- the token ids the end of a turn depends on (``STOP_IDS``), so that sampling defaults do not split a group: from
  ``generation_config.json``, or, for a checkpoint that ships none, from ``config.json`` as
  ``GenerationConfig.from_model_config`` reads it, ``text_config`` included;
- from ``config.json``, the two fields that choose the tokenizer class;
- for a checkpoint whose tokenizer class is the vendor's own code (``tokenizer_config.json`` names it in ``auto_map``),
  every Python file and vocabulary file at the repository's root (``VENDOR_SUFFIXES``): the class may import any of
  them, and the vendor-code oracle runs it in a sandbox that holds only files the manifest lists;
- for a checkpoint that ships no chat template but a prompt encoder, the vendor's module under ``encoding/`` whose
  ``encode_messages`` renders the chat format (DeepSeek V3.2 and V4; ``ENCODER_PATTERN``): the engine's server renders
  such a checkpoint with its own copy of that module, never with a template, and the vendor-code oracle runs the
  checkpoint's own in the sandbox, as the engine's tokenizer mode for the architecture calls it.

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
    "processor_config.json": ("chat_template", "processor_class", "auto_map"),
    "generation_config.json": STOP_IDS,
    "config.json": ("model_type", "tokenizer_class"),
}
FILES = tuple(dict.fromkeys((*TOKENIZER_FILES, *TEMPLATE_FILES, *NARROWED)))
PATTERNS = (*FILES, f"{NAMED_TEMPLATES}/*.jinja")
# The files at the root of a checkpoint whose tokenizer class is the vendor's code that are inputs as well: its Python
# modules, which the class may import, and the vocabulary formats such classes read.
VENDOR_SUFFIXES = (".py", ".model", ".tiktoken", ".vocab", ".bpe")
VENDOR_PATTERNS = tuple(f"*{suffix}" for suffix in VENDOR_SUFFIXES)
# The prompt encoder of a checkpoint that ships no chat template: the vendor's module under ``encoding/`` named
# ``encoding*.py`` (its tests there are ``test_encoding*.py``, never fetched), the one that defines ``encode_messages``.
ENCODER_DIR = "encoding"
ENCODER_PATTERN = f"{ENCODER_DIR}/encoding*.py"
ENCODER_ENTRY = "def encode_messages("
# The engine's tokenizer mode per architecture the engine's server renders with a copy of the checkpoint's prompt
# encoder, whatever else the checkpoint ships (``vllm/config/model.py:709-717`` at v0.31.0): such a checkpoint's
# reference is its encoder even when it ships a template too (DeepSeek-V4.1-Flash ships both).
ENCODER_MODES = {
    "DeepseekV32ForCausalLM": "deepseek_v32",
    "DeepseekV4ForCausalLM": "deepseek_v4",
    "DeepseekV4ForConditionalGeneration": "deepseek_v4",
    "DeepseekV41ForCausalLM": "deepseek_v41",
}


def oracle_inputs(model: str, revision: str) -> dict[str, str]:
    """``{file: sha256}`` for each oracle input the checkpoint has at ``revision``, sorted by file name; a named
    template is keyed by its path, ``additional_chat_templates/<name>.jinja``."""
    directory = checkpoint_dir(model, revision)
    names = [name for name in FILES if (directory / name).is_file() and _is_input(directory / name)]
    if (directory / NAMED_TEMPLATES).is_dir():
        names += [
            f"{NAMED_TEMPLATES}/{path.name}"
            for path in (directory / NAMED_TEMPLATES).iterdir()
            if path.is_file() and path.suffix == ".jinja"
        ]
    if vendor_tokenizer(directory):
        names += [
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix in VENDOR_SUFFIXES and path.name not in names
        ]
    encoder = vendor_encoder(directory)
    if encoder is not None:
        names.append(encoder)
    stop_ids_from_config = "generation_config.json" not in names
    return {name: _sha256(directory / name, stop_ids_from_config) for name in sorted(names)}


def vendor_tokenizer(directory: Path) -> bool:
    """Whether the checkpoint's tokenizer class is the vendor's own code: ``tokenizer_config.json`` names an
    ``AutoTokenizer`` in its ``auto_map``. Read as JSON; nothing in the repository is run."""
    path = directory / "tokenizer_config.json"
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_bytes())
    except ValueError:
        return False
    auto_map = config.get("auto_map") if isinstance(config, dict) else None
    return isinstance(auto_map, dict) and "AutoTokenizer" in auto_map


def ships_a_template(directory: Path) -> bool:
    """Whether the checkpoint ships a chat template anywhere the reference oracle reads one: the ``chat_template`` of
    ``tokenizer_config.json``, ``processor_config.json`` or ``chat_template.json``, a ``chat_template.jinja``, or a
    named template. Read as JSON and by name; nothing in the repository is run."""
    for name in ("tokenizer_config.json", "processor_config.json", "chat_template.json"):
        path = directory / name
        if not path.is_file():
            continue
        try:
            loaded = json.loads(path.read_bytes())
        except ValueError:
            continue
        if isinstance(loaded, dict) and loaded.get("chat_template") is not None:
            return True
    if (directory / "chat_template.jinja").is_file():
        return True
    named = directory / NAMED_TEMPLATES
    return named.is_dir() and any(path.suffix == ".jinja" for path in named.iterdir() if path.is_file())


def encoder_architecture(directory: Path) -> bool:
    """Whether ``config.json`` names an architecture the engine's server renders with the checkpoint's prompt encoder
    (``ENCODER_MODES``). Read as JSON; nothing in the repository is run."""
    path = directory / "config.json"
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_bytes())
    except ValueError:
        return False
    architectures = config.get("architectures") if isinstance(config, dict) else None
    return isinstance(architectures, list) and any(name in ENCODER_MODES for name in architectures)


def vendor_encoder(directory: Path) -> str | None:
    """The checkpoint's prompt encoder, keyed by its path (``encoding/encoding_dsv32.py``): the one module matching
    ``ENCODER_PATTERN`` whose text defines ``encode_messages``, for a checkpoint that ships no chat template or whose
    architecture the engine renders with the encoder whatever it ships (``encoder_architecture``). None for any other
    checkpoint with a template, whose template stays its reference, and for one with no such module or more than one.
    Read as text; nothing in the repository is run."""
    encoders = directory / ENCODER_DIR
    if not encoders.is_dir() or (ships_a_template(directory) and not encoder_architecture(directory)):
        return None
    found = sorted(
        path.name
        for path in encoders.glob("encoding*.py")
        if path.is_file() and ENCODER_ENTRY in path.read_text(encoding="utf-8", errors="replace")
    )
    return f"{ENCODER_DIR}/{found[0]}" if len(found) == 1 else None


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
        if vendor_tokenizer(Path(found)):
            patterns = [*PATTERNS, *VENDOR_PATTERNS]
            found = snapshot_download(model, revision=revision, allow_patterns=patterns, local_files_only=offline)
        elif not ships_a_template(Path(found)) or encoder_architecture(Path(found)):
            # A checkpoint without a template, or of an architecture the engine renders with an encoder, may ship a
            # prompt encoder: its module is fetched when there is one, and nothing else of its directory.
            patterns = [*PATTERNS, ENCODER_PATTERN]
            found = snapshot_download(model, revision=revision, allow_patterns=patterns, local_files_only=offline)
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


def _is_input(path: Path) -> bool:
    """Whether a file the checkpoint ships is one the oracle reads: every file but ``processor_config.json``, which is
    an input only when it carries one of the keys an oracle reads (its ``NARROWED`` fields: ``chat_template``, the key
    the processor's template is read from, or ``processor_class`` or ``auto_map``, which decide whether the processor
    builds); without any of them the file holds nothing an oracle reads, and the groups recorded before it was fetched
    keep their inputs."""
    if path.name != "processor_config.json":
        return True
    try:
        loaded = json.loads(path.read_bytes())
    except ValueError:
        return False
    return isinstance(loaded, dict) and any(loaded.get(key) is not None for key in NARROWED[path.name])


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
