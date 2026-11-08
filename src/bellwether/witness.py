"""bellwether witness: an independent library's reading of recorded parse cases, kept beside the reference.

A parse case's reference is the round trip through the checkpoint's own template or encoder: the message the
corpus states, rendered as the final turn, is what the recorded output must parse back to. A witness is another
reader's reading of that same recorded output. ``record`` keeps the engine witnesses (vLLM, SGLang) by id while
the request they were recorded for is unchanged; this command adds a library witness the same way.

The first library is the vendor's own request conversion and output parser for the DeepSeek V4 line
(``deepseek-recipe``, the Python distribution of the Rust crates). For every case of a parse set it converts the
recorded request, feeds the recorded output ids to its stream processor and folds the Chat Completions chunks into
one message. The message and the finish reason go into ``witnesses["deepseek-recipe"]`` with the library's version;
a case the library refuses (its request validation is the vendor API's, stricter than an engine's: a tool choice it
does not serve in thinking mode, a tool schema that is not an object) records the refusal text instead, and a case
whose reading fails records that error. The reference, the output and every other field of the line stay as they
are; ``sets.toml`` gets the set's new plain size and hash and keeps its counts and stop ids.

The library is an optional dependency (``pip install -e .[deepseek]``): without it the command says so and exits 2.
The checkpoint's ``tokenizer.json`` at the manifest's revision decodes the output ids, read from the Hugging Face
cache as the other oracles read their inputs (``bellwether.inputs.checkpoint_dir``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from bellwether.inputs import checkpoint_dir
from bellwether.manifest import Manifest, find_manifest
from bellwether.record import sets as set_tables
from bellwether.record.fixtures import read_fixture_file, write_fixture_file
from bellwether.storage import COMPRESSED_SUFFIX, is_compressed, plain_text, stem

NOT_IMPLEMENTED = 2
DEEPSEEK_RECIPE = "deepseek-recipe"
#: The manifest encoders each library reads, with the library's name for the model version.
DEEPSEEK_VERSIONS = {"deepseek_v4": "DeepseekV4Pro", "deepseek_v41": "DeepseekV41Flash"}

Reader = Callable[[dict], dict]


def deepseek_recipe_reader(manifest: Manifest) -> tuple[str, Reader]:
    """The library's version and a reader of one recorded parse case for the manifest's checkpoint.

    Raises ``ImportError`` without the library and ``ValueError`` for a checkpoint it does not read.
    """
    import deepseek_recipe as recipe

    version_name = DEEPSEEK_VERSIONS.get(manifest.encoder or "")
    if version_name is None:
        raise ValueError(
            f"{DEEPSEEK_RECIPE} reads the DeepSeek V4 and V4.1 encoders; {manifest.model} has encoder "
            f"{manifest.encoder!r}"
        )
    model_version = getattr(recipe.ModelVersion, version_name)
    options = recipe.ConversionOptions().with_model_alias(manifest.model, model_version)
    tokenizer = recipe.Tokenizer.from_file(str(checkpoint_dir(manifest.model, manifest.revision) / "tokenizer.json"))

    def read(case: dict) -> dict:
        body = library_body(case["request"], manifest.model)
        try:
            converted = recipe.ChatCompletionRequest(body).convert(options)
        except recipe.ConversionError as error:
            return {"version": recipe.__version__, "rejected": str(error)}
        generator = recipe.ChatCompletionRequest.chunk_generator(converted, case["id"], manifest.model)
        processor = recipe.StreamProcessor(generator, converted.parsing_options, tokenizer)
        response = recipe.ChatCompletionResponse(case["id"], manifest.model, 0, 0, 0)
        chunks = [
            recipe.InferenceChunk.ready(),
            recipe.InferenceChunk.tokens(case["output_ids"]),
            recipe.InferenceChunk.finish(recipe.InferenceFinishReason.Stop),
        ]
        try:
            for chunk in chunks:
                for out in processor.push(chunk):
                    response.accumulate(out)
            for out in processor.finish():
                response.accumulate(out)
        except RuntimeError as error:
            return {"version": recipe.__version__, "rejected": f"stream processor: {error}"}
        choice = json.loads(response.to_json())["choices"][0]
        message = choice["message"]
        witnessed = {"role": message.get("role", "assistant"), "content": message.get("content")}
        if message.get("reasoning_content") is not None:
            witnessed["reasoning_content"] = message["reasoning_content"]
        if message.get("tool_calls"):
            witnessed["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
                }
                for call in message["tool_calls"]
            ]
        return {"version": recipe.__version__, "message": witnessed, "finish_reason": choice["finish_reason"]}

    return recipe.__version__, read


def library_body(request: dict, model: str) -> dict:
    """The library's Chat Completions body for a recorded request: the messages and tools as recorded, the thinking
    switch and the effort the request states through its template kwargs spelled as the library's own fields."""
    body = {"model": model, "messages": request["messages"]}
    if request.get("tools"):
        body["tools"] = request["tools"]
    for key in ("tool_choice", "reasoning_effort", "thinking", "response_format"):
        if key in request:
            body[key] = request[key]
    kwargs = request.get("chat_template_kwargs") or {}
    for key in ("thinking", "enable_thinking"):
        if isinstance(kwargs.get(key), bool) and "thinking" not in body:
            body["thinking"] = {"type": "enabled" if kwargs[key] else "disabled"}
            break
    if isinstance(kwargs.get("reasoning_effort"), str) and "reasoning_effort" not in body:
        body["reasoning_effort"] = kwargs["reasoning_effort"]
    return body


LIBRARIES: dict[str, Callable[[Manifest], tuple[str, Reader]]] = {DEEPSEEK_RECIPE: deepseek_recipe_reader}


def witness_sets(
    fixtures: Path, manifest: Manifest, library: str, reader: Reader, wanted: list[str] | None = None
) -> list[str]:
    """Add ``library``'s reading to every case of the manifest's parse sets (or of ``wanted``), rewrite each set
    canonically and refresh its ``sets.toml`` table; return one summary line per set. A set Git LFS has not fetched
    is reported and left alone."""
    kind_dir = fixtures / manifest.slug / "parse"
    tables_path = fixtures / manifest.slug / set_tables.FILE
    existing = set_tables.read(tables_path)
    summaries: list[str] = []
    tables: dict[str, dict | None] = {}
    for path in sorted(p for p in kind_dir.iterdir() if p.is_file() and stem(p) is not None):
        name = stem(path)
        if wanted and name not in wanted:
            continue
        started = time.monotonic()
        try:
            cases = read_fixture_file(path)
        except ValueError as error:
            summaries.append(f"{path}: not witnessed, {error}")
            continue
        refused = 0
        for case in cases.values():
            reading = reader(case)
            if "rejected" in reading:
                refused += 1
            case.setdefault("witnesses", {})[library] = reading
        write_fixture_file(path, cases)
        old = existing.get(("parse", name), {})
        form = "zstd" if is_compressed(path) else "plain"
        generate = {field: old[field] for field in set_tables.GENERATE_FIELDS if field in old}
        tables[name] = set_tables.entry(form, plain_text(path), len(cases), old.get("rejected", 0), **generate)
        summaries.append(
            f"{path}: {len(cases)} cases witnessed by {library}, {refused} refused by it, "
            f"{time.monotonic() - started:.1f} s"
        )
    if tables:
        set_tables.update(tables_path, "parse", tables)
    return summaries


def run(args: argparse.Namespace) -> int:
    if args.kind != "parse":
        print(f"bellwether witness: kind={args.kind} is not implemented yet (parse landed first).", file=sys.stderr)
        return NOT_IMPLEMENTED
    try:
        manifest = find_manifest(args.fixtures, args.model)
    except (OSError, ValueError) as error:
        print(f"bellwether witness: {error}", file=sys.stderr)
        return 1
    try:
        version, reader = LIBRARIES[args.library](manifest)
    except ImportError as error:
        print(
            f"bellwether witness: the library {args.library} is not installed ({error}); "
            f"install the extra: pip install -e '.[deepseek]'",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED
    except ValueError as error:
        print(f"bellwether witness: {error}", file=sys.stderr)
        return 1
    print(f"{args.library} {version} over {manifest.slug}/parse")
    for line in witness_sets(args.fixtures, manifest, args.library, reader, list(args.sets or [])):
        print(line)
    return 0


__all__ = ["COMPRESSED_SUFFIX", "LIBRARIES", "library_body", "run", "witness_sets"]
