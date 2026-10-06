# Fixtures

One directory per model, named by a lowercase slug of the Hugging Face id:

```
fixtures/
  kimi-k3/
    manifest.toml          # model, pinned revision, authority order, SMG and engine parser names
    render/*.jsonl         # request -> prompt token ids
    parse/*.jsonl          # output token ids -> response, whole and per chunk plan (see below)
    tokenize/*.jsonl       # text -> ids
    detokenize/*.jsonl     # ids -> incremental text pieces
```

A set recorded from an imported corpus set (a benchmark set) is stored as `<set>.jsonl.zst`, zstd-compressed, in Git
LFS; hand-written sets stay plain `<set>.jsonl`. `.lfsconfig` keeps a clone from fetching the compressed sets; fetch
one with `git lfs pull --include 'fixtures/<slug>/<kind>/<set>.jsonl.zst' --exclude ''`, or write every set as plain
JSON Lines into one tree with `bellwether unpack` (`--out fixtures-plain` by default, `--model` to pick one), which
is the root consumers point at. Each model's `sets.toml`, written by `record`, lists every set with its form, cases,
rejected cases, plain size and plain-content sha256, so it can be counted and checked without fetching the set; a parse
set's table also names the ids transformers' `generate` stops on and the file they come from (Parse lines). A run reads
it again at its end and replaces only its own kind's tables, under a lock on `sets.toml.lock` beside it (ignored by
git), so a render run and a parse run of one model can overlap.

Every line validates against `src/bellwether/schemas/case.schema.json`. Fixtures are recorded by
`bellwether record`, never edited by hand; a re-record is a pull request whose diff is the review.
Each reference carries its provenance (oracle versions, Hugging Face revision, vendor file sha,
bellwether commit).

## Manifest

```toml
model    = "Qwen/Qwen3-8B"
revision = "b968826d9c46dd6066d109eabc6255188de91218"   # HF commit the fixtures are recorded at

[authority]                                              # first source that exists for a case wins
render     = ["hf-template", "engine:vllm", "engine:sglang"]
parse      = ["roundtrip", "engine:vllm", "engine:sglang"]
tokenize   = ["hf-tokenizer"]
detokenize = ["engine:vllm", "engine:sglang"]

[smg]                                                    # the names `gaps` marks as having fixtures
tool_parser      = "qwen"
reasoning_parser = "qwen3"

[engines]
vllm   = { tool_parser = "hermes", reasoning_parser = "qwen3" }
sglang = { tool_parser = "qwen25", reasoning_parser = "qwen3" }
```

`record --model <id>` finds the manifest whose `model` is that id, reads the corpus under
`corpus/<kind>/`, and writes `fixtures/<slug>/<kind>/<set>.jsonl`: one line per case, sorted by id,
canonical JSON (top-level keys in a fixed order, sorted keys in what bellwether produced, the request
exactly as the corpus wrote it, since its key order is part of what the oracle rendered), checked
against the schema. A re-run with the reference oracle replaces each line's
`reference` and keeps its `witnesses`; an engine oracle will do the reverse when engine recording lands
(M2, second half). The directory mirrors the corpus: a case or a set the corpus no longer has is removed
from it.

## Parse lines

`record --kind parse --oracle reference` records the round trip: the corpus states the assistant
message, the template renders it as the final assistant turn, and the turn up to where generation
stops is the output.

Generation stops on vLLM's stop set: the `eos_token_id` of the checkpoint's `generation_config.json`
at the manifest revision, or of its `config.json` when it ships none (read as transformers'
`GenerationConfig.from_model_config` reads it, `text_config` included), and the tokenizer's eos when
it has one. The output ends before the first of those stop ids in the rendered turn; after it the
turn may hold only whitespace and further stop ids (Phi-4-mini writes `<|end|><|endoftext|>`). A
turn with no stop id is the output whole when the next message (a user message after content, one
tool message per call after tool calls) opens with one, right after the turn and on a token
boundary: GLM writes no end marker, and its `<|user|>` and `<|observation|>` are in its generation
config. Any other case is reported and not recorded, and so is a case whose message's own text
(content, reasoning, a call's name or arguments, as the template gets them) holds a stop id, since
generation would stop inside the message. The reference's `end_of_turn`, beside `finish_reason`,
holds the `stop_id` and the step that found it (`found_by`: `turn` or `next-message`). The round
trip knows which stop id the template writes where the turn ends when the turn is the last message,
not which one a model emits there: Olmo-3-7B-Instruct's template writes `<|endoftext|>` (100257)
after a last turn and `<|im_end|>` (100265) after the same turn once a user message follows. Both
are stop ids, generation stops on either, and the line records 100257.

transformers' `generate` stops on the generation config's ids alone. Where it would not stop where
vLLM does (Qwen3.5-9B ships no `generation_config.json`, its `config.json` lists only
`<|endoftext|>`, and its turns end with the tokenizer's `<|im_end|>`), the output still ends where
vLLM stops, as serving engines do. What `generate` stops on is a fact of the checkpoint, not of a
case: each parse set's table in `sets.toml` holds the ids (`generate_stop_ids`) and the file they
come from (`generate_stop_ids_from`), so a line whose `stop_id` is not among them is one `generate`
would not end, and the run prints one line for the model, `stop sets differ for <model>: ...`, when
it recorded such outputs. The table is per set because `record --set` records one set at a time, so
each table states what its own set was recorded against. `generation_config.json` must be cached at the
revision or known absent, through the hub cache's `.no_exist` marker, which
`hf download <model> generation_config.json --revision <sha>` leaves when the repository has no such
file. Offline, transformers would take a file that is merely not cached for one the repository does
not ship, so `record` stops with an error naming the file and that command.

The template gets every assistant message, the request's history and the final turn alike, as vLLM
gives it to a template (`_postprocess_messages` in `vllm/entrypoints/chat_utils.py` at v0.31.0, the
release whose image bellwether pins): a call's arguments become the object they decode to, anything
that does not decode to a JSON object becomes `{}`, and an empty `tool_calls` is dropped. SGLang
differs: at 7d22b7a8 it rejects a string that is not a JSON object, except under Kimi-K3's encoding,
and leaves missing or null arguments as they are. That difference is recorded here, not decided. A
parse case's own call must carry its arguments as a JSON object string, the one a parser returns:
that is a rule of the corpus, not of an engine. A template that cannot take an object fails the
case, which is reported as a finding.

A line carries `request` (what a replay sends to SMG),
`tools`, `output_ids` (the output's tokens, the stop id excluded), `output_pieces` (the text
each of those tokens contributes under the tokenizer's incremental decode, tokenizers' `DecodeStream`:
a token that does not complete a character contributes an empty piece and the token that completes it
carries the whole character; one piece per id, and joined they are the output text), `malformed: false`,
`chunk_plans` (`whole` and `per_token` derived at replay time; `size-<n>` fixed lengths; `split-<k>`
every two-way split for outputs of at most 32 tokens; `random-<seed>` thirty seeded plans with chunks
of one to eight tokens) and `reference` with `source: roundtrip`, the `message` (with `role`), the
`finish_reason` (`tool_calls` when the message has calls, else `stop`), `end_of_turn`, the output
`text` and the provenance. `output_ids` are the tokenizer's encoding of the output text on its own, not ids a model
sampled in context; a replay feeds them, with their pieces, as the engine's output. A case whose
template does not extend the generation prompt when the turn is appended, or whose tokens do not give
back its text under the incremental decode, is reported and not recorded, and the run exits 1;
recording it is left to the manifest's next authority, which nothing here invokes.
