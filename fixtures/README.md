# Fixtures

One directory per checkpoint, named by a lowercase slug of the Hugging Face id:

```
fixtures/
  models.tsv               # the checkpoints the manifests are built from, with their downloads (see below)
  kimi-k3/
    manifest.toml          # model, pinned revision, tier, oracle inputs, authority order, parser names
    render/*.jsonl         # request -> prompt token ids
    parse/*.jsonl          # output token ids -> response, whole and per chunk plan (see below)
    tokenize/*.jsonl       # text -> ids
    detokenize/*.jsonl     # ids -> incremental text pieces
```

Checkpoints whose oracle inputs are equal render and parse identically, so they form one checkpoint group, recorded
once under the slug of its primary (`docs/benchmark-sets.md`, "Which models"). Every other member's directory holds
only its manifest, which names the group. Fixture ids carry the group's slug, so `qwen3-8b/parse/call-unicode-arguments`
is also Qwen3-0.6B's case.

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
tier     = 1                                             # the checkpoint's place in the recording order

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

[inputs]  # sha256 of each oracle input; the two config files over a few fields only (bellwether.inputs)
"config.json"            = "9b7728ead4a0106331ff2688b0eca40619af8f48e5ccb89270c6623f6420aab0"
"generation_config.json" = "7c21ad7edddca3978395226a08b102905092d7ffa0c45e4a3e88a271f553fc87"
"merges.txt"             = "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
"tokenizer.json"         = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
"tokenizer_config.json"  = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
"vocab.json"             = "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
```

`revision` is a 40-character commit hash, never a branch or tag, which could move under a consumer that caches the
fixtures. `[inputs]` lists every file the oracle reads, each with its sha256: the tokenizer files, the chat template
files, each named template (`additional_chat_templates/<name>.jinja`, of which transformers takes `tool_use` when a
request has tools), and `config.json` and `generation_config.json`, which are hashed over the canonical JSON
(`json.dumps(..., sort_keys=True)`, a missing field as null) of only `model_type` and `tokenizer_class`, and of only
`bos_token_id`, `eos_token_id` and `pad_token_id`, so that sampling defaults do not split a group. A checkpoint that
ships no `generation_config.json` has the end of a turn read its token ids from `config.json`, as
`GenerationConfig.from_model_config` reads them (`text_config` included, with the defaults of the class transformers
has for its `model_type`), so there `config.json` is hashed over those three ids as well. A member's manifest
adds `group = "<primary's slug>"` and has no `[smg]` or `[engines]`, since its fixtures are its group's. The parser
tables of a new group are left out until someone maps them. A manifest that `bellwether manifests` creates has no
`[authority]`: the order above predates the two sources of truth (`docs/benchmark-sets.md`, "Recording"), and
restating it for each checkpoint is the sponsor's call.

`bellwether manifests` writes them all from the list beside them, `fixtures/models.tsv` (`--models` reads another
file). The list has one row per checkpoint, `model<TAB>revision<TAB>downloads<TAB>day<TAB>tier`: `downloads` is the
Hub's count over the last 30 days and `day` the day it was read, and both are `-` where nobody read a count. The
command computes each checkpoint's inputs at its revision from the Hugging Face cache (`HF_HUB_OFFLINE=1` reads the
cache only), groups equal ones, and prints the groups. A group that is already recorded keeps its slug; a new group
takes its most-downloaded member's, a member with a count before one without, ties going to the first model id. The
list names every checkpoint that has a manifest, and the tests check that each manifest pins the list's revision and
tier and that each group recorded nowhere has the list's most-downloaded member as its primary. The command keeps
every line a person wrote in an existing manifest, never moves a recorded group's revision, and writes the same files
when run again, so new downloads or another checkpoint are a change to the list, reviewed as a diff of the list and
the manifests.

`record --model <id>` refuses a member, naming the group to record instead, and refuses a checkpoint whose inputs at
the pinned revision differ from its manifest's list, naming the files; it exits 1 in both cases, recording nothing.

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
it has one. vLLM reads `config.json` through the config class of its model type, whose defaults
count (`{"model_type": "llama"}` states no eos, and `LlamaConfig`'s is 2); here that class is
transformers' own for a model type transformers knows. Bellwether never runs vendor code, so this is
vLLM's stop set within one limit: a default eos that `config.json` does not state is missed when the
class is the vendor's (`auto_map`, which vLLM runs under `--trust-remote-code`) or one of vLLM's own
(`_CONFIG_REGISTRY` in `vllm/transformers_utils/config.py`). Only a checkpoint that ships no
`generation_config.json` can be affected.

The output ends before the first of those stop ids in the rendered turn; after it the turn may hold
only whitespace and further stop ids (Phi-4-mini writes `<|end|><|endoftext|>`). A turn with no stop
id is the output whole when the next message (a user message after content, one tool message per
call after tool calls) opens with one, right after the turn and on a token boundary: GLM writes no
end marker, and its `<|user|>` and `<|observation|>` are in its generation config. Any other case is
reported and not recorded, and so is a case whose message's own text (content, reasoning, a call's
name or arguments, as the template gets them) holds a stop id, since generation would stop inside
the message.

The reference's `end_of_turn`, beside `finish_reason`, holds the `stop_id` and the step that found it
(`found_by`: `turn` or `next-message`). The round trip knows which stop id the template writes where
the turn ends when the turn is the last message, not which one a model emits there:
Olmo-3-7B-Instruct's template writes `<|endoftext|>` (100257) after a last turn and `<|im_end|>`
(100265) after the same turn once a user message follows. Both are stop ids, generation stops on
either, and the line records 100257.

transformers' `generate` stops on the generation config's ids alone. Where it would not stop where
vLLM does (Qwen3.5-9B ships no `generation_config.json`, its `config.json` lists only
`<|endoftext|>`, and its turns end with the tokenizer's `<|im_end|>`), the output still ends where
vLLM stops, as serving engines do. What `generate` stops on is a fact of the checkpoint, not of a
case: each parse set's table in `sets.toml` holds the ids (`generate_stop_ids`) and the file they
come from (`generate_stop_ids_from`), so a line whose `stop_id` is not among them is one `generate`
would not end, and the run prints one line for the model, `stop sets differ for <model>: ...`, when
it recorded such outputs. The table is per set because `record --set` records one set at a time, so
each table states what its own set was recorded against.

`generation_config.json` must be cached at the revision or known absent, through the hub cache's
`.no_exist` marker, which `hf download <model> generation_config.json --revision <sha>` leaves when
the repository has no such file. Offline, transformers would take a file that is merely not cached
for one the repository does not ship, so `record` stops with an error naming the file and that
command.

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
sampled in context; a replay feeds them, with their pieces, as the engine's output. Where the
tokenizer's own ids decode to other text, because its normalizer applies a Unicode normalization form
(NFC, NFD, NFKC or NFKD), they are the encoding of the text as written, without those forms (#57).
A case whose template does not extend the generation prompt when the turn is appended, or whose
tokens do not give back its text under the incremental decode, is reported and not recorded, and the
run exits 1; recording it is left to the manifest's next authority, which nothing here invokes.
