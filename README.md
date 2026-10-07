# bellwether

> Given a request, what's expected tokens. Given a token, what's expected output text. With insane
> amount of data. Then smg uses it in many places. Tokenizer, detokenization, gRPC router, and
> symphony.
>
> Simo, 2026-10-06

Bellwether holds expected values for several consumers in smg: the tokenizer crate (encoding and
incremental decoding), the gRPC router's request path, and Symphony's parsers.
`docs/benchmark-sets.md` says how they are recorded and read.

Does SMG send the engine the same prompt tokens the model vendor's own code would, and does it
turn the engine's tokens back into the same response? Bellwether answers that without a GPU.

SMG renders chat messages and tools into token ids before the engine sees them, and detokenizes
and parses the engine's token stream into content, reasoning and tool calls. Both halves are CPU
work with a reference to compare against: the vendor's encoding code or chat template on Hugging
Face, and the Python frontends of vLLM and SGLang. Bellwether records what those references
produce, replays the same cases through a real SMG fronting a scripted mock engine, and
classifies every difference.

The bellwether is the sheep that wears the bell and leads the flock. Here it is the reference each
result is measured against.

## How it works

1. **gaps.** Enumerate the tool parsers, reasoning parsers, renderers and tokenizer modes vLLM and
   SGLang register, map them onto SMG's, and print the matrix. Every row missing an SMG column is a
   parser to write, and `record` hands the implementer fixtures before a line of Rust exists.
2. **record.** Run a request corpus through the references and the engines on CPU and commit the
   results as fixtures: prompt token ids for the render side, parsed responses at several chunk
   plans for the parse side, plus tokenize and incremental detokenize cases.
3. **verify.** Replay fixtures against SMG fronting its `mock-worker`, which captures the token ids
   SMG sent and streams back the recorded output tokens. Compare exactly, classify, apply waivers,
   fail only on SMG regressions.

## Who is right

Two sources of truth, equal in authority (Simo, 2026-10-06):

- **Hugging Face:** the checkpoint's own material at a pinned revision. That is the vendor's shipped
  encoder when there is one, else the chat template and tokenizer. For parsing it is the round trip:
  rendering the parsed message back through the template must reproduce the output.
- **vLLM:** at a pinned release, through its GPU-less render server.

When they agree, the case is settled. When they disagree, the case is `disputed`, the disagreement
is an issue, and nothing in bellwether is configured to make them agree. A line's `reference` is
Hugging Face's result and vLLM's is a `witness`. SGLang and engine runs on a GPU are witnesses too,
as evidence without authority. Where vLLM's docs name one of its example tool templates for a
checkpoint, a second reference rendered with that template sits beside the first; a difference
between the two is an issue, not a dispute, since they answer for two setups.
`docs/benchmark-sets.md` has the rules.

Every undisputed case gets a verdict: `match`, `engine_defect` (SMG agrees with the sources of
truth, a witnessing engine does not), `engines_split`, `policy` (malformed output, documented
fallback expected), or `regression` (SMG disagrees with them). A `disputed` case gets none; it waits
for its disagreement's issue. A case that has only its Hugging Face result so far is judged against
it, and becomes settled or disputed once vLLM's result arrives. Only `regression`, an
`engine_defect` without a waiver, and a stale waiver fail the gate. Waivers live in
`waivers/engine_defects.toml`, carry evidence and an upstream link, and expire when the engine
catches up.

## Vocabulary

| Term | Meaning |
|---|---|
| case | one request or one engine output, with everything needed to reproduce it |
| fixture | a case plus the recorded reference result and each witness's result |
| reference | Hugging Face's result: the checkpoint's own material at its pinned revision; one of the two sources of truth |
| second reference | the reference's oracle run with a tool chat template from vLLM's `examples/` that vLLM's docs name for the checkpoint, in place of its own template; recorded beside the reference, one per such template |
| witness | an engine's result; vLLM's, at the pinned release, is the other source of truth, and the rest are evidence, not authority |
| disputed | a case whose two sources of truth disagree; it carries the disagreement's fingerprint, and no parity is counted against it |
| verdict | the classification of one case after comparing SMG with reference and witnesses |
| waiver | a reviewed, expiring record explaining an `engine_defect`, with an upstream link |
| manifest | per-model file: revision, authority order, SMG and engine parser names |
| chunk plan | how an output token stream is cut into engine chunks for a streaming replay |
| capture | the mock worker's record of each request SMG sends it, one JSON line per request |
| known difference | a case where SMG is known to differ from the reference, listed for `verify --known` with the outcome SMG gives on it, the reason and the issue |

## Status

`gaps` (M1, 2026-10-05), `record` for render and parse with the reference oracle (M2, M4), `import`,
`count`, `unpack`, and `verify` on render cases against a running SMG (the first half of M3) are
implemented. `report`, `record` with an engine oracle or for tokenize and detokenize, and `verify` for
the other kinds exist and exit with status 2 until their milestone lands:

| Milestone | Deliverable |
|---|---|
| M1 | `gaps` against the live registries and SMG (done) |
| M2 | render fixtures from the checkpoint template, in the order Symphony needs them: Qwen3-8B and DeepSeek-R1 (done), then DeepSeek-V4.1-Flash, GLM-5.3-Flash, MiniMax-M3, Kimi-K3; engine witnesses on Linux; fixture-driven tests in SMG |
| M3 | `mock-worker --script/--capture` in SMG; `verify` end to end on render cases (against a running SMG: done; starting SMG and the mock per model: next) |
| M4 | parse and detokenize fixtures with chunk plans, the round-trip oracle (Qwen3-8B done; DeepSeek-R1 needs a decision, issue #14), waivers |
| M5 | CI in both repositories; weekly record against engine nightlies; reports to `smg-project/artifacts` |
| M6 | coverage work from the gaps list |

## Quick start

```bash
uv sync --extra dev
uv run bellwether --help
uv run pytest -q

# The coverage matrix: engine registries at a pinned commit or tag, SMG from a checkout.
uv run bellwether gaps --vllm-ref v0.30.0 --sglang-ref v0.5.20 --smg-src ../smg
uv run bellwether gaps --vllm-src ../vllm --sglang-src ../sglang --smg-src ../smg --format json --out gaps.json
```

`gaps` reads the registries from source with `ast` and regular expressions, never by importing an
engine: vLLM's four `name -> (module, class)` tables, SGLang's two name lists and class maps plus
its native renderer and tokenizer modules, and SMG's two factories, renderer enum and tokenizer
types. Names that differ across systems for one format are merged through
`src/bellwether/gaps/aliases.toml`, where every entry states its evidence; the matrix also lists
one implementation behind several rows as alias candidates, so the table is kept honest by what
the code says rather than by memory.

## Recording render fixtures

```bash
uv run bellwether record --model Qwen/Qwen3-8B --kind render --oracle reference
```

Finds the manifest whose `model` is the given id (`fixtures/qwen3-8b/manifest.toml`), runs every
case under `corpus/render/` through the checkpoint's own chat template at the pinned revision
(`transformers.apply_chat_template`), and writes `fixtures/qwen3-8b/render/<set>.jsonl`, or
`<set>.jsonl.zst` in Git LFS for an imported set (`corpus/README.md`), with each set's counts and
plain-content sha256 in `fixtures/qwen3-8b/sets.toml`. `--set NAME` records only the named sets. Per
case the file holds the request, the prompt token ids, the rendered text and the oracle versions. A
re-run replaces each reference, keeps the witnesses already recorded for an unchanged request, and
removes cases and sets the corpus no longer has. A case the template cannot render is reported on
stderr and left out, and the command exits 1. Engine witnesses (`--oracle vllm|sglang`) run on Linux
inside the engine's image and are the second half of M2.

## Recording parse fixtures (the round trip)

```bash
uv run bellwether record --model Qwen/Qwen3-8B --kind parse --oracle reference
```

Each case under `corpus/parse/` states the assistant message (content, reasoning, tool calls) the
output must parse to. The checkpoint's template renders it as the final assistant turn after the
generation prompt; the turn up to where generation stops, at the first of vLLM's stop ids, is the
output, its token ids the engine chunks a replay feeds, and the fixture carries the chunk plans (fixed
sizes, every two-way split for short outputs, thirty seeded random plans) and the stop id that ends
the output (`end_of_turn`). A template that does not extend the generation prompt when the turn is
appended (DeepSeek-R1 never renders `<think>`) cannot be this oracle for that case; the case is
reported and not recorded, and the run exits 1. Recording it is left to the manifest's next authority,
the engine witnesses, which nothing here invokes. `fixtures/README.md` has the rules.

The stop ids are the eos ids of the checkpoint's `generation_config.json` (of its `config.json` when it
ships none) together with the tokenizer's eos, so `record --kind parse` needs `generation_config.json`
cached at the manifest's revision, or known to be absent: offline, transformers would take a
file that is merely not cached for one the repository does not ship. The run stops with an error
naming the file and the command that fetches it,
`hf download <model> generation_config.json --revision <sha>`, which leaves the hub cache's
`.no_exist` marker when the repository has no such file.

## The list of models

```bash
uv run bellwether models --registry-only           # the registries alone; writes the committed models.jsonl
uv run bellwether models --registry-only --check   # build it again and compare with models.jsonl, as CI does
uv run bellwether models                           # registries and the Hub; writes runs/models-<date>.jsonl
```

`models` builds the list of checkpoints to record: every generative model vLLM or SGLang supports,
except gpt-oss (the rule is in `docs/benchmark-sets.md`, "Which models"); the gpt-oss checkpoints the
registries name, by name or by vLLM's `GptOssForCausalLM`, are named on stderr as set aside. It reads
two sources:

- **The engines' registries at pinned commits.** vLLM's `tests/models/registry.py` at v0.31.0, read
  with `ast`: every checkpoint its text-generation and multimodal tables name, except under a
  ranking or classification head. SGLang's three "Text Generation" docs pages at 7d22b7a8: every id
  in a table's example column. SGLang's code at the same commit, read with `ast`: each module of
  `python/sglang/srt/models/` names the architectures it serves as `EntryClass` (251), and the
  multimodal processors name those they serve with images, audio or video. The registered
  architectures are vLLM's generative tables and those SGLang's code serves, leaving out pooling and
  speculative-decoding draft heads. The files come from a checkout given with `--vllm-src` and
  `--sglang-src` (read with `git show`, whatever the checkout has checked out) or from GitHub, are
  checked against their pinned sha256 on every use, and are cached by commit under
  `~/.cache/bellwether/registries`.
- **The Hugging Face Hub as it is today.** Every model of the organizations whose checkpoints the
  engines give as an architecture's example (vLLM's default checkpoint, the ids in SGLang's docs),
  or as one of vLLM's extras that is a real checkpoint rather than a tiny or random test model or a
  quantized copy (NousResearch's Hermes 3, mistral-community's Pixtral), whose `config.json` names a
  registered architecture and that ships a chat template, leaving out quantized and converted copies
  (GGUF, AWQ, GPTQ, MLX, ONNX, FP8, NVFP4, MXFP4, MXFP8, Int4, Int8 or bitsandbytes in the name, or
  the Hub's `base_model:quantized` tag), embedding, reranking and classification models, and
  checkpoints created before 2025 that no registry names. A token (`HF_TOKEN`) raises the Hub's rate
  limits; none is needed, and a rate-limited call waits and is made again. A model whose tokenizer or
  processor config cannot be read (gated, an error from the Hub, a file that is not JSON) is kept,
  with a status that says so. A checkpoint vLLM loads with another repository's tokenizer
  (moondream3-preview with starmie-v1) is judged on that repository's template. When the Hub gives no
  answer about a model at all, a checkpoint a registry names keeps its row with the error as its
  status, and a model or an organization only a listing would have added is left out and named on
  stderr.

The file's first line says what the list was built from: each engine's repository, ref and commit,
and `hub`, the day the Hub was read, or `null` without it. Tier 2 and downloads depend on that day;
nothing in a list built without the Hub depends on a date. One JSON line follows per checkpoint, and
one per registry entry that names none, ordered by tier and then within the tier:

| field | meaning |
|---|---|
| `model` | the Hugging Face id, as the Hub spells it; for a registry entry that names no checkpoint, the entry's name (vLLM's architecture, SGLang's model family) |
| `revision` | the Hub's sha when the list was built; for a checkpoint vLLM's registry pins (`refs/pr/17` for ERNIE-4.5-VL, a commit for HyperCLOVAX-SEED-Think-32B), the sha of that revision, and without the Hub the revision as vLLM writes it |
| `tier` | 1: Simo's models, in his order; 2: created in the twelve months before the build; 3: the rest. Within tiers 2 and 3, by 30-day downloads. `null` where the Hub decides the tier and was not asked (`--registry-only`) or gave no answer; those rows come after tier 2, by id |
| `status` | `pending`, or why nothing can be recorded yet: `no-checkpoint-named` (the registry entry names no checkpoint), `gated` (also when a config answers 401 or 403), `needs-vendor-code` (vLLM loads it with `trust_remote_code`, so the oracle would need the vendor's code; known without the Hub too), `no-chat-template`, `processor-chat-template` (the template is only in the processor's files, `chat_template.json` or the `chat_template` in `processor_config.json`, which `AutoProcessor` and vLLM read but the oracle's `AutoTokenizer` does not; the oracle reading processor templates is the follow-up), `not-on-hub`; when a config could not be read, `invalid-tokenizer-config` or `invalid-processor-config` (not a JSON object), or `hub-error-` and the HTTP status or the error (`hub-error-503`, `hub-error-read-timeout`); `unchecked` without the Hub |
| `created`, `downloads` | the Hub's creation date and downloads over the last 30 days |
| `modality` | `multimodal` when one of its architectures is multimodal in either engine's code (vLLM's multimodal table, an SGLang multimodal processor), else `text`; its architectures are those vLLM lists it under and, with the Hub, those its config names. `null` when none is known: an id only SGLang's docs give, without the Hub |
| `sources` | the engines whose registries name the checkpoint, or `hub` |

Without the Hub the list depends on the pins alone, so it is committed: `models.jsonl` at the
repository's root, which CI builds again, with the registry files cached, and compares (`--check`),
as the importers check their sets. A list read from the Hub changes every day, its downloads and
their order with it, so it is that day's evidence, not a committed file: it goes to
`runs/models-<date>.jsonl` and is published to smg-project/artifacts, each row pinning the sha the
Hub gave that day.

## Verifying render fixtures against a running SMG

```bash
uv run bellwether verify --smg http://127.0.0.1:30000 --capture capture.jsonl --model Qwen/Qwen3-8B \
  --report runs/verify.json --junit runs/verify.xml
```

SMG runs in front of its `mock-worker`, started with `--capture capture.jsonl`, which creates the file
and appends each request SMG sends it as one JSON line; `verify` opens the file before its first
request. `verify` posts every render fixture's request to `/v1/chat/completions` with the manifest's
`model`, the fixture id as `rid`, `stream: false` and, unless the request sets a limit, `max_tokens: 1`;
none of these reach the chat template. SMG passes the `rid` to the engine as `request_id`, verbatim, or
in prefill-decode mode as `<rid>-<uuid>`, with a fresh UUID for each attempt; `verify` takes that suffix
off, so the capture lines written during the run are joined to the cases on it, and each case gets a
verdict. Every render set of the selected models is read whole, plain or compressed, before the first
request, and nothing in them is passed over: a set Git LFS has not fetched stops the run with the
command that fetches it, and so does a set `sets.toml` lists that the checkout lacks, a set that is not
zstd or is cut short, a line that is not a case with a request and reference ids, and a case id in two
of a model's sets. A model named with `--model` must have render cases; any other model without them is
named in the report. `--set NAME` verifies only the named render sets, and each must be some selected
model's. That first read keeps only the case ids; the cases are then sent, judged and written one at a
time, so a run's memory does not grow with its number of cases, and the JSON report and the JUnit XML
are put together when the run finishes.

| Verdict | Meaning |
|---|---|
| `match` | the `input_ids` SMG sent equal the reference's |
| `regression` | they do not; the report gives the first differing index, the ids around it on both sides, and whether the prompt text SMG sent equals the reference text (equal text points at tokenization, other text at rendering) |
| `rejected` | SMG refused the request: a 400 whose body is SMG's error object; its code and message are kept |
| `missing` | SMG answered 200, but no capture line written during the run carries the case's id: the file is not the one the engine behind this SMG writes, or the `rid` did not reach it |
| `measurement_failed` | SMG answered neither 200 nor its refusal (a redirect, a 404 for a model no worker serves, a 429 or 5xx, a proxy's page) and no capture line carries the case's id: nothing was measured |

Every capture line that carries a case's id is compared, whatever SMG answered after sending it, and the
answer is kept beside the verdict. A request can reach the engine more than once (a retry; under
prefill-decode, both legs), so a case matches only when each of its lines does. The report counts the
lines written during the run, those that joined a case and those that did not (another client's, a line
for no case), so a line no case takes is never dropped unseen. `missing` and `measurement_failed` are
about the setup, so no known difference excuses them. Before the first case, `verify` asks SMG's
`/v1/models` for the models it serves; a model with cases that SMG does not list stops the run, as does
an answer other than SMG's list (a 404 there usually means `--smg` ends in `/v1`). The report and the
messages give `--smg` without a user and password, should the URL carry them.

`--known PATH` lists SMG's known differences, one TOML table per fixture id stating the outcome SMG is
known to give, why, and the issue that tracks it; bellwether ships no such list:

```toml
["qwen3-8b/render/some-case"]
verdict = "rejected"   # or "regression"
code = "bad_request"   # a rejected entry's: SMG's error code, or 400 for its validation errors
reason = "what SMG does on this case, and why it is not fixed yet"
issue = "https://github.com/smg-project/smg/issues/NNN"
```

A listed case passes while it has exactly that outcome and fails on any other: a match, so the entry
goes as soon as SMG is fixed; another verdict or code; or an outcome about the setup. A listed id that
can name no case fails the run: one that is not a fixture id, one whose slug no manifest has, or a
render id of a verified model that a run of every set did not find. Ids of models, sets or kinds the run
does not verify are counted and listed in the report, so the list cannot go stale. The exit status is 0
when every case passes, 1 when one does not, and 2 when the run gives no verdict (no such model, a
manifest or a set verify cannot read, a capture file verify cannot open, no answer from SMG, a model it
does not serve) or stops before its last case. After the first request, a case left without a verdict
(no answer from SMG, a capture line verify cannot read, a set that cannot be read again) stops the
sending: the cases answered so far are judged and reported, and the report names the case the run
stopped at and every case it did not send. A case is judged against its reference alone for now: telling
an `engine_defect` or `engines_split` from a `regression` needs the engine witnesses, which come once
they are recorded.

## Layout

```
src/bellwether/      package: cli.py, one subpackage per command, schemas/case.schema.json (the fixture line format)
corpus/              request corpora, one JSON Lines file per set (see corpus/README.md)
fixtures/            per-model manifests and recorded cases (see fixtures/README.md)
waivers/             engine_defects.toml
tests/
```

## Related

- [smg](https://github.com/smg-project/smg): the gateway under test; `crates/mock_worker` is the replay engine.
- [providers-verifier](https://github.com/smg-project/providers-verifier): record a vendor's API, replay against SMG. Its request catalogue is bellwether's first corpus.
- [deepseek-provider-verifier](https://github.com/smg-project/deepseek-provider-verifier).
- [artifacts](https://github.com/smg-project/artifacts): published evidence.

Apache-2.0.
