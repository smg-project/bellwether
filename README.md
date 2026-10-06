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
as evidence without authority. `docs/benchmark-sets.md` has the rules.

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
| witness | an engine's result; vLLM's, at the pinned release, is the other source of truth, and the rest are evidence, not authority |
| disputed | a case whose two sources of truth disagree; it carries the disagreement's fingerprint, and no parity is counted against it |
| verdict | the classification of one case after comparing SMG with reference and witnesses |
| waiver | a reviewed, expiring record explaining an `engine_defect`, with an upstream link |
| manifest | per-model file: revision, authority order, SMG and engine parser names |
| chunk plan | how an output token stream is cut into engine chunks for a streaming replay |

## Status

`gaps` is implemented (M1, 2026-10-05). The other subcommands exist and exit with status 2 until
their milestone lands:

| Milestone | Deliverable |
|---|---|
| M1 | `gaps` against the live registries and SMG (done) |
| M2 | render fixtures from the checkpoint template, in the order Symphony needs them: Qwen3-8B and DeepSeek-R1 (done), then DeepSeek-V4.1-Flash, GLM-5.3-Flash, MiniMax-M3, Kimi-K3; engine witnesses on Linux; fixture-driven tests in SMG |
| M3 | `mock-worker --script/--capture` in SMG; `verify` end to end on render cases |
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
generation prompt; the text in between is the output, its token ids the engine chunks a replay feeds,
and the fixture carries the chunk plans (fixed sizes, every two-way split for short outputs, thirty
seeded random plans). A template that does not extend the generation prompt when the turn is appended
(DeepSeek-R1 never renders `<think>`) cannot be this oracle for that case; the case is reported and
not recorded, and the run exits 1. Recording it is left to the manifest's next authority, the engine
witnesses, which nothing here invokes.

## The list of models

```bash
uv run bellwether models                   # registries and the Hugging Face Hub; writes models.jsonl
uv run bellwether models --registry-only   # the registries alone, offline once their files are cached
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
  engines give as an architecture's example (vLLM's default checkpoint, the ids in SGLang's docs)
  whose `config.json` names a registered architecture and that ships a chat template, leaving out
  quantized and converted copies (GGUF, AWQ, GPTQ, MLX, ONNX, FP8, NVFP4, MXFP4, MXFP8, Int4,
  Int8 or bitsandbytes in the name, or the Hub's `base_model:quantized` tag), embedding, reranking
  and classification models, and checkpoints created before 2025 that no registry names. A token
  (`HF_TOKEN`) raises the Hub's rate limits; none is needed, and a rate-limited call waits and is
  made again. A model whose tokenizer or processor config cannot be read (gated, an error from the
  Hub, a file that is not JSON) is kept, with a status that says so. A checkpoint vLLM loads with
  another repository's tokenizer (moondream3-preview with starmie-v1) is judged on that repository's
  template. When the Hub gives no answer
  about a model at all, a checkpoint a registry names keeps its row with the error as its status, and
  a model or an organization only a listing would have added is left out and named on stderr.

It writes one JSON line per checkpoint, and one per registry entry that names none, ordered by tier
and then within the tier:

| field | meaning |
|---|---|
| `model` | the Hugging Face id, as the Hub spells it; for a registry entry that names no checkpoint, the entry's name (vLLM's architecture, SGLang's model family) |
| `revision` | the Hub's sha when the list was built; for a checkpoint vLLM's registry pins (`refs/pr/17` for ERNIE-4.5-VL, a commit for HyperCLOVAX-SEED-Think-32B), the sha of that revision, and without the Hub the revision as vLLM writes it |
| `tier` | 1: Simo's models, in his order; 2: created in the twelve months before the build; 3: the rest. Within tiers 2 and 3, by 30-day downloads. `null` where the Hub decides the tier and was not asked (`--registry-only`) or gave no answer; those rows come after tier 2, by id |
| `status` | `pending`, or why nothing can be recorded yet: `no-checkpoint-named` (the registry entry names no checkpoint), `gated` (also when a config answers 401 or 403), `needs-vendor-code` (vLLM loads it with `trust_remote_code`, so the oracle would need the vendor's code; known without the Hub too), `no-chat-template`, `processor-chat-template` (the template is only in the processor's files, `chat_template.json` or the `chat_template` in `processor_config.json`, which `AutoProcessor` and vLLM read but the oracle's `AutoTokenizer` does not; the oracle reading processor templates is the follow-up), `not-on-hub`; when a config could not be read, `invalid-tokenizer-config` or `invalid-processor-config` (not a JSON object), or `hub-error-` and the HTTP status or the error (`hub-error-503`, `hub-error-read-timeout`); `unchecked` without the Hub |
| `created`, `downloads` | the Hub's creation date and downloads over the last 30 days |
| `modality` | `multimodal` when a registry or the architecture says so, else `text` |
| `sources` | the engines whose registries name the checkpoint, or `hub` |

The registries are pinned and the Hub is not, so a rebuild can change rows: what is reproducible is
the committed `models.jsonl` with each row's revision, and a rebuild's diff shows what changed.

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
