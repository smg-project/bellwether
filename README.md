# bellwether

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

The engines are witnesses, not the judge. Each model's manifest names its authority order: the
vendor's shipped encoder when there is one, else the chat template and tokenizer at a pinned
revision, then vendor golden sets, then the engines, with vLLM as the tie-break between engines.
For parsing, rendering the parsed message back through the template must reproduce the output,
which gives an engine-independent oracle for every model with a template.

Every case gets a verdict: `match`, `engine_defect` (SMG agrees with the reference, an engine does
not), `engines_split`, `policy` (malformed output, documented fallback expected), or `regression`
(SMG disagrees with the reference). Only `regression`, an `engine_defect` without a waiver, and a
stale waiver fail the gate. Waivers live in `waivers/engine_defects.toml`, carry evidence and an
upstream link, and expire when the engine catches up.

## Vocabulary

| Term | Meaning |
|---|---|
| case | one request or one engine output, with everything needed to reproduce it |
| fixture | a case plus the recorded reference result and each witness's result |
| reference | the result the vendor's own material produces; ground truth |
| witness | an engine's result; evidence, not authority |
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
hand-written case under `corpus/render/` through the checkpoint's own chat template at the pinned
revision (`transformers.apply_chat_template`), and writes `fixtures/qwen3-8b/render/<set>.jsonl`. An
imported set (`corpus/README.md`) is left out, and named on stdout, until the storage form of
`docs/benchmark-sets.md` lands; `--set NAME` records only the named sets. Per case the file holds the
request, the prompt token ids, the rendered text and the oracle versions. A re-run replaces each
reference, keeps the witnesses already recorded for an unchanged request, and removes cases and sets
the corpus no longer has. A case the template cannot render is reported on
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
