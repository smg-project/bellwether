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

Scaffold, 2026-10-04. The subcommands exist and exit with status 2 until their milestone lands:

| Milestone | Deliverable |
|---|---|
| M1 | `gaps` against the live registries and SMG |
| M2 | render fixtures for Kimi-K3, DeepSeek-V4.1-Flash, MiniMax-M3, GLM-5.3-Flash; fixture-driven tests in SMG |
| M3 | `mock-worker --script/--capture` in SMG; `verify` end to end on render cases |
| M4 | parse and detokenize fixtures with chunk plans, the round-trip oracle, waivers |
| M5 | CI in both repositories; weekly record against engine nightlies; reports to `smg-project/artifacts` |
| M6 | coverage work from the gaps list |

## Quick start

```bash
uv sync --extra dev
uv run bellwether --help
uv run pytest -q
```

## Layout

```
src/bellwether/      package: cli.py and one subpackage per command
schemas/             case.schema.json, the fixture line format
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
