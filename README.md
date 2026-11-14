# bellwether

Bellwether records what a model's own reference code produces and measures a serving stack against it:
the prompt token ids a checkpoint's template renders for a request, and the content, reasoning and
tool calls an engine's token stream parses back to. Any gateway or engine that turns chat messages
into tokens and tokens into messages can be checked against the same fixtures. SMG is the first
consumer: its tokenizer crate, its gRPC request path and its Symphony parsers read them;
`docs/benchmark-sets.md` says how they are recorded and read.

Does a gateway send the engine the same prompt tokens the model vendor's own code would, and does it
turn the engine's tokens back into the same response? Bellwether answers that without a GPU.

A gateway renders chat messages and tools into token ids before the engine sees them, and
detokenizes and parses the engine's token stream into content, reasoning and tool calls. Both halves
are CPU work with a reference to compare against: the vendor's encoding code or chat template on
Hugging Face, and the Python frontends of vLLM and SGLang. Bellwether records what those references
produce, replays the same cases through the gateway under test fronting a scripted mock engine, and
classifies every difference.

The bellwether is the sheep that wears the bell and leads the flock. Here it is the reference each
result is measured against.

## How it works

1. **gaps.** Enumerate the tool parsers, reasoning parsers, renderers and tokenizer modes vLLM and
   SGLang register, map them onto the gateway's, and print the matrix. Every row missing a gateway
   column is a parser to write, and `record` hands the implementer fixtures before a line of it exists.
2. **record.** Run a request corpus through the references and the engines on CPU and commit the
   results as fixtures: prompt token ids for the render side, parsed responses at several chunk
   plans for the parse side, plus tokenize and incremental detokenize cases.
3. **verify.** Replay fixtures against the gateway fronting a scripted mock engine, which captures
   the token ids the gateway sent and streams back the recorded output tokens. Compare exactly,
   classify, apply waivers, fail only on the gateway's regressions.

## Who is right

Two sources of truth, equal in authority (the maintainer, 2026-10-06):

- **Hugging Face:** the checkpoint's own material at a pinned revision. That is the vendor's prompt
  encoder for an architecture the engine renders with one (in the copy the engine renders it with,
  whatever the checkpoint ships, since the engine never reads the checkpoint's:
  `src/bellwether/record/engine_encoders.toml`), else the chat template and tokenizer.
  For parsing it is the round trip: rendering the parsed message back through the template must
  reproduce the output.
- **vLLM:** at a pinned release, through its GPU-less render server.

When they agree, the case is settled. When they disagree, the case is `disputed`, the disagreement
is an issue, and nothing in bellwether is configured to make them agree. A line's `reference` is
Hugging Face's result and vLLM's is a `witness`. SGLang and engine runs on a GPU are witnesses too,
as evidence without authority. Where vLLM's docs name one of its example tool templates for a
checkpoint, a second reference rendered with that template sits beside the first; a difference
between the two is an issue, not a dispute, since they answer for two setups.
`docs/benchmark-sets.md` has the rules.

Every undisputed case gets a verdict: `match`, `engine_defect` (the gateway agrees with the sources
of truth, a witnessing engine does not), `engines_split`, `policy` (malformed output, documented
fallback expected), or `regression` (the gateway disagrees with them). A `disputed` case gets none; it waits
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
| verdict | the classification of one case after comparing the gateway under test with reference and witnesses |
| waiver | a reviewed, expiring record explaining an `engine_defect`, with an upstream link |
| manifest | per-checkpoint file: revision, tier, oracle inputs, group, authority order, the gateway's and the engines' parser names |
| oracle inputs | every file the oracle reads for a checkpoint, each with its sha256 |
| checkpoint group | checkpoints with equal oracle inputs, recorded once under the slug of its primary |
| member | any checkpoint of a checkpoint group, its primary included; the other members' manifests name the group |
| primary | the member a checkpoint group is recorded by, under its own slug, once for all its members |
| chunk plan | how an output token stream is cut into engine chunks for a streaming replay |
| capture | the mock engine's record of each request the gateway sends it, one JSON line per request |
| known difference | a case where the gateway is known to differ from the reference, listed for `verify --known` with the outcome it gives on it, the reason and the issue |

## Status

`gaps` (M1, 2026-10-05), `record` for render and parse with the reference oracle (M2, M4), `import`,
`count`, `unpack`, and `verify` on render cases against a running gateway (the first half of M3) are
implemented. `report`, `record` with an engine oracle or for tokenize and detokenize, and `verify` for
the other kinds exist and exit with status 2 until their milestone lands:

| Milestone | Deliverable |
|---|---|
| M1 | `gaps` against the live registries and the first gateway, SMG (done) |
| M2 | render fixtures from the checkpoint template, in the order Symphony needs them: Qwen3-8B and DeepSeek-R1 (done), then DeepSeek-V4.1-Flash, GLM-5.3-Flash, MiniMax-M3, Kimi-K3; engine witnesses on Linux; fixture-driven tests in SMG |
| M3 | a scripted mock engine with a capture file (SMG's `mock-worker --script/--capture`); `verify` end to end on render cases (against a running gateway: done; starting the gateway and the mock per model: next) |
| M4 | parse and detokenize fixtures with chunk plans, the round-trip oracle (Qwen3-8B done; DeepSeek-R1 needs a decision, issue #14), waivers |
| M5 | CI in both repositories; weekly record against engine nightlies; reports to `smg-project/artifacts` |
| M6 | coverage work from the gaps list |

## What bellwether holds

`INVENTORY.md` lists every recorded checkpoint group with its render and parse case counts, every
corpus source, and each image, audio and video set. `uv run bellwether count --readme INVENTORY.md`
writes it from `fixtures/` and `corpus/`, and CI runs it with `--check`, so it says what `main`
holds. `corpus/README.md` describes the request corpora and `fixtures/README.md` the recorded tree.

## What a gateway under test provides

- an OpenAI-style `/v1/chat/completions` route that renders the request with the checkpoint's template and sends the
  engine the token ids, and `/v1/models` listing what it serves;
- a request id passed through to the engine, so a capture line joins its case;
- a scripted mock engine that writes each request's token ids to a capture file and streams back the recorded output.

The adapters bellwether ships today speak to SMG's routes, error object and `mock-worker` capture format
(`docs/verify.md`, `docs/gaps.md`); a second gateway adds its own.

## Quick start

```bash
uv sync --extra dev
uv run bellwether --help
uv run pytest -q
```

## Commands

Each command has a guide with its options, what it reads and writes, and how it fails.

| Command | What it does | Guide |
|---|---|---|
| `gaps` | the coverage matrix: the parsers, renderers and tokenizer modes the engines register, mapped onto the gateway's | `docs/gaps.md` |
| `manifests` | groups the committed checkpoints by their oracle inputs and writes every `fixtures/<slug>/manifest.toml` | `docs/recording.md` |
| `record` | records a group's render or parse reference from its template at the pinned revision | `docs/recording.md` |
| `sandbox-record` | the same for a checkpoint whose tokenizer is the vendor's own code, inside a container | `docs/recording.md` |
| `patterns` | records what Python's `regex` module says about the response templates' patterns | `docs/recording.md` |
| `models` | builds the list of checkpoints to record from the engines' registries and the Hub | `docs/models.md` |
| `import`, `derive` | bring a public dataset into `corpus/` at a pinned revision; derive sets from imported ones | `corpus/README.md`, `docs/benchmark-sets.md` |
| `count` | the cases per checkpoint, kind and source; `--readme` writes `INVENTORY.md` | `INVENTORY.md` |
| `unpack` | writes every fixture set as plain JSON Lines into the tree consumers read | `docs/benchmark-sets.md`, "Fixture ids and the tree consumers read" |
| `verify` | replays render fixtures against a running gateway and gives each case a verdict | `docs/verify.md` |
| `report` | a stub until its milestone lands; exits 2 | |

## Layout

```
src/bellwether/      package: cli.py, one subpackage per command, schemas/case.schema.json (the fixture line format)
corpus/              request corpora, one JSON Lines file per set, and the media they reference (see corpus/README.md)
fixtures/            per-model manifests and recorded cases (see fixtures/README.md)
patterns/            what Python's regex module says about the response templates' patterns (bellwether patterns)
policy/              what SMG's routing policies pick, recorded for a second implementation to replay (see policy/README.md)
docs/                the design (benchmark-sets.md) and the command guides
docker/vendor/       the vendor-code oracle's base image (bellwether sandbox-record)
waivers/             engine_defects.toml
models.jsonl         the committed list of checkpoints the registries name (bellwether models --registry-only)
INVENTORY.md         what main holds, written by bellwether count
tests/
```

## Related

- [smg](https://github.com/smg-project/smg): the first gateway under test; `crates/mock_worker` is its replay engine.
- [providers-verifier](https://github.com/smg-project/providers-verifier): record a vendor's API, replay against a gateway. Its request catalogue is bellwether's first corpus.
- [deepseek-provider-verifier](https://github.com/smg-project/deepseek-provider-verifier).
- [artifacts](https://github.com/smg-project/artifacts): published evidence.

Apache-2.0.
