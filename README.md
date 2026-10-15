# bellwether

> Given a request, what's expected tokens. Given a token, what's expected output text. With insane
> amount of data. Then smg uses it in many places. Tokenizer, detokenization, gRPC router, and
> symphony.
>
> The maintainer, 2026-10-06

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

Two sources of truth, equal in authority (the maintainer, 2026-10-06):

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
| manifest | per-checkpoint file: revision, tier, oracle inputs, group, authority order, SMG and engine parser names |
| oracle inputs | every file the oracle reads for a checkpoint, each with its sha256 |
| checkpoint group | checkpoints with equal oracle inputs, recorded once under the slug of its primary |
| member | any checkpoint of a checkpoint group, its primary included; the other members' manifests name the group |
| primary | the member a checkpoint group is recorded by, under its own slug, once for all its members |
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

## What bellwether holds

`uv run bellwether count --readme README.md` writes the two tables below from `fixtures/` and `corpus/`, and CI runs it with `--check`, so they say what main holds.

<!-- bellwether count --readme writes everything from here to the end marker: edit none of it by hand -->

### Models

Each checkpoint group is recorded once, by its primary, for every checkpoint that shares its tokenizer and template; the counts are each group's `sets.toml`.

| Group | Model | Checkpoints | Tier | Render cases | Parse cases | Refused | Sources |
|---|---|---:|---:|---:|---:|---:|---|
| [a.x-3.1](fixtures/a.x-3.1/sets.toml) | skt/A.X-3.1 @ 1d994491 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [a.x-3.1-light](fixtures/a.x-3.1-light/sets.toml) | skt/A.X-3.1-Light @ 9b41bb24 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [a.x-4.0](fixtures/a.x-4.0/sets.toml) | skt/A.X-4.0 @ a3393fd6 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [a.x-4.0-light](fixtures/a.x-4.0-light/sets.toml) | skt/A.X-4.0-Light @ ba21c20e | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [academic-ds-9b](fixtures/academic-ds-9b/sets.toml) | ByteDance-Seed/academic-ds-9B @ 2a205727 | 1 | 3 | 531736 | 3114 | 332750 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [aceinstruct-1.5b](fixtures/aceinstruct-1.5b/sets.toml) | nvidia/AceInstruct-1.5B @ 1e3d0207 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [acemath-1.5b-instruct](fixtures/acemath-1.5b-instruct/sets.toml) | nvidia/AceMath-1.5B-Instruct @ 166818c3 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [acereason-nemotron-1.1-7b](fixtures/acereason-nemotron-1.1-7b/sets.toml) | nvidia/AceReason-Nemotron-1.1-7B @ 2be9ed2f | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [acereason-nemotron-7b](fixtures/acereason-nemotron-7b/sets.toml) | nvidia/AceReason-Nemotron-7B @ 7d385d30 | 4 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [afm-4.5b](fixtures/afm-4.5b/sets.toml) | arcee-ai/AFM-4.5B @ e84d1971 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [afm-4.5b-preview](fixtures/afm-4.5b-preview/sets.toml) | arcee-ai/AFM-4.5B-Preview @ 5cf715ac | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [agentcpm-explore](fixtures/agentcpm-explore/sets.toml) | openbmb/AgentCPM-Explore @ f528bbb5 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [agentcpm-report](fixtures/agentcpm-report/sets.toml) | openbmb/AgentCPM-Report @ c6232d64 | 1 | 2 | 531736 | 328660 | 111770 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [aguvis-7b-720p](fixtures/aguvis-7b-720p/sets.toml) | xlangai/Aguvis-7B-720P @ 6dd54127 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ai21-jamba-reasoning-3b](fixtures/ai21-jamba-reasoning-3b/sets.toml) | ai21labs/AI21-Jamba-Reasoning-3B @ 75243704 | 1 | 3 | 531736 | 12885 | 11 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ai21-jamba2-3b](fixtures/ai21-jamba2-3b/sets.toml) | ai21labs/AI21-Jamba2-3B @ 525c6c8e | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [androidgen-llama-3-70b](fixtures/androidgen-llama-3-70b/sets.toml) | zai-org/androidgen-llama-3-70b @ 1c0e09bd | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [apertus-70b-instruct-2509](fixtures/apertus-70b-instruct-2509/sets.toml) | swiss-ai/Apertus-70B-Instruct-2509 @ 0f276766 | 1 | 3 | 136946 | 137178 | 382850 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [apertus-8b-instruct-2509](fixtures/apertus-8b-instruct-2509/sets.toml) | swiss-ai/Apertus-8B-Instruct-2509 @ b946d404 | 1 | 2 | 136946 | 137178 | 382850 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [apertus-v1.1-4b-instruct](fixtures/apertus-v1.1-4b-instruct/sets.toml) | swiss-ai/Apertus-v1.1-4B-Instruct @ 0d004ae5 | 3 | 2 | 238488 | 184972 | 533009 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arcee-blitz](fixtures/arcee-blitz/sets.toml) | arcee-ai/Arcee-Blitz @ d8f23453 | 1 | 3 | 238488 | 184972 | 533009 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arcee-blitz-caller-beta](fixtures/arcee-blitz-caller-beta/sets.toml) | arcee-ai/arcee-blitz-caller-beta @ a0785999 | 1 | 3 | 531736 | 324160 | 142324 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arcee-maestro-7b-preview](fixtures/arcee-maestro-7b-preview/sets.toml) | arcee-ai/Arcee-Maestro-7B-Preview @ dc3ee7b5 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [arcee-supernova-v1](fixtures/arcee-supernova-v1/sets.toml) | arcee-ai/Arcee-SuperNova-v1 @ 7380333e | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-2-14b](fixtures/areal-boba-2-14b/sets.toml) | inclusionAI/AReaL-boba-2-14B @ 3642aaaa | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-2-32b](fixtures/areal-boba-2-32b/sets.toml) | inclusionAI/AReaL-boba-2-32B @ 803ae052 | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-2-8b](fixtures/areal-boba-2-8b/sets.toml) | inclusionAI/AReaL-boba-2-8B @ 88ba93ee | 3 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-rl-7b](fixtures/areal-boba-rl-7b/sets.toml) | inclusionAI/AReaL-boba-RL-7B @ 5040fc18 | 3 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-sft-32b](fixtures/areal-boba-sft-32b/sets.toml) | inclusionAI/AReaL-boba-SFT-32B @ 85f0f152 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-sea-235b-a22b](fixtures/areal-sea-235b-a22b/sets.toml) | inclusionAI/AReaL-SEA-235B-A22B @ a186e783 | 2 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-tau2-telecom-sft-30b](fixtures/areal-tau2-telecom-sft-30b/sets.toml) | inclusionAI/AReaL-tau2-telecom-sft-30B @ 41d6b078 | 3 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-tau2-telecom-user-sft-30b](fixtures/areal-tau2-telecom-user-sft-30b/sets.toml) | inclusionAI/AReaL-tau2-telecom-user-sft-30B @ 5a61a868 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arex-2](fixtures/arex-2/sets.toml) | BAAI/AREX-2 @ d4e3502f | 1 | 2 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arex-base](fixtures/arex-base/sets.toml) | BAAI/AREX-Base @ 970a3279 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arex-turbo](fixtures/arex-turbo/sets.toml) | BAAI/AREX-Turbo @ e455a798 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [aria](fixtures/aria/sets.toml) | rhymes-ai/Aria @ 53a09c77 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [asearcher-local-7b](fixtures/asearcher-local-7b/sets.toml) | inclusionAI/ASearcher-Local-7B @ bd1b05b8 | 4 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [asearcher-web-qwq](fixtures/asearcher-web-qwq/sets.toml) | inclusionAI/ASearcher-Web-QwQ @ 6bf738b9 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [astabrief-8b](fixtures/astabrief-8b/sets.toml) | allenai/AstaBrief_8B @ 6a34f54d | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [atria-dawn-preview](fixtures/atria-dawn-preview/sets.toml) | internlm/Atria-Dawn-Preview @ 311a9b8c | 2 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [audio-flamingo-3-hf](fixtures/audio-flamingo-3-hf/sets.toml) | nvidia/audio-flamingo-3-hf @ 7d4bae64 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [autoglm-phone-9b](fixtures/autoglm-phone-9b/sets.toml) | zai-org/AutoGLM-Phone-9B @ 66a46ea2 | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [autoglm-phone-9b-multilingual](fixtures/autoglm-phone-9b-multilingual/sets.toml) | zai-org/AutoGLM-Phone-9B-Multilingual @ 832ab5e0 | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bagel-7b-mot](fixtures/bagel-7b-mot/sets.toml) | ByteDance-Seed/BAGEL-7B-MoT @ 5019f57d | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [baichuan-m2-32b](fixtures/baichuan-m2-32b/sets.toml) | baichuan-inc/Baichuan-M2-32B @ 360f684d | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [baichuan-m3-235b](fixtures/baichuan-m3-235b/sets.toml) | baichuan-inc/Baichuan-M3-235B @ 03a1fd5e | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [baichuanmed-ocr-7b](fixtures/baichuanmed-ocr-7b/sets.toml) | baichuan-inc/BaichuanMed-OCR-7B @ 826944c7 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-2x7b-base](fixtures/bar-2x7b-base/sets.toml) | allenai/BAR-2x7B-Base @ 80987e28 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-2x7b-code](fixtures/bar-2x7b-code/sets.toml) | allenai/BAR-2x7B-Code @ dc5c4003 | 5 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-2x7b-math](fixtures/bar-2x7b-math/sets.toml) | allenai/BAR-2x7B-Math @ 772eb58b | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-7b](fixtures/bar-7b/sets.toml) | allenai/BAR-7B @ bee49db0 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bee-8b-sft](fixtures/bee-8b-sft/sets.toml) | Open-Bee/Bee-8B-SFT @ a81758f4 | 3 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [bfs-prover-v1-7b](fixtures/bfs-prover-v1-7b/sets.toml) | ByteDance-Seed/BFS-Prover-V1-7B @ 750e3903 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [bfs-prover-v2-7b](fixtures/bfs-prover-v2-7b/sets.toml) | ByteDance-Seed/BFS-Prover-V2-7B @ 533d59fc | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [bitcpm-cann-3b-unquantized](fixtures/bitcpm-cann-3b-unquantized/sets.toml) | openbmb/BitCPM-CANN-3B-unquantized @ 6971d179 | 4 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bytedance-seed-oss-36b-alternate-tokenizer](fixtures/bytedance-seed-oss-36b-alternate-tokenizer/sets.toml) | NousResearch/ByteDance-Seed-OSS-36B-Alternate-Tokenizer @ e82f3d5c | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [c4ai-command-a-03-2025](fixtures/c4ai-command-a-03-2025/sets.toml) | CohereLabs/c4ai-command-a-03-2025 @ 066daaa1 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [c4ai-command-r-v01](fixtures/c4ai-command-r-v01/sets.toml) | CohereLabs/c4ai-command-r-v01 @ 760ddb6c | 1 | 3 | 491702 | 137178 | 210594 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [caller](fixtures/caller/sets.toml) | arcee-ai/Caller @ 2e71e25e | 1 | 3 | 531736 | 324160 | 142324 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-3b](fixtures/caprl-3b/sets.toml) | internlm/CapRL-3B @ f83373e8 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-eval-3b](fixtures/caprl-eval-3b/sets.toml) | internlm/CapRL-Eval-3B @ 484b8ca3 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-internvl3.5-8b](fixtures/caprl-internvl3.5-8b/sets.toml) | internlm/CapRL-InternVL3.5-8B @ e7ebc6db | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-qwen3vl-4b](fixtures/caprl-qwen3vl-4b/sets.toml) | internlm/CapRL-Qwen3VL-4B @ 1db1c1dd | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-video-4b](fixtures/caprl-video-4b/sets.toml) | internlm/CapRL-Video-4B @ d932a039 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [characterjudge](fixtures/characterjudge/sets.toml) | thu-coai/CharacterJudge @ f2272fa1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cogalign-internvl2-5-mpo-4b](fixtures/cogalign-internvl2-5-mpo-4b/sets.toml) | Salesforce/cogalign-internvl2_5-mpo-4b @ 2860db15 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cogalign-llava-ov-0-5b](fixtures/cogalign-llava-ov-0-5b/sets.toml) | Salesforce/cogalign-llava-ov-0_5b @ a7298047 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [command-a-plus-05-2026-bf16](fixtures/command-a-plus-05-2026-bf16/sets.toml) | CohereLabs/command-a-plus-05-2026-bf16 @ 5fb6fde5 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [command-a-reasoning-08-2025](fixtures/command-a-reasoning-08-2025/sets.toml) | CohereLabs/command-a-reasoning-08-2025 @ bd39aafe | 1 | 3 | 531736 | 483083 | 92696 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [contextpilot-14b](fixtures/contextpilot-14b/sets.toml) | tencent/ContextPilot-14B @ 8eafd835 | 8 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [contextpilot-e4b](fixtures/contextpilot-e4b/sets.toml) | tencent/ContextPilot-E4B @ a7ba41c1 | 1 | 2 | 531720 | 486159 | 92636 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosineverifier-tool-4b](fixtures/cosineverifier-tool-4b/sets.toml) | Nanbeige/CoSineVerifier-Tool-4B @ 393d18e0 | 1 | 2 | 531736 | 565691 | 23968 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos-reason1-7b](fixtures/cosmos-reason1-7b/sets.toml) | nvidia/Cosmos-Reason1-7B @ 375e2400 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos3-edge](fixtures/cosmos3-edge/sets.toml) | nvidia/Cosmos3-Edge @ 344d602b | 1 | 2 | 238489 | 12829 | 278544 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos3-edge-policy-droid](fixtures/cosmos3-edge-policy-droid/sets.toml) | nvidia/Cosmos3-Edge-Policy-DROID @ a7c7288f | 1 | 2 | 238489 | 12829 | 278544 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos3-super](fixtures/cosmos3-super/sets.toml) | nvidia/Cosmos3-Super @ f543c562 | 7 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [crispers-7b-v1](fixtures/crispers-7b-v1/sets.toml) | thu-coai/Crispers-7B-v1 @ 9dfb2cea | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cuda-autocomplete](fixtures/cuda-autocomplete/sets.toml) | nvidia/CUDA-Autocomplete @ 1878aa0a | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [d1-3b](fixtures/d1-3b/sets.toml) | LiquidAI/d1-3B @ 051bcc46 | 1 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deep-ignorance-unfiltered-instruct-test](fixtures/deep-ignorance-unfiltered-instruct-test/sets.toml) | EleutherAI/deep-ignorance-unfiltered-instruct-test @ 1e498ea5 | 5 | 2 | 531736 | 273678 | 166752 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deep-ignorance-unfiltered-instruct-test-v2](fixtures/deep-ignorance-unfiltered-instruct-test-v2/sets.toml) | EleutherAI/deep-ignorance-unfiltered-instruct-test-v2 @ 824398e6 | 1 | 2 | 531736 | 273678 | 166752 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deephermes-3-llama-3-8b-preview](fixtures/deephermes-3-llama-3-8b-preview/sets.toml) | NousResearch/DeepHermes-3-Llama-3-8B-Preview @ 53d902c7 | 2 | 3 | 531736 | 137178 | 199366 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [deephermes-3-mistral-24b-preview](fixtures/deephermes-3-mistral-24b-preview/sets.toml) | NousResearch/DeepHermes-3-Mistral-24B-Preview @ 48072dc6 | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deephermes-financial-fundamentals-prediction-specialist-atropos](fixtures/deephermes-financial-fundamentals-prediction-specialist-atropos/sets.toml) | NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos @ 31aa97fb | 5 | 3 | 531736 | 137178 | 199366 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [deepseek-prover-v2-7b](fixtures/deepseek-prover-v2-7b/sets.toml) | deepseek-ai/DeepSeek-Prover-V2-7B @ a8d9e144 | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-r1](fixtures/deepseek-r1/sets.toml) | deepseek-ai/DeepSeek-R1 @ 56d4cbbb | 2 | 3 | 493319 | 0 | 1 | glaive-v2, hand-written, swehero, tau2 |
| [deepseek-r1-0528](fixtures/deepseek-r1-0528/sets.toml) | deepseek-ai/DeepSeek-R1-0528 @ 4236a6af | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-r1-0528-qwen3-8b](fixtures/deepseek-r1-0528-qwen3-8b/sets.toml) | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B @ 6e8885a6 | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-r1-bf16](fixtures/deepseek-r1-bf16/sets.toml) | arcee-ai/DeepSeek-R1-bf16 @ c5eb2865 | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-r1-distill-llama-8b](fixtures/deepseek-r1-distill-llama-8b/sets.toml) | deepseek-ai/DeepSeek-R1-Distill-Llama-8B @ 6a6f4aa4 | 2 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [deepseek-r1-distill-qwen-1.5b](fixtures/deepseek-r1-distill-qwen-1.5b/sets.toml) | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B @ ad9f0ae0 | 4 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [deepseek-v3-0324](fixtures/deepseek-v3-0324/sets.toml) | deepseek-ai/DeepSeek-V3-0324 @ e9b33add | 2 | 3 | 531735 | 19299 | 328694 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3-0324-bf16](fixtures/deepseek-v3-0324-bf16/sets.toml) | arcee-ai/DeepSeek-V3-0324-bf16 @ 0fcb3b10 | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3.1](fixtures/deepseek-v3.1/sets.toml) | deepseek-ai/DeepSeek-V3.1 @ c0781d03 | 3 | 3 | 531735 | 19299 | 328694 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3.2-exp](fixtures/deepseek-v3.2-exp/sets.toml) | deepseek-ai/DeepSeek-V3.2-Exp @ 194c67e1 | 2 | 3 | 531735 | 184972 | 254532 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v4.1-flash](fixtures/deepseek-v4.1-flash/sets.toml) | deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42 | 1 | 1 | 531735 | 589659 | 1 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [devstral-2-123b-instruct-2512](fixtures/devstral-2-123b-instruct-2512/sets.toml) | mistralai/Devstral-2-123B-Instruct-2512 @ 1613bf01 | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dler-llama-nemotron-8b-merge-research](fixtures/dler-llama-nemotron-8b-merge-research/sets.toml) | nvidia/DLER-Llama-Nemotron-8B-Merge-Research @ 029e0043 | 1 | 3 | 471553 | 416850 | 176459 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dler-r1-1.5b-research](fixtures/dler-r1-1.5b-research/sets.toml) | nvidia/DLER-R1-1.5B-Research @ d0523143 | 2 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [dogr](fixtures/dogr/sets.toml) | tencent/DOGR @ ee374bd5 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dots.mocr](fixtures/dots.mocr/sets.toml) | dots-studio/dots.mocr @ e539fbb5 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dots.ocr.base](fixtures/dots.ocr.base/sets.toml) | dots-studio/dots.ocr.base @ ca7fe482 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dots3-note-prev](fixtures/dots3-note-prev/sets.toml) | dots-studio/dots3-note-prev @ 1e1e7b0c | 1 | 3 | 238490 | 589296 | 278839 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dr-venus-4b-rl](fixtures/dr-venus-4b-rl/sets.toml) | inclusionAI/DR-Venus-4B-RL @ a7da9211 | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dr-venus-4b-sft](fixtures/dr-venus-4b-sft/sets.toml) | inclusionAI/DR-Venus-4B-SFT @ 13837c51 | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dse-qwen-2.5-sota](fixtures/dse-qwen-2.5-sota/sets.toml) | lightonai/DSE_Qwen_2.5_SOTA @ 70fdc18b | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [e1-acereason-14b](fixtures/e1-acereason-14b/sets.toml) | Salesforce/E1-AceReason-14B @ 54537936 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [e1-code-14b](fixtures/e1-code-14b/sets.toml) | Salesforce/E1-Code-14B @ 9eebf1fb | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [eagle2-2b](fixtures/eagle2-2b/sets.toml) | nvidia/Eagle2-2B @ 3ab0a536 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [egm-4b-sft](fixtures/egm-4b-sft/sets.toml) | nvidia/EGM-4B-SFT @ 60cfbf34 | 2 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [egm-8b](fixtures/egm-8b/sets.toml) | nvidia/EGM-8B @ f5bc2e9d | 2 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ernie-4.5-0.3b-pt](fixtures/ernie-4.5-0.3b-pt/sets.toml) | baidu/ERNIE-4.5-0.3B-PT @ b565cf6c | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ernie-4.5-21b-a3b-pt](fixtures/ernie-4.5-21b-a3b-pt/sets.toml) | baidu/ERNIE-4.5-21B-A3B-PT @ 87db9548 | 4 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ernie-4.5-21b-a3b-thinking](fixtures/ernie-4.5-21b-a3b-thinking/sets.toml) | baidu/ERNIE-4.5-21B-A3B-Thinking @ 4341bb42 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ernie-image-aes](fixtures/ernie-image-aes/sets.toml) | baidu/ERNIE-Image-Aes @ 8a0bb5ea | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [evisrag-7b](fixtures/evisrag-7b/sets.toml) | openbmb/EVisRAG-7B @ a3a1f464 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [exaone-3.5-7.8b-instruct](fixtures/exaone-3.5-7.8b-instruct/sets.toml) | LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct @ 553ea250 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [exaone-4.0-32b](fixtures/exaone-4.0-32b/sets.toml) | LGAI-EXAONE/EXAONE-4.0-32B @ a1d54d1c | 3 | 3 | 531735 | 576772 | 8 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [exaone-4.5-33b](fixtures/exaone-4.5-33b/sets.toml) | LGAI-EXAONE/EXAONE-4.5-33B @ 570aa4b1 | 1 | 2 | 531735 | 503634 | 86026 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-e-3b-base-prequantized](fixtures/falcon-e-3b-base-prequantized/sets.toml) | tiiuae/Falcon-E-3B-Base-prequantized @ 5feb4d43 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [falcon-h1-0.5b-instruct](fixtures/falcon-h1-0.5b-instruct/sets.toml) | tiiuae/Falcon-H1-0.5B-Instruct @ 8f2587ca | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-1.5b-deep-instruct](fixtures/falcon-h1-1.5b-deep-instruct/sets.toml) | tiiuae/Falcon-H1-1.5B-Deep-Instruct @ b6648636 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-1.5b-instruct](fixtures/falcon-h1-1.5b-instruct/sets.toml) | tiiuae/Falcon-H1-1.5B-Instruct @ 80ebc50d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-34b-instruct](fixtures/falcon-h1-34b-instruct/sets.toml) | tiiuae/Falcon-H1-34B-Instruct @ 6e6890e2 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-3b-instruct](fixtures/falcon-h1-3b-instruct/sets.toml) | tiiuae/Falcon-H1-3B-Instruct @ 01087ec4 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-7b-instruct](fixtures/falcon-h1-7b-instruct/sets.toml) | tiiuae/Falcon-H1-7B-Instruct @ 41e72f27 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-90m-instruct](fixtures/falcon-h1-tiny-90m-instruct/sets.toml) | tiiuae/Falcon-H1-Tiny-90M-Instruct @ e6389502 | 2 | 2 | 531736 | 429360 | 140619 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-multilingual-100m-instruct](fixtures/falcon-h1-tiny-multilingual-100m-instruct/sets.toml) | tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct @ 295b1eb9 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-r-0.6b](fixtures/falcon-h1-tiny-r-0.6b/sets.toml) | tiiuae/Falcon-H1-Tiny-R-0.6B @ ccf3d617 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-r-0.6b-pre-grpo](fixtures/falcon-h1-tiny-r-0.6b-pre-grpo/sets.toml) | tiiuae/Falcon-H1-Tiny-R-0.6B-pre-GRPO @ 24ec8732 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-r-90m](fixtures/falcon-h1-tiny-r-90m/sets.toml) | tiiuae/Falcon-H1-Tiny-R-90M @ 7385612b | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-tool-calling-90m](fixtures/falcon-h1-tiny-tool-calling-90m/sets.toml) | tiiuae/Falcon-H1-Tiny-Tool-Calling-90M @ fc099d79 | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [falcon-h1r-7b](fixtures/falcon-h1r-7b/sets.toml) | tiiuae/Falcon-H1R-7B @ a6f74bf1 | 1 | 2 | 531736 | 442119 | 140740 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-mamba-7b-instruct](fixtures/falcon-mamba-7b-instruct/sets.toml) | tiiuae/falcon-mamba-7b-instruct @ b250fc93 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon3-10b-base-1.58bit-prequantized](fixtures/falcon3-10b-base-1.58bit-prequantized/sets.toml) | tiiuae/Falcon3-10B-Base-1.58bit-prequantized @ 87126c86 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [fara-7b](fixtures/fara-7b/sets.toml) | microsoft/Fara-7B @ 4fca9cde | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fara1.5-27b](fixtures/fara1.5-27b/sets.toml) | microsoft/Fara1.5-27B @ 299c8406 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fara1.5-9b](fixtures/fara1.5-9b/sets.toml) | microsoft/Fara1.5-9B @ 1a93677c | 2 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fare-8b](fixtures/fare-8b/sets.toml) | Salesforce/FARE-8B @ a0f4f685 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [finetuned-reworded-qwen3-4b-public](fixtures/finetuned-reworded-qwen3-4b-public/sets.toml) | EleutherAI/finetuned-reworded-Qwen3-4B-public @ 1715bca8 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fleming-vl-38b](fixtures/fleming-vl-38b/sets.toml) | IQuestLab/Fleming-VL-38B @ 6602b2e5 | 1 | 2 | 199970 | 145044 | 320255 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [fleming-vl-8b](fixtures/fleming-vl-8b/sets.toml) | IQuestLab/Fleming-VL-8B @ 801e2bef | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [flex-public-7b-1t](fixtures/flex-public-7b-1t/sets.toml) | allenai/Flex-public-7B-1T @ f78a73be | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [flex-reddit-2x7b-1t](fixtures/flex-reddit-2x7b-1t/sets.toml) | allenai/Flex-reddit-2x7B-1T @ 55379e32 | 6 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [flexolmo-7x7b-1t](fixtures/flexolmo-7x7b-1t/sets.toml) | allenai/FlexOlmo-7x7B-1T @ 0dabfa2e | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [flexolmo-7x7b-1t-rt](fixtures/flexolmo-7x7b-1t-rt/sets.toml) | allenai/FlexOlmo-7x7B-1T-RT @ 193b1e91 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [frogboss-32b-2510](fixtures/frogboss-32b-2510/sets.toml) | microsoft/FrogBoss-32B-2510 @ cc930952 | 1 | 2 | 238490 | 582708 | 285427 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [frogmini-14b-2510](fixtures/frogmini-14b-2510/sets.toml) | microsoft/FrogMini-14B-2510 @ bde8b2b5 | 1 | 2 | 238490 | 409289 | 458846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fun-asr-nano-2512-vllm](fixtures/fun-asr-nano-2512-vllm/sets.toml) | allendou/Fun-ASR-Nano-2512-vllm @ e718b36e | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fun-audio-chat-8b](fixtures/fun-audio-chat-8b/sets.toml) | FunAudioLLM/Fun-Audio-Chat-8B @ 7bf72dc7 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [functiongemma-270m-it](fixtures/functiongemma-270m-it/sets.toml) | google/functiongemma-270m-it @ 39eccb09 | 1 | 2 | 238483 | 246750 | 499217 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [game-rl-internvl3-8b](fixtures/game-rl-internvl3-8b/sets.toml) | OpenMOSS-Team/Game-RL-InternVL3-8B @ defc1e86 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [game-rl-llava-ov-7b](fixtures/game-rl-llava-ov-7b/sets.toml) | OpenMOSS-Team/Game-RL-LLaVA-OV-7B @ 464e45f0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [game-rl-qwen2.5-vl-7b](fixtures/game-rl-qwen2.5-vl-7b/sets.toml) | OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B @ 205b5934 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gelab-zero-4b-preview](fixtures/gelab-zero-4b-preview/sets.toml) | stepfun-ai/GELab-Zero-4B-preview @ 838a527e | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-1.1-2b-it](fixtures/gemma-1.1-2b-it/sets.toml) | google/gemma-1.1-2b-it @ d750f5ec | 1 | 3 | 86317 | 0 | 238454 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [gemma-2-2b-it](fixtures/gemma-2-2b-it/sets.toml) | google/gemma-2-2b-it @ 299a8560 | 1 | 3 | 86317 | 47052 | 530046 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [gemma-3-12b-it-qat-q4-0-unquantized](fixtures/gemma-3-12b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-3-12b-it-qat-q4_0-unquantized @ 68f7ee4f | 3 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-1b-it](fixtures/gemma-3-1b-it/sets.toml) | google/gemma-3-1b-it @ dcc83ea8 | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-1b-it-qat-q4-0-unquantized](fixtures/gemma-3-1b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-3-1b-it-qat-q4_0-unquantized @ a6692c19 | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-270m-it](fixtures/gemma-3-270m-it/sets.toml) | google/gemma-3-270m-it @ ac82b4e8 | 2 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-4b-it](fixtures/gemma-3-4b-it/sets.toml) | google/gemma-3-4b-it @ 093f9f38 | 3 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3n-e2b-it](fixtures/gemma-3n-e2b-it/sets.toml) | google/gemma-3n-E2B-it @ 5e092ebc | 2 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-4-12b-it](fixtures/gemma-4-12b-it/sets.toml) | google/gemma-4-12B-it @ 707f0a3b | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-12b-it-qat-q4-0-unquantized-assistant](fixtures/gemma-4-12b-it-qat-q4-0-unquantized-assistant/sets.toml) | google/gemma-4-12B-it-qat-q4_0-unquantized-assistant @ 18934064 | 1 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-26b-a4b-it](fixtures/gemma-4-26b-a4b-it/sets.toml) | google/gemma-4-26B-A4B-it @ 4d7ae498 | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-31b-it-qat-q4-0-unquantized](fixtures/gemma-4-31b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-4-31B-it-qat-q4_0-unquantized @ 1e4d8bee | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-31b-it-qat-q4-0-unquantized-assistant](fixtures/gemma-4-31b-it-qat-q4-0-unquantized-assistant/sets.toml) | google/gemma-4-31B-it-qat-q4_0-unquantized-assistant @ 96d4c8ca | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-e2b-it-qat-q4-0-unquantized](fixtures/gemma-4-e2b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-4-E2B-it-qat-q4_0-unquantized @ 6befbaca | 2 | 2 | 238484 | 478241 | 389900 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-4-e4b-it](fixtures/gemma-4-e4b-it/sets.toml) | google/gemma-4-E4B-it @ ee0ef602 | 2 | 2 | 238484 | 478241 | 389900 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4-32b-0414](fixtures/glm-4-32b-0414/sets.toml) | zai-org/GLM-4-32B-0414 @ 077b5c2f | 1 | 3 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4-9b-0414](fixtures/glm-4-9b-0414/sets.toml) | zai-org/GLM-4-9B-0414 @ 645b8482 | 1 | 3 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4-9b-chat-hf](fixtures/glm-4-9b-chat-hf/sets.toml) | zai-org/glm-4-9b-chat-hf @ 8599336f | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.1v-9b-base](fixtures/glm-4.1v-9b-base/sets.toml) | zai-org/GLM-4.1V-9B-Base @ 5e8d1942 | 1 | 3 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.5v](fixtures/glm-4.5v/sets.toml) | zai-org/GLM-4.5V @ ed47433b | 1 | 3 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6](fixtures/glm-4.6/sets.toml) | zai-org/GLM-4.6 @ be721948 | 3 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6v](fixtures/glm-4.6v/sets.toml) | zai-org/GLM-4.6V @ 4e2d47eb | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6v-flash](fixtures/glm-4.6v-flash/sets.toml) | zai-org/GLM-4.6V-Flash @ 411bb4d7 | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.7](fixtures/glm-4.7/sets.toml) | zai-org/GLM-4.7 @ 602d01ef | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.7-flash](fixtures/glm-4.7-flash/sets.toml) | zai-org/GLM-4.7-Flash @ 7dd20894 | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5](fixtures/glm-5/sets.toml) | zai-org/GLM-5 @ c183ef8c | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.1](fixtures/glm-5.1/sets.toml) | zai-org/GLM-5.1 @ 26e1bd6e | 1 | 2 | 238490 | 163746 | 677804 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.2](fixtures/glm-5.2/sets.toml) | zai-org/GLM-5.2 @ cf457fa7 | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.3](fixtures/glm-5.3/sets.toml) | zai-org/GLM-5.3 @ aca966e4 | 2 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.3-flash](fixtures/glm-5.3-flash/sets.toml) | zai-org/GLM-5.3-Flash @ eb9eb208 | 2 | 1 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-asr-nano-2512](fixtures/glm-asr-nano-2512/sets.toml) | zai-org/GLM-ASR-Nano-2512 @ 61ba4e0b | 1 | 2 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-ocr](fixtures/glm-ocr/sets.toml) | zai-org/GLM-OCR @ 2e85a628 | 1 | 2 | 238490 | 328645 | 390261 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-z1-32b-0414](fixtures/glm-z1-32b-0414/sets.toml) | zai-org/GLM-Z1-32B-0414 @ 8eb28589 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [glm-z1-rumination-32b-0414](fixtures/glm-z1-rumination-32b-0414/sets.toml) | zai-org/GLM-Z1-Rumination-32B-0414 @ 6ae9ac61 | 1 | 3 | 531736 | 328645 | 111785 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glyph](fixtures/glyph/sets.toml) | zai-org/Glyph @ 074a0727 | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.1-3b-a800m-instruct](fixtures/granite-3.1-3b-a800m-instruct/sets.toml) | ibm-granite/granite-3.1-3b-a800m-instruct @ a0278068 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.1-8b-instruct](fixtures/granite-3.1-8b-instruct/sets.toml) | ibm-granite/granite-3.1-8b-instruct @ 4009206d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.2-8b-instruct](fixtures/granite-3.2-8b-instruct/sets.toml) | ibm-granite/granite-3.2-8b-instruct @ 610d8c6e | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.2-8b-instruct-preview](fixtures/granite-3.2-8b-instruct-preview/sets.toml) | ibm-granite/granite-3.2-8b-instruct-preview @ 191a764d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.3-2b-instruct](fixtures/granite-3.3-2b-instruct/sets.toml) | ibm-granite/granite-3.3-2b-instruct @ 707f574c | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.3-8b-math-prm-v2](fixtures/granite-3.3-8b-math-prm-v2/sets.toml) | ibm-granite/granite-3.3-8b-math-prm-v2 @ 301abdda | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-1b-speech](fixtures/granite-4.0-1b-speech/sets.toml) | ibm-granite/granite-4.0-1b-speech @ bd87ab86 | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [granite-4.0-3b-vision](fixtures/granite-4.0-3b-vision/sets.toml) | ibm-granite/granite-4.0-3b-vision @ bf108f36 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-h-1b](fixtures/granite-4.0-h-1b/sets.toml) | ibm-granite/granite-4.0-h-1b @ d18cca4c | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-h-3b-ar](fixtures/granite-4.0-h-3b-ar/sets.toml) | ibm-research/granite-4.0-h-3b-ar @ b32260f6 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-h-tiny](fixtures/granite-4.0-h-tiny/sets.toml) | ibm-granite/granite-4.0-h-tiny @ 791e0d3d | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-tiny-preview](fixtures/granite-4.0-tiny-preview/sets.toml) | ibm-granite/granite-4.0-tiny-preview @ 4ec5f963 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.1-3b](fixtures/granite-4.1-3b/sets.toml) | ibm-granite/granite-4.1-3b @ c0650403 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-docling-258m](fixtures/granite-docling-258m/sets.toml) | ibm-granite/granite-docling-258M @ 982fe3b4 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-guardian-3.2-3b-a800m](fixtures/granite-guardian-3.2-3b-a800m/sets.toml) | ibm-granite/granite-guardian-3.2-3b-a800m @ 3de033d8 | 1 | 3 | 86321 | 0 | 238450 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [granite-guardian-3.2-5b](fixtures/granite-guardian-3.2-5b/sets.toml) | ibm-granite/granite-guardian-3.2-5b @ d9790371 | 1 | 3 | 86321 | 0 | 238450 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [granite-guardian-3.3-8b](fixtures/granite-guardian-3.3-8b/sets.toml) | ibm-granite/granite-guardian-3.3-8b @ b3421eda | 1 | 3 | 319375 | 0 | 182628 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [granite-guardian-4.1-8b](fixtures/granite-guardian-4.1-8b/sets.toml) | ibm-granite/granite-guardian-4.1-8b @ ab01ccca | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-speech-3.2-8b](fixtures/granite-speech-3.2-8b/sets.toml) | ibm-granite/granite-speech-3.2-8b @ d5835026 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-speech-3.3-2b](fixtures/granite-speech-3.3-2b/sets.toml) | ibm-granite/granite-speech-3.3-2b @ 4ac2f02f | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-speech-3.3-8b](fixtures/granite-speech-3.3-8b/sets.toml) | ibm-granite/granite-speech-3.3-8b @ 5670dcfa | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-speech-4.1-2b-plus](fixtures/granite-speech-4.1-2b-plus/sets.toml) | ibm-granite/granite-speech-4.1-2b-plus @ 1454e6e1 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.1-2b-preview](fixtures/granite-vision-3.1-2b-preview/sets.toml) | ibm-granite/granite-vision-3.1-2b-preview @ bd1c154e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.2-2b](fixtures/granite-vision-3.2-2b/sets.toml) | ibm-granite/granite-vision-3.2-2b @ ec894191 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.3-2b](fixtures/granite-vision-3.3-2b/sets.toml) | ibm-granite/granite-vision-3.3-2b @ bf66a824 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.3-2b-chart2csv-preview](fixtures/granite-vision-3.3-2b-chart2csv-preview/sets.toml) | ibm-granite/granite-vision-3.3-2b-chart2csv-preview @ 298ed4d8 | 1 | 2 | 498897 | 328662 | 144607 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-4.1-4b](fixtures/granite-vision-4.1-4b/sets.toml) | ibm-granite/granite-vision-4.1-4b @ 37d591f0 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [graspmolmo](fixtures/graspmolmo/sets.toml) | allenai/GraspMolmo @ 9aa184d6 | 1 | 3 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [h2o-lightning-4b](fixtures/h2o-lightning-4b/sets.toml) | h2oai/h2o-lightning-4b @ 672dc01e | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [harc-llama-3.1-8b-instruct](fixtures/harc-llama-3.1-8b-instruct/sets.toml) | microsoft/HARC-Llama-3.1-8B-Instruct @ c6f25949 | 1 | 2 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [harc-qwen2.5-7b-instruct](fixtures/harc-qwen2.5-7b-instruct/sets.toml) | microsoft/HARC-Qwen2.5-7B-Instruct @ f650ffae | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-3-llama-3.1-8b](fixtures/hermes-3-llama-3.1-8b/sets.toml) | NousResearch/Hermes-3-Llama-3.1-8B @ 896ea440 | 1 | 3 | 531736 | 324160 | 142324 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4-14b](fixtures/hermes-4-14b/sets.toml) | NousResearch/Hermes-4-14B @ d6ce765c | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4-405b](fixtures/hermes-4-405b/sets.toml) | NousResearch/Hermes-4-405B @ 88e3dce0 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4-70b](fixtures/hermes-4-70b/sets.toml) | NousResearch/Hermes-4-70B @ d5dec2bd | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4.3-36b](fixtures/hermes-4.3-36b/sets.toml) | NousResearch/Hermes-4.3-36B @ 3899db2b | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4.3-36b-centralized](fixtures/hermes-4.3-36b-centralized/sets.toml) | NousResearch/Hermes-4.3-36B-centralized @ 8f3c006f | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [himed-8b](fixtures/himed-8b/sets.toml) | FreedomIntelligence/HiMed-8B @ 0573f67b | 1 | 2 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hpd-parsing](fixtures/hpd-parsing/sets.toml) | PaddlePaddle/HPD-Parsing @ 91de8005 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-27b](fixtures/huatuogpt-3-27b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-27B @ e759ddae | 1 | 2 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-8b](fixtures/huatuogpt-3-8b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-8B @ 507f63a1 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-9b](fixtures/huatuogpt-3-9b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-9B @ 6870b60e | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-grader-8b](fixtures/huatuogpt-3-grader-8b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-Grader-8B @ 8b17b10d | 1 | 2 | 531736 | 541389 | 6580 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-1.8b-instruct](fixtures/hunyuan-1.8b-instruct/sets.toml) | tencent/Hunyuan-1.8B-Instruct @ de940610 | 6 | 3 | 531736 | 570174 | 6605 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-7b-instruct](fixtures/hunyuan-7b-instruct/sets.toml) | tencent/Hunyuan-7B-Instruct @ 6fd6ecb0 | 1 | 3 | 531735 | 328662 | 111769 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-a13b-instruct](fixtures/hunyuan-a13b-instruct/sets.toml) | tencent/Hunyuan-A13B-Instruct @ 290ddb9a | 1 | 3 | 531735 | 328662 | 111769 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-mt-7b](fixtures/hunyuan-mt-7b/sets.toml) | tencent/Hunyuan-MT-7B @ 9305c783 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuanocr](fixtures/hunyuanocr/sets.toml) | tencent/HunyuanOCR @ 47644ecc | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy-mt2-1.8b](fixtures/hy-mt2-1.8b/sets.toml) | tencent/Hy-MT2-1.8B @ 9a341cd1 | 4 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [hy-mt2-30b-a3b](fixtures/hy-mt2-30b-a3b/sets.toml) | tencent/Hy-MT2-30B-A3B @ d3ead4db | 1 | 2 | 238490 | 569901 | 285354 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy-mt2-7b](fixtures/hy-mt2-7b/sets.toml) | tencent/Hy-MT2-7B @ 9b0eb4e8 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy3](fixtures/hy3/sets.toml) | tencent/Hy3 @ a960ebc3 | 1 | 2 | 238490 | 569901 | 285354 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy3-preview-base](fixtures/hy3-preview-base/sets.toml) | tencent/Hy3-preview-Base @ 54a62bb0 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [hy4-preview](fixtures/hy4-preview/sets.toml) | tencent/Hy4-preview @ 705d81ee | 1 | 1 | 238489 | 589298 | 278838 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hyperclovax-seed-omni-8b](fixtures/hyperclovax-seed-omni-8b/sets.toml) | naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B @ f1240321 | 1 | 2 | 238490 | 582728 | 285407 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hyperclovax-seed-text-instruct-0.5b](fixtures/hyperclovax-seed-text-instruct-0.5b/sets.toml) | naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B @ 3da5046f | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [hyperclovax-seed-think-14b](fixtures/hyperclovax-seed-think-14b/sets.toml) | naver-hyperclovax/HyperCLOVAX-SEED-Think-14B @ 9b74e35d | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [idefics3-8b-llama3](fixtures/idefics3-8b-llama3/sets.toml) | HuggingFaceM4/Idefics3-8B-Llama3 @ fddb4ff7 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [if-critic-14b](fixtures/if-critic-14b/sets.toml) | thu-coai/IF-CRITIC-14B @ 091e3cd5 | 3 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [infllm-v2-short-dense-base](fixtures/infllm-v2-short-dense-base/sets.toml) | openbmb/InfLLM-V2-Short-Dense-Base @ aba491d0 | 2 | 2 | 531736 | 328660 | 111770 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [inkling](fixtures/inkling/sets.toml) | thinkingmachines/Inkling @ 828496ee | 2 | 3 | 238489 | 589659 | 278477 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intent-aware-lfqa-llama3-8b-intent-explicit](fixtures/intent-aware-lfqa-llama3-8b-intent-explicit/sets.toml) | allenai/intent-aware-lfqa-llama3-8b-intent-explicit @ fdf98cd4 | 4 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intent-aware-lfqa-qwen3-4b-intent-explicit](fixtures/intent-aware-lfqa-qwen3-4b-intent-explicit/sets.toml) | allenai/intent-aware-lfqa-qwen3-4b-intent-explicit @ 07bd65c9 | 8 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intern-decision-0.8b](fixtures/intern-decision-0.8b/sets.toml) | internlm/Intern-Decision-0.8B @ 85a0cc5a | 2 | 2 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intern-decision-4b](fixtures/intern-decision-4b/sets.toml) | internlm/Intern-Decision-4B @ 0e5e6aa7 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl2-5-2b-mpo-hf](fixtures/internvl2-5-2b-mpo-hf/sets.toml) | OpenGVLab/InternVL2_5-2B-MPO-hf @ 7118a7ea | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-1b-hf](fixtures/internvl3-1b-hf/sets.toml) | OpenGVLab/InternVL3-1B-hf @ 014c0583 | 6 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-1b-instruct](fixtures/internvl3-1b-instruct/sets.toml) | OpenGVLab/InternVL3-1B-Instruct @ f91c6391 | 12 | 3 | 199970 | 145044 | 320255 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [internvl3-38b](fixtures/internvl3-38b/sets.toml) | OpenGVLab/InternVL3-38B @ b99bcba4 | 2 | 3 | 199970 | 145044 | 320255 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [internvl3-5-30b-a3b-instruct](fixtures/internvl3-5-30b-a3b-instruct/sets.toml) | OpenGVLab/InternVL3_5-30B-A3B-Instruct @ 83f9a51d | 4 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-5-8b](fixtures/internvl3-5-8b/sets.toml) | OpenGVLab/InternVL3_5-8B @ 9bb6a56a | 14 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-5-8b-hf](fixtures/internvl3-5-8b-hf/sets.toml) | OpenGVLab/InternVL3_5-8B-HF @ 741a7d03 | 8 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-5-8b-instruct](fixtures/internvl3-5-8b-instruct/sets.toml) | OpenGVLab/InternVL3_5-8B-Instruct @ 27506812 | 12 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-8b](fixtures/internvl3-8b/sets.toml) | OpenGVLab/InternVL3-8B @ 853e3a79 | 3 | 3 | 199970 | 145044 | 320255 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [iquest-q1](fixtures/iquest-q1/sets.toml) | IQuestLab/IQuest-Q1 @ 5c21b063 | 1 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [isaac-0.2-1b](fixtures/isaac-0.2-1b/sets.toml) | PerceptronAI/Isaac-0.2-1B @ bf8a8b26 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ising-calibration-1-35b-a3b](fixtures/ising-calibration-1-35b-a3b/sets.toml) | nvidia/Ising-Calibration-1-35B-A3B @ b07b7d22 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ising-calibration-1.5-31b-bf16](fixtures/ising-calibration-1.5-31b-bf16/sets.toml) | nvidia/Ising-Calibration-1.5-31B-BF16 @ b9b4f513 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [jamba-tiny-dev](fixtures/jamba-tiny-dev/sets.toml) | ai21labs/Jamba-tiny-dev @ ed303361 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [januscoder-14b](fixtures/januscoder-14b/sets.toml) | internlm/JanusCoder-14B @ 28487c47 | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [januscoderv-7b](fixtures/januscoderv-7b/sets.toml) | internlm/JanusCoderV-7B @ a78918c8 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [januscoderv-8b](fixtures/januscoderv-8b/sets.toml) | internlm/JanusCoderV-8B @ 24935a7a | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [jedi-3b-1080p](fixtures/jedi-3b-1080p/sets.toml) | xlangai/Jedi-3B-1080p @ 4aef56c6 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [jedi-7b-1080p](fixtures/jedi-7b-1080p/sets.toml) | xlangai/Jedi-7B-1080p @ 7ccf6e40 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [jet-nemotron-2b](fixtures/jet-nemotron-2b/sets.toml) | jet-ai/Jet-Nemotron-2B @ afd4ee28 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [justrl-ii-base-model](fixtures/justrl-ii-base-model/sets.toml) | openbmb/JustRL-II-base-model @ b558bbc7 | 1 | 2 | 238490 | 589265 | 278870 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k-exaone-2.0-750b-a37b](fixtures/k-exaone-2.0-750b-a37b/sets.toml) | LGAI-EXAONE/K-EXAONE-2.0-750B-A37B @ fec0c8dd | 2 | 2 | 238489 | 503273 | 364863 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k-exaone-236b-a23b](fixtures/k-exaone-236b-a23b/sets.toml) | LGAI-EXAONE/K-EXAONE-236B-A23B @ 61e6d578 | 1 | 2 | 531735 | 503634 | 86026 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k2-horizon-0.9b](fixtures/k2-horizon-0.9b/sets.toml) | IFM/K2-Horizon-0.9B @ 383cb663 | 1 | 2 | 238442 | 0 | 278524 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [k2-horizon-36b](fixtures/k2-horizon-36b/sets.toml) | IFM/K2-Horizon-36B @ cca48b66 | 3 | 2 | 70766 | 12814 | 254087 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [k2-horizon-7b](fixtures/k2-horizon-7b/sets.toml) | IFM/K2-Horizon-7B @ 3b6ae394 | 2 | 2 | 70766 | 1908 | 254113 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [k2-think](fixtures/k2-think/sets.toml) | IFM/K2-Think @ 9e218ac1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k2-think-v2](fixtures/k2-think-v2/sets.toml) | IFM/K2-Think-V2 @ f86e5d11 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [k2-type-0.9b](fixtures/k2-type-0.9b/sets.toml) | IFM/K2-Type-0.9B @ 0648c43e | 1 | 2 | 86274 | 0 | 238497 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [k2-v2-instruct](fixtures/k2-v2-instruct/sets.toml) | IFM/K2-V2-Instruct @ 6ef3e31c | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kanana-1.5-15.7b-a3b-base](fixtures/kanana-1.5-15.7b-a3b-base/sets.toml) | kakaocorp/kanana-1.5-15.7b-a3b-base @ c9f2bbfb | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kanana-1.5-15.7b-a3b-instruct](fixtures/kanana-1.5-15.7b-a3b-instruct/sets.toml) | kakaocorp/kanana-1.5-15.7b-a3b-instruct @ df9d415c | 1 | 3 | 531735 | 184972 | 254532 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-1.5-8b-base](fixtures/kanana-1.5-8b-base/sets.toml) | kakaocorp/kanana-1.5-8b-base @ 5a5aad57 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kanana-1.5-8b-instruct-2505](fixtures/kanana-1.5-8b-instruct-2505/sets.toml) | kakaocorp/kanana-1.5-8b-instruct-2505 @ c963a5f4 | 2 | 3 | 531735 | 184972 | 254532 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-2-30b-a3b-instruct](fixtures/kanana-2-30b-a3b-instruct/sets.toml) | kakaocorp/kanana-2-30b-a3b-instruct @ bf0054a9 | 1 | 2 | 531736 | 589657 | 2 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-2-30b-a3b-instruct-2601](fixtures/kanana-2-30b-a3b-instruct-2601/sets.toml) | kakaocorp/kanana-2-30b-a3b-instruct-2601 @ 4a781fe5 | 1 | 2 | 531736 | 589657 | 2 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-2-30b-a3b-thinking-2601](fixtures/kanana-2-30b-a3b-thinking-2601/sets.toml) | kakaocorp/kanana-2-30b-a3b-thinking-2601 @ cdcbf8b8 | 2 | 2 | 531736 | 12885 | 11 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-2-3b-instruct](fixtures/kanana-2-3b-instruct/sets.toml) | kakaocorp/kanana-2-3b-instruct @ 6a5d7889 | 1 | 2 | 502928 | 461740 | 86231 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-safeguard-8b](fixtures/kanana-safeguard-8b/sets.toml) | kakaocorp/kanana-safeguard-8b @ 2f4a6864 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kanana-safeguard-siren-8b](fixtures/kanana-safeguard-siren-8b/sets.toml) | kakaocorp/kanana-safeguard-siren-8b @ dc65aab3 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kernelllm](fixtures/kernelllm/sets.toml) | facebook/KernelLLM @ 26d1191a | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kimi-dev-72b](fixtures/kimi-dev-72b/sets.toml) | moonshotai/Kimi-Dev-72B @ 8791d798 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kimi-k3](fixtures/kimi-k3/sets.toml) | moonshotai/Kimi-K3 @ f831ab66 | 1 | 1 | 38439 | 54418 | 1 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [laguna-m.1](fixtures/laguna-m.1/sets.toml) | poolside/Laguna-M.1 @ 2bf8a4ab | 1 | 2 | 238490 | 490452 | 364803 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [laguna-s-2.1](fixtures/laguna-s-2.1/sets.toml) | poolside/Laguna-S-2.1 @ 0f573140 | 1 | 2 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [laguna-xs-2.1](fixtures/laguna-xs-2.1/sets.toml) | poolside/Laguna-XS-2.1 @ c5f36269 | 1 | 2 | 238490 | 576475 | 278780 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [laguna-xs.2](fixtures/laguna-xs.2/sets.toml) | poolside/Laguna-XS.2 @ a397bde0 | 1 | 3 | 238490 | 490452 | 364803 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-1.2b](fixtures/lfm2-1.2b/sets.toml) | LiquidAI/LFM2-1.2B @ 40f3da0d | 3 | 3 | 243122 | 576772 | 273851 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-1.2b-rag](fixtures/lfm2-1.2b-rag/sets.toml) | LiquidAI/LFM2-1.2B-RAG @ cdbd46d3 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-2.6b](fixtures/lfm2-2.6b/sets.toml) | LiquidAI/LFM2-2.6B @ 36ed799f | 1 | 3 | 243122 | 576772 | 273851 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-2.6b-exp](fixtures/lfm2-2.6b-exp/sets.toml) | LiquidAI/LFM2-2.6B-Exp @ 2251c865 | 1 | 2 | 243122 | 576772 | 273851 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-2.6b-longevity](fixtures/lfm2-2.6b-longevity/sets.toml) | LiquidAI/LFM2-2.6B-Longevity @ 05b12a8d | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-24b-a2b](fixtures/lfm2-24b-a2b/sets.toml) | LiquidAI/LFM2-24B-A2B @ a3bbacd9 | 1 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-350m-extract](fixtures/lfm2-350m-extract/sets.toml) | LiquidAI/LFM2-350M-Extract @ d99a6f06 | 4 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-350m-math](fixtures/lfm2-350m-math/sets.toml) | LiquidAI/LFM2-350M-Math @ 8eb8b708 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-350m-pii-extract-jp](fixtures/lfm2-350m-pii-extract-jp/sets.toml) | LiquidAI/LFM2-350M-PII-Extract-JP @ c9d13581 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-8b-a1b](fixtures/lfm2-8b-a1b/sets.toml) | LiquidAI/LFM2-8B-A1B @ c1c44ff9 | 1 | 3 | 243122 | 576772 | 273851 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-vl-1.6b](fixtures/lfm2-vl-1.6b/sets.toml) | LiquidAI/LFM2-VL-1.6B @ 2c2e7f9b | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2-vl-3b](fixtures/lfm2-vl-3b/sets.toml) | LiquidAI/LFM2-VL-3B @ 97a81f5c | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-1.2b-base](fixtures/lfm2.5-1.2b-base/sets.toml) | LiquidAI/LFM2.5-1.2B-Base @ 7453bca9 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-1.2b-instruct](fixtures/lfm2.5-1.2b-instruct/sets.toml) | LiquidAI/LFM2.5-1.2B-Instruct @ 0f604ada | 2 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-1.2b-jp](fixtures/lfm2.5-1.2b-jp/sets.toml) | LiquidAI/LFM2.5-1.2B-JP @ 58ef35ca | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-1.2b-jp-202606](fixtures/lfm2.5-1.2b-jp-202606/sets.toml) | LiquidAI/LFM2.5-1.2B-JP-202606 @ 52b8b447 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-2.6b](fixtures/lfm2.5-2.6b/sets.toml) | LiquidAI/LFM2.5-2.6B @ 654f9463 | 1 | 2 | 243122 | 12885 | 273855 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-2.6b-base](fixtures/lfm2.5-2.6b-base/sets.toml) | LiquidAI/LFM2.5-2.6B-Base @ c57bdaed | 1 | 2 | 238490 | 576772 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-230m](fixtures/lfm2.5-230m/sets.toml) | LiquidAI/LFM2.5-230M @ 40cb2ad3 | 1 | 2 | 238490 | 589657 | 278478 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-230m-base](fixtures/lfm2.5-230m-base/sets.toml) | LiquidAI/LFM2.5-230M-Base @ 9d2be551 | 1 | 2 | 238490 | 576772 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-350m](fixtures/lfm2.5-350m/sets.toml) | LiquidAI/LFM2.5-350M @ 9e6c6ccf | 1 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-350m-base](fixtures/lfm2.5-350m-base/sets.toml) | LiquidAI/LFM2.5-350M-Base @ 9960764e | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-8b-a1b](fixtures/lfm2.5-8b-a1b/sets.toml) | LiquidAI/LFM2.5-8B-A1B @ 5dd22602 | 1 | 2 | 238490 | 589657 | 278478 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-8b-a1b-base](fixtures/lfm2.5-8b-a1b-base/sets.toml) | LiquidAI/LFM2.5-8B-A1B-Base @ 6cfb83b0 | 1 | 2 | 238490 | 576772 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-vl-1.6b](fixtures/lfm2.5-vl-1.6b/sets.toml) | LiquidAI/LFM2.5-VL-1.6B @ 919fde3d | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-vl-1.6b-extract](fixtures/lfm2.5-vl-1.6b-extract/sets.toml) | LiquidAI/LFM2.5-VL-1.6B-Extract @ 21073aa6 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-vl-3b](fixtures/lfm2.5-vl-3b/sets.toml) | LiquidAI/LFM2.5-VL-3B @ 35a118d9 | 1 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-vl-450m](fixtures/lfm2.5-vl-450m/sets.toml) | LiquidAI/LFM2.5-VL-450M @ fc6221ca | 1 | 2 | 238490 | 576772 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-vl-450m-extract](fixtures/lfm2.5-vl-450m-extract/sets.toml) | LiquidAI/LFM2.5-VL-450M-Extract @ 7d5a831d | 1 | 2 | 238490 | 576772 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lighton-rerank-lw-2b](fixtures/lighton-rerank-lw-2b/sets.toml) | lightonai/LightOn-rerank-LW-2B @ 4629d733 | 2 | 2 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lighton-rerank-lw-4b](fixtures/lighton-rerank-lw-4b/sets.toml) | lightonai/LightOn-rerank-LW-4B @ 92b0661f | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lighton-rerank-pw-2b](fixtures/lighton-rerank-pw-2b/sets.toml) | lightonai/LightOn-rerank-PW-2B @ c5333533 | 2 | 2 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lighton-rerank-pw-4b](fixtures/lighton-rerank-pw-4b/sets.toml) | lightonai/LightOn-rerank-PW-4B @ 8f0e0c2d | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lightonocr-0.9b-16k-1025](fixtures/lightonocr-0.9b-16k-1025/sets.toml) | lightonai/LightOnOCR-0.9B-16k-1025 @ e376cca4 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [lightonocr-0.9b-32k-1025](fixtures/lightonocr-0.9b-32k-1025/sets.toml) | lightonai/LightOnOCR-0.9B-32k-1025 @ d69131f6 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [lightonocr-1b-1025](fixtures/lightonocr-1b-1025/sets.toml) | lightonai/LightOnOCR-1B-1025 @ 7e3e7b0c | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lightonocr-2-1b](fixtures/lightonocr-2-1b/sets.toml) | lightonai/LightOnOCR-2-1B @ c97bd377 | 3 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lightonocr-2-1b-base](fixtures/lightonocr-2-1b-base/sets.toml) | lightonai/LightOnOCR-2-1B-base @ c1bc8eb6 | 3 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-1t](fixtures/ling-1t/sets.toml) | inclusionAI/Ling-1T @ 9fd8c406 | 1 | 3 | 531736 | 589657 | 2 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-2.6-1t](fixtures/ling-2.6-1t/sets.toml) | inclusionAI/Ling-2.6-1T @ 653026b0 | 2 | 2 | 531736 | 589657 | 2 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-2.6-1t-base](fixtures/ling-2.6-1t-base/sets.toml) | inclusionAI/Ling-2.6-1T-base @ 64beeb8a | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-2.6-flash](fixtures/ling-2.6-flash/sets.toml) | inclusionAI/Ling-2.6-flash @ 89f8441a | 2 | 2 | 531736 | 589657 | 2 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-flash](fixtures/ling-3.0-flash/sets.toml) | inclusionAI/Ling-3.0-flash @ ef06d91f | 2 | 3 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-flash-fin](fixtures/ling-3.0-flash-fin/sets.toml) | inclusionAI/Ling-3.0-flash-Fin @ a5b4afb4 | 1 | 2 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-flash-vl-fp4](fixtures/ling-3.0-flash-vl-fp4/sets.toml) | inclusionAI/Ling-3.0-flash-VL-fp4 @ 8dca6a0d | 1 | 2 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-tiny](fixtures/ling-3.0-tiny/sets.toml) | inclusionAI/Ling-3.0-tiny @ 9a98e35f | 1 | 2 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-tiny-base](fixtures/ling-3.0-tiny-base/sets.toml) | inclusionAI/Ling-3.0-tiny-base @ bab7297f | 4 | 2 | 238490 | 0 | 278476 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-coder-lite](fixtures/ling-coder-lite/sets.toml) | inclusionAI/Ling-Coder-lite @ e83c3b87 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-flash-2.0](fixtures/ling-flash-2.0/sets.toml) | inclusionAI/Ling-flash-2.0 @ 96f9b7a7 | 1 | 3 | 531736 | 589657 | 2 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-lite](fixtures/ling-lite/sets.toml) | inclusionAI/Ling-lite @ d80333a0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-lite-1.5-2507](fixtures/ling-lite-1.5-2507/sets.toml) | inclusionAI/Ling-lite-1.5-2507 @ 6656efdc | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-lite-base-1.5](fixtures/ling-lite-base-1.5/sets.toml) | inclusionAI/Ling-lite-base-1.5 @ 69500a0d | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-mini-base-2.0](fixtures/ling-mini-base-2.0/sets.toml) | inclusionAI/Ling-mini-base-2.0 @ 1b53ca4c | 6 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-plus](fixtures/ling-plus/sets.toml) | inclusionAI/Ling-plus @ a31ce668 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-plus-base](fixtures/ling-plus-base/sets.toml) | inclusionAI/Ling-plus-base @ 7d722cc1 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llada2.0-mini](fixtures/llada2.0-mini/sets.toml) | inclusionAI/LLaDA2.0-mini @ dad945ca | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llada2.0-mini-preview](fixtures/llada2.0-mini-preview/sets.toml) | inclusionAI/LLaDA2.0-mini-preview @ 98062a17 | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llada2.0-uni](fixtures/llada2.0-uni/sets.toml) | inclusionAI/LLaDA2.0-Uni @ dc2cde43 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llada2.1-flash](fixtures/llada2.1-flash/sets.toml) | inclusionAI/LLaDA2.1-flash @ 2bf95e86 | 4 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3-1-nemotron-ultra-253b-cpt-v1](fixtures/llama-3-1-nemotron-ultra-253b-cpt-v1/sets.toml) | nvidia/Llama-3_1-Nemotron-Ultra-253B-CPT-v1 @ 00e37b5d | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3-1-nemotron-ultra-253b-v1](fixtures/llama-3-1-nemotron-ultra-253b-v1/sets.toml) | nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 @ 5b47def5 | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3-3-nemotron-super-49b-genrm](fixtures/llama-3-3-nemotron-super-49b-genrm/sets.toml) | nvidia/Llama-3_3-Nemotron-Super-49B-GenRM @ 31609793 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3-3-nemotron-super-49b-genrm-multilingual](fixtures/llama-3-3-nemotron-super-49b-genrm-multilingual/sets.toml) | nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual @ 334fda42 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3-3-nemotron-super-49b-v1-5](fixtures/llama-3-3-nemotron-super-49b-v1-5/sets.toml) | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 @ 420ba7d2 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-8b-ultralong-1m-instruct](fixtures/llama-3.1-nemotron-8b-ultralong-1m-instruct/sets.toml) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct @ fa3a10b9 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-8b-ultralong-2m-instruct](fixtures/llama-3.1-nemotron-8b-ultralong-2m-instruct/sets.toml) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct @ 243daed3 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-8b-ultralong-4m-instruct](fixtures/llama-3.1-nemotron-8b-ultralong-4m-instruct/sets.toml) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct @ 02ae0431 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-nano-4b-v1.1](fixtures/llama-3.1-nemotron-nano-4b-v1.1/sets.toml) | nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 @ d552708a | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-nano-8b-v1](fixtures/llama-3.1-nemotron-nano-8b-v1/sets.toml) | nvidia/Llama-3.1-Nemotron-Nano-8B-v1 @ 54641c16 | 1 | 3 | 471553 | 416850 | 176459 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad](fixtures/llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad/sets.toml) | nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD @ aefd101d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-safety-guard-8b-v3](fixtures/llama-3.1-nemotron-safety-guard-8b-v3/sets.toml) | nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 @ 8fdc246b | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-tulu-3-405b-sft](fixtures/llama-3.1-tulu-3-405b-sft/sets.toml) | allenai/Llama-3.1-Tulu-3-405B-SFT @ 7c866216 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-tulu-3.1-8b](fixtures/llama-3.1-tulu-3.1-8b/sets.toml) | allenai/Llama-3.1-Tulu-3.1-8B @ 46239c2d | 3 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.2-11b-vision-instruct](fixtures/llama-3.2-11b-vision-instruct/sets.toml) | meta-llama/Llama-3.2-11B-Vision-Instruct @ 9eb2daaa | 1 | 3 | 508776 | 458203 | 140096 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.2-1b-instruct](fixtures/llama-3.2-1b-instruct/sets.toml) | meta-llama/Llama-3.2-1B-Instruct @ 92131767 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.2-1b-instruct-fp8](fixtures/llama-3.2-1b-instruct-fp8/sets.toml) | RedHatAI/Llama-3.2-1B-Instruct-FP8 @ 6b92f622 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.3-nemotron-70b-reward](fixtures/llama-3.3-nemotron-70b-reward/sets.toml) | nvidia/Llama-3.3-Nemotron-70B-Reward @ 2500cf14 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3.3-nemotron-70b-reward-principle](fixtures/llama-3.3-nemotron-70b-reward-principle/sets.toml) | nvidia/Llama-3.3-Nemotron-70B-Reward-Principle @ ac3e237f | 3 | 2 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-4-maverick-17b-128e-instruct](fixtures/llama-4-maverick-17b-128e-instruct/sets.toml) | meta-llama/Llama-4-Maverick-17B-128E-Instruct @ 73d14711 | 1 | 3 | 531736 | 576438 | 341 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-4-scout-17b-16e](fixtures/llama-4-scout-17b-16e/sets.toml) | meta-llama/Llama-4-Scout-17B-16E @ 14d516bd | 2 | 3 | 531726 | 576758 | 31 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-4-scout-17b-16e-instruct](fixtures/llama-4-scout-17b-16e-instruct/sets.toml) | meta-llama/Llama-4-Scout-17B-16E-Instruct @ 92f3b159 | 1 | 3 | 531736 | 576438 | 341 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-fin-8b](fixtures/llama-fin-8b/sets.toml) | Salesforce/Llama-Fin-8b @ 491d4c7c | 1 | 2 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-guard-3-1b](fixtures/llama-guard-3-1b/sets.toml) | meta-llama/Llama-Guard-3-1B @ acf7aafa | 1 | 3 | 86317 | 0 | 238454 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [llama-guard-4-12b](fixtures/llama-guard-4-12b/sets.toml) | meta-llama/Llama-Guard-4-12B @ 87acb4b9 | 1 | 3 | 86317 | 0 | 238454 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [llama-xlam-2-70b-fc-r](fixtures/llama-xlam-2-70b-fc-r/sets.toml) | Salesforce/Llama-xLAM-2-70b-fc-r @ 783caf59 | 1 | 3 | 531736 | 483062 | 92717 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-xlam-2-8b-fc-r](fixtures/llama-xlam-2-8b-fc-r/sets.toml) | Salesforce/Llama-xLAM-2-8b-fc-r @ a0efe39a | 1 | 3 | 531736 | 483062 | 92717 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llamathought-eurus-matheval](fixtures/llamathought-eurus-matheval/sets.toml) | imdatta0/llamathought_eurus_matheval @ 03e86570 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-1.5-7b-hf](fixtures/llava-1.5-7b-hf/sets.toml) | llava-hf/llava-1.5-7b-hf @ b234b804 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llava-critic-r1-7b-llama32v](fixtures/llava-critic-r1-7b-llama32v/sets.toml) | lmms-lab/LLaVA-Critic-R1-7B-LLaMA32v @ d79e4913 | 1 | 3 | 508776 | 458205 | 140094 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-critic-r1-7b-plus-mimo](fixtures/llava-critic-r1-7b-plus-mimo/sets.toml) | lmms-lab/LLaVA-Critic-R1-7B-Plus-Mimo @ 64466481 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-critic-r1-7b-plus-qwen](fixtures/llava-critic-r1-7b-plus-qwen/sets.toml) | lmms-lab/LLaVA-Critic-R1-7B-Plus-Qwen @ 3d690e52 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-next-72b](fixtures/llava-next-72b/sets.toml) | lmms-lab/llava-next-72b @ 412be716 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-onevision-qwen2-0.5b-ov-hf](fixtures/llava-onevision-qwen2-0.5b-ov-hf/sets.toml) | llava-hf/llava-onevision-qwen2-0.5b-ov-hf @ 74dd0bf8 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-onevision-qwen2-7b-ov](fixtures/llava-onevision-qwen2-7b-ov/sets.toml) | lmms-lab/llava-onevision-qwen2-7b-ov @ 0b07bf75 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-v1.6-mistral-7b-hf](fixtures/llava-v1.6-mistral-7b-hf/sets.toml) | llava-hf/llava-v1.6-mistral-7b-hf @ 2424fdd4 | 1 | 3 | 86316 | 47052 | 530047 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [locateanything-3b](fixtures/locateanything-3b/sets.toml) | nvidia/LocateAnything-3B @ c32291ca | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [longcat-flash-thinking](fixtures/longcat-flash-thinking/sets.toml) | meituan-longcat/LongCat-Flash-Thinking @ 7c9aad1e | 1 | 3 | 238489 | 0 | 278477 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [m3-agent-control](fixtures/m3-agent-control/sets.toml) | ByteDance-Seed/M3-Agent-Control @ b06d8a33 | 1 | 3 | 531736 | 328660 | 111770 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [magenticbrain](fixtures/magenticbrain/sets.toml) | microsoft/MagenticBrain @ db8eb934 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mai-ds-r1](fixtures/mai-ds-r1/sets.toml) | microsoft/MAI-DS-R1 @ a96d011a | 1 | 3 | 531735 | 0 | 1 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [marco-deepresearch-8b](fixtures/marco-deepresearch-8b/sets.toml) | ATH-MaaS/Marco-DeepResearch-8B @ 2868bc1d | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [marco-llm-ar-v2](fixtures/marco-llm-ar-v2/sets.toml) | ATH-MaaS/Marco-LLM-AR-V2 @ 015c30e7 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [marco-llm-es](fixtures/marco-llm-es/sets.toml) | ATH-MaaS/Marco-LLM-ES @ 47621899 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [marco-mini-instruct](fixtures/marco-mini-instruct/sets.toml) | ATH-MaaS/Marco-Mini-Instruct @ a85af582 | 3 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [marco-mt-algharb](fixtures/marco-mt-algharb/sets.toml) | ATH-MaaS/Marco-MT-Algharb @ bdbf1241 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [marco-nano-instruct](fixtures/marco-nano-instruct/sets.toml) | ATH-MaaS/Marco-Nano-Instruct @ 4ff10d07 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mathform-8b](fixtures/mathform-8b/sets.toml) | openbmb/MathForm-8B @ 8f6d09e6 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [medgemma-27b-it](fixtures/medgemma-27b-it/sets.toml) | google/medgemma-27b-it @ 2d3e00ea | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mediphi](fixtures/mediphi/sets.toml) | microsoft/MediPhi @ 0c60932a | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mediphi-instruct](fixtures/mediphi-instruct/sets.toml) | microsoft/MediPhi-Instruct @ 8a524615 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mediphi-pubmed](fixtures/mediphi-pubmed/sets.toml) | microsoft/MediPhi-PubMed @ c8e5a3a6 | 5 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mellum2-12b-a2.5b-instruct](fixtures/mellum2-12b-a2.5b-instruct/sets.toml) | JetBrains/Mellum2-12B-A2.5B-Instruct @ 273d9c1c | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mellum2-12b-a2.5b-thinking](fixtures/mellum2-12b-a2.5b-thinking/sets.toml) | JetBrains/Mellum2-12B-A2.5B-Thinking @ a7311550 | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mellum2.1-12b-a2.5b-thinking](fixtures/mellum2.1-12b-a2.5b-thinking/sets.toml) | JetBrains/Mellum2.1-12B-A2.5B-Thinking @ 92ddae9f | 1 | 2 | 531736 | 589647 | 12 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [meta-llama-3.1-8b-instruct-fp8](fixtures/meta-llama-3.1-8b-instruct-fp8/sets.toml) | RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 @ 870a1177 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [meta-wiki-expert](fixtures/meta-wiki-expert/sets.toml) | facebook/meta-wiki-expert @ 14fafb95 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [midashenglm-0.6b-fp32](fixtures/midashenglm-0.6b-fp32/sets.toml) | mispeech/midashenglm-0.6b-fp32 @ 3969e451 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [midashenglm-7b-0804-bf16](fixtures/midashenglm-7b-0804-bf16/sets.toml) | mispeech/midashenglm-7b-0804-bf16 @ 2d596167 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [midashenglm-7b-1021-bf16](fixtures/midashenglm-7b-1021-bf16/sets.toml) | mispeech/midashenglm-7b-1021-bf16 @ f7b0fdb7 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-7b-base](fixtures/mimo-7b-base/sets.toml) | XiaomiMiMo/MiMo-7B-Base @ c72df458 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mimo-7b-rl-0530](fixtures/mimo-7b-rl-0530/sets.toml) | XiaomiMiMo/MiMo-7B-RL-0530 @ 32340059 | 1 | 3 | 531736 | 576762 | 17 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-7b-sft](fixtures/mimo-7b-sft/sets.toml) | XiaomiMiMo/MiMo-7B-SFT @ 2c8c99d5 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-embodied-7b](fixtures/mimo-embodied-7b/sets.toml) | XiaomiMiMo/MiMo-Embodied-7B @ 6857af64 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-v2.5](fixtures/mimo-v2.5/sets.toml) | XiaomiMiMo/MiMo-V2.5 @ 63651580 | 2 | 2 | 238490 | 589278 | 278857 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-v2.5-asr](fixtures/mimo-v2.5-asr/sets.toml) | XiaomiMiMo/MiMo-V2.5-ASR @ 98641d53 | 1 | 2 | 86321 | 0 | 238450 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [mimo-v2.5-pro-base](fixtures/mimo-v2.5-pro-base/sets.toml) | XiaomiMiMo/MiMo-V2.5-Pro-Base @ 1e94ccfd | 1 | 2 | 238490 | 589278 | 278857 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-v2.6-distill-qwen-9b](fixtures/mimo-v2.6-distill-qwen-9b/sets.toml) | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B @ 2367e865 | 1 | 2 | 531736 | 589298 | 361 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-v2.6-flash-rl](fixtures/mimo-v2.6-flash-rl/sets.toml) | XiaomiMiMo/MiMo-V2.6-Flash-RL @ 5711b268 | 1 | 2 | 531736 | 589298 | 361 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-v2.6-pro-rl](fixtures/mimo-v2.6-pro-rl/sets.toml) | XiaomiMiMo/MiMo-V2.6-Pro-RL @ 73875d00 | 3 | 2 | 531736 | 589298 | 361 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-vl-7b-rl-2508](fixtures/mimo-vl-7b-rl-2508/sets.toml) | XiaomiMiMo/MiMo-VL-7B-RL-2508 @ 4bfb2707 | 4 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mindlink-32b-0801](fixtures/mindlink-32b-0801/sets.toml) | Skywork/MindLink-32B-0801 @ 1c102e40 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mindlink-72b-0801](fixtures/mindlink-72b-0801/sets.toml) | Skywork/MindLink-72B-0801 @ d7d8b63c | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm-v-4.6](fixtures/minicpm-v-4.6/sets.toml) | openbmb/MiniCPM-V-4.6 @ 36f34a66 | 1 | 2 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm-v-4.6-thinking](fixtures/minicpm-v-4.6-thinking/sets.toml) | openbmb/MiniCPM-V-4.6-Thinking @ 93d8f4b6 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-0.5b](fixtures/minicpm4-0.5b/sets.toml) | openbmb/MiniCPM4-0.5B @ 5253c7fc | 4 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-8b](fixtures/minicpm4-8b/sets.toml) | openbmb/MiniCPM4-8B @ bb2ae14c | 4 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-8b-eagle-frspec](fixtures/minicpm4-8b-eagle-frspec/sets.toml) | openbmb/MiniCPM4-8B-Eagle-FRSpec @ 1182e40b | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-8b-eagle-frspec-qat-cpmcu](fixtures/minicpm4-8b-eagle-frspec-qat-cpmcu/sets.toml) | openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu @ 259b5380 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-8b-marlin-cpmcu](fixtures/minicpm4-8b-marlin-cpmcu/sets.toml) | openbmb/MiniCPM4-8B-marlin-cpmcu @ e7e7fcc2 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-8b-marlin-vllm](fixtures/minicpm4-8b-marlin-vllm/sets.toml) | openbmb/MiniCPM4-8B-marlin-vLLM @ b978ad45 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4-mcp](fixtures/minicpm4-mcp/sets.toml) | openbmb/MiniCPM4-MCP @ a86a48b5 | 1 | 3 | 238484 | 576758 | 278503 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4.1-8b-autoawq](fixtures/minicpm4.1-8b-autoawq/sets.toml) | openbmb/MiniCPM4.1-8B-AutoAWQ @ e199301c | 1 | 3 | 531736 | 328660 | 111770 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm4.1-8b-marlin](fixtures/minicpm4.1-8b-marlin/sets.toml) | openbmb/MiniCPM4.1-8B-Marlin @ bc89322c | 1 | 3 | 531736 | 328660 | 111770 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-1b](fixtures/minicpm5-1b/sets.toml) | openbmb/MiniCPM5-1B @ 87179e5c | 1 | 2 | 238490 | 589265 | 278870 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-1b-base](fixtures/minicpm5-1b-base/sets.toml) | openbmb/MiniCPM5-1B-Base @ 15617069 | 1 | 2 | 238490 | 589265 | 278870 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-1b-sft](fixtures/minicpm5-1b-sft/sets.toml) | openbmb/MiniCPM5-1B-SFT @ a60b37f1 | 1 | 2 | 238490 | 589265 | 278870 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-2b](fixtures/minicpm5-2b/sets.toml) | openbmb/MiniCPM5-2B @ f9740005 | 1 | 2 | 238490 | 589277 | 278858 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-2b-base](fixtures/minicpm5-2b-base/sets.toml) | openbmb/MiniCPM5-2B-Base @ 96a57cd5 | 1 | 2 | 238490 | 589277 | 278858 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-2b-sft](fixtures/minicpm5-2b-sft/sets.toml) | openbmb/MiniCPM5-2B-SFT @ 3e0690c5 | 2 | 2 | 238490 | 589277 | 278858 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2](fixtures/minimax-m2/sets.toml) | MiniMaxAI/MiniMax-M2 @ 757303d4 | 1 | 2 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2.1](fixtures/minimax-m2.1/sets.toml) | MiniMaxAI/MiniMax-M2.1 @ cd97f591 | 1 | 2 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2.5](fixtures/minimax-m2.5/sets.toml) | MiniMaxAI/MiniMax-M2.5 @ f710177d | 1 | 2 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2.7](fixtures/minimax-m2.7/sets.toml) | MiniMaxAI/MiniMax-M2.7 @ d494266a | 1 | 3 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m3](fixtures/minimax-m3/sets.toml) | MiniMaxAI/MiniMax-M3 @ f0e1c1e0 | 1 | 1 | 238490 | 589302 | 278833 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-14b-instruct-2512-bf16](fixtures/ministral-3-14b-instruct-2512-bf16/sets.toml) | mistralai/Ministral-3-14B-Instruct-2512-BF16 @ 3cea74c1 | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-14b-reasoning-2512](fixtures/ministral-3-14b-reasoning-2512/sets.toml) | mistralai/Ministral-3-14B-Reasoning-2512 @ 51f9210f | 3 | 2 | 531730 | 583079 | 6586 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-3b-instruct-2512](fixtures/ministral-3-3b-instruct-2512/sets.toml) | mistralai/Ministral-3-3B-Instruct-2512 @ b35d4dfe | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-3b-instruct-2512-bf16](fixtures/ministral-3-3b-instruct-2512-bf16/sets.toml) | mistralai/Ministral-3-3B-Instruct-2512-BF16 @ b6d637be | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-8b-instruct-2512-bf16](fixtures/ministral-3-8b-instruct-2512-bf16/sets.toml) | mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae979 | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-7b-instruct-v0.2](fixtures/mistral-7b-instruct-v0.2/sets.toml) | mistralai/Mistral-7B-Instruct-v0.2 @ 63a8b081 | 2 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mistral-7b-instruct-v0.3](fixtures/mistral-7b-instruct-v0.3/sets.toml) | mistralai/Mistral-7B-Instruct-v0.3 @ c170c708 | 1 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mistral-medium-3.5-128b](fixtures/mistral-medium-3.5-128b/sets.toml) | mistralai/Mistral-Medium-3.5-128B @ 22b2b868 | 1 | 2 | 531729 | 583079 | 6587 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-small-24b-instruct-2501](fixtures/mistral-small-24b-instruct-2501/sets.toml) | mistralai/Mistral-Small-24B-Instruct-2501 @ 9527884b | 1 | 3 | 238488 | 184972 | 533009 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-small-4-119b-2603](fixtures/mistral-small-4-119b-2603/sets.toml) | mistralai/Mistral-Small-4-119B-2603 @ a11f36be | 1 | 2 | 531729 | 583079 | 6587 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mixtral-8x7b-instruct-v0.1](fixtures/mixtral-8x7b-instruct-v0.1/sets.toml) | mistralai/Mixtral-8x7B-Instruct-v0.1 @ eba92302 | 1 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mmsearch-r1-7b-0807](fixtures/mmsearch-r1-7b-0807/sets.toml) | lmms-lab/MMSearch-R1-7B-0807 @ 74587c1c | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [moirai-agent](fixtures/moirai-agent/sets.toml) | Salesforce/moirai-agent @ 010bb0fc | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [molmo2-4b](fixtures/molmo2-4b/sets.toml) | allenai/Molmo2-4B @ 042abfa7 | 2 | 2 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [molmo2-er](fixtures/molmo2-er/sets.toml) | allenai/Molmo2-ER @ dab22564 | 1 | 2 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [molmomotion-4b-h3-f30](fixtures/molmomotion-4b-h3-f30/sets.toml) | allenai/MolmoMotion-4B-H3-F30 @ 3f5e790a | 2 | 2 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [molmoweb-4b](fixtures/molmoweb-4b/sets.toml) | allenai/MolmoWeb-4B @ c96d1cc0 | 2 | 2 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [moss-vl-realtime](fixtures/moss-vl-realtime/sets.toml) | OpenMOSS-Team/MOSS-VL-Realtime @ d1f71a58 | 5 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [moss-vl-realtime-sglang](fixtures/moss-vl-realtime-sglang/sets.toml) | OpenMOSS-Team/MOSS-VL-Realtime-SGLANG @ 3d6407cc | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [muse-glimmer-30b](fixtures/muse-glimmer-30b/sets.toml) | meta-models/Muse-Glimmer-30B @ a4e59da5 | 1 | 2 | 238490 | 494673 | 371462 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [music-flamingo-hf](fixtures/music-flamingo-hf/sets.toml) | nvidia/music-flamingo-hf @ e560c332 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nanbeige4-3b-thinking-2510](fixtures/nanbeige4-3b-thinking-2510/sets.toml) | Nanbeige/Nanbeige4-3B-Thinking-2510 @ 78d97933 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nanbeige4-3b-thinking-2511](fixtures/nanbeige4-3b-thinking-2511/sets.toml) | Nanbeige/Nanbeige4-3B-Thinking-2511 @ 396a1fb7 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nanbeige4.1-3b](fixtures/nanbeige4.1-3b/sets.toml) | Nanbeige/Nanbeige4.1-3B @ 9c4555c3 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nanbeige4.2-3b](fixtures/nanbeige4.2-3b/sets.toml) | Nanbeige/Nanbeige4.2-3B @ b82e54bd | 1 | 2 | 238489 | 589298 | 278838 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nanollama](fixtures/nanollama/sets.toml) | imdatta0/nanollama @ 9a222fd4 | 2 | 2 | 238484 | 0 | 278482 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nanowhale-100m-base](fixtures/nanowhale-100m-base/sets.toml) | HuggingFaceTB/nanowhale-100m-base @ 8ea74cc0 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-3-content-safety](fixtures/nemotron-3-content-safety/sets.toml) | nvidia/Nemotron-3-Content-Safety @ b47f6f6c | 1 | 2 | 238484 | 0 | 278482 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nemotron-3.5-content-safety](fixtures/nemotron-3.5-content-safety/sets.toml) | nvidia/Nemotron-3.5-Content-Safety @ 35645ed3 | 1 | 2 | 238484 | 0 | 278482 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nemotron-cascade-14b-thinking](fixtures/nemotron-cascade-14b-thinking/sets.toml) | nvidia/Nemotron-Cascade-14B-Thinking @ 8aa18a07 | 3 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-cascade-2-30b-a3b](fixtures/nemotron-cascade-2-30b-a3b/sets.toml) | nvidia/Nemotron-Cascade-2-30B-A3B @ 6327cdbc | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-orchestrator-8b](fixtures/nemotron-orchestrator-8b/sets.toml) | nvidia/Nemotron-Orchestrator-8B @ 26df4b9a | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-research-goosereason-4b-instruct](fixtures/nemotron-research-goosereason-4b-instruct/sets.toml) | nvidia/Nemotron-Research-GooseReason-4B-Instruct @ 82197937 | 3 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nextcoder-14b](fixtures/nextcoder-14b/sets.toml) | microsoft/NextCoder-14B @ fed05267 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nextcoder-7b](fixtures/nextcoder-7b/sets.toml) | microsoft/NextCoder-7B @ 962f9fab | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nextstep-1-large-edit](fixtures/nextstep-1-large-edit/sets.toml) | stepfun-ai/NextStep-1-Large-Edit @ bac0d9ff | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nextstep-1-large-pretrain](fixtures/nextstep-1-large-pretrain/sets.toml) | stepfun-ai/NextStep-1-Large-Pretrain @ 199ee4b4 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nextstep-1.1](fixtures/nextstep-1.1/sets.toml) | stepfun-ai/NextStep-1.1 @ 05486ff9 | 3 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nft-32b](fixtures/nft-32b/sets.toml) | nvidia/NFT-32B @ f60778f1 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nft-7b](fixtures/nft-7b/sets.toml) | nvidia/NFT-7B @ a7b4aec0 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [north-micro-vision-instruct](fixtures/north-micro-vision-instruct/sets.toml) | CohereLabs/North-Micro-Vision-Instruct @ fa548d9a | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [north-mini-code-1.0](fixtures/north-mini-code-1.0/sets.toml) | CohereLabs/North-Mini-Code-1.0 @ d11e61a8 | 1 | 2 | 531736 | 11885 | 11 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nv-reason-cxr-3b](fixtures/nv-reason-cxr-3b/sets.toml) | nvidia/NV-Reason-CXR-3B @ 056bd038 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-nano-30b-a3b-bf16](fixtures/nvidia-nemotron-3-nano-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 @ bf77c317 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-nano-4b-bf16](fixtures/nvidia-nemotron-3-nano-4b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-super-120b-a12b-bf16](fixtures/nvidia-nemotron-3-super-120b-a12b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a | 3 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-ultra-550b-a55b-bf16](fixtures/nvidia-nemotron-3-ultra-550b-a55b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 @ 77df655d | 7 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-ultra-550b-a55b-genrm](fixtures/nvidia-nemotron-3-ultra-550b-a55b-genrm/sets.toml) | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM @ 03c50a66 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotron-3.5-lightning-30b-a3b-bf16](fixtures/nvidia-nemotron-3.5-lightning-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ a9904d24 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-labs-3-elastic-30b-a3b-bf16](fixtures/nvidia-nemotron-labs-3-elastic-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 @ 4bcc4326 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16](fixtures/nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 @ 8fe55468 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-nano-9b-v2](fixtures/nvidia-nemotron-nano-9b-v2/sets.toml) | nvidia/NVIDIA-Nemotron-Nano-9B-v2 @ 6533e8de | 1 | 3 | 531726 | 328643 | 111797 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-nano-9b-v2-japanese](fixtures/nvidia-nemotron-nano-9b-v2-japanese/sets.toml) | nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese @ 3979dd16 | 1 | 2 | 531726 | 328643 | 111797 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-parse-2.0](fixtures/nvidia-nemotron-parse-2.0/sets.toml) | nvidia/NVIDIA-Nemotron-Parse-2.0 @ b6742064 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotron-parse-v1.1](fixtures/nvidia-nemotron-parse-v1.1/sets.toml) | nvidia/NVIDIA-Nemotron-Parse-v1.1 @ fd1e3c6b | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotronlabs-ai-for-media-sports-tennis](fixtures/nvidia-nemotronlabs-ai-for-media-sports-tennis/sets.toml) | nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis @ af26673c | 1 | 2 | 238489 | 12825 | 278548 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvila-15b-hf](fixtures/nvila-15b-hf/sets.toml) | Efficient-Large-Model/NVILA-15B-hf @ 7735732b | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvila-8b-hd-video](fixtures/nvila-8b-hd-video/sets.toml) | nvidia/NVILA-8B-HD-Video @ 7a5670e2 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvila-8b-hf](fixtures/nvila-8b-hf/sets.toml) | Efficient-Large-Model/NVILA-8B-hf @ e722ed68 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvila-lite-15b-hf](fixtures/nvila-lite-15b-hf/sets.toml) | Efficient-Large-Model/NVILA-Lite-15B-hf @ d333c18c | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvila-lite-2b-hf](fixtures/nvila-lite-2b-hf/sets.toml) | Efficient-Large-Model/NVILA-Lite-2B-hf @ 2e658896 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-2-0325-32b-dpo](fixtures/olmo-2-0325-32b-dpo/sets.toml) | allenai/OLMo-2-0325-32B-DPO @ 2c8ee0bc | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-2-0325-32b-instruct](fixtures/olmo-2-0325-32b-instruct/sets.toml) | allenai/OLMo-2-0325-32B-Instruct @ b9602434 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-2-0425-1b-instruct](fixtures/olmo-2-0425-1b-instruct/sets.toml) | allenai/OLMo-2-0425-1B-Instruct @ 48d788ec | 5 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-3-32b-think](fixtures/olmo-3-32b-think/sets.toml) | allenai/Olmo-3-32B-Think @ f2edda15 | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3-32b-think-sft](fixtures/olmo-3-32b-think-sft/sets.toml) | allenai/Olmo-3-32B-Think-SFT @ a6d7f3cf | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3-7b-instruct](fixtures/olmo-3-7b-instruct/sets.toml) | allenai/Olmo-3-7B-Instruct @ 6e5971d9 | 1 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-3-7b-instruct-dpo](fixtures/olmo-3-7b-instruct-dpo/sets.toml) | allenai/Olmo-3-7B-Instruct-DPO @ b33130b7 | 3 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-3-7b-instruct-sft](fixtures/olmo-3-7b-instruct-sft/sets.toml) | allenai/Olmo-3-7B-Instruct-SFT @ e1452fc5 | 1 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-3-7b-rl-zero-if](fixtures/olmo-3-7b-rl-zero-if/sets.toml) | allenai/Olmo-3-7B-RL-Zero-IF @ 3cae007c | 3 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3-7b-rl-zero-math](fixtures/olmo-3-7b-rl-zero-math/sets.toml) | allenai/Olmo-3-7B-RL-Zero-Math @ dee26a40 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3-7b-think](fixtures/olmo-3-7b-think/sets.toml) | allenai/Olmo-3-7B-Think @ d97e442d | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3-7b-think-sft](fixtures/olmo-3-7b-think-sft/sets.toml) | allenai/Olmo-3-7B-Think-SFT @ 6ff85758 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3.1-32b-instruct](fixtures/olmo-3.1-32b-instruct/sets.toml) | allenai/Olmo-3.1-32B-Instruct @ ac0587e4 | 1 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-3.1-32b-think](fixtures/olmo-3.1-32b-think/sets.toml) | allenai/Olmo-3.1-32B-Think @ 832c3f54 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3.1-7b-rl-zero-code](fixtures/olmo-3.1-7b-rl-zero-code/sets.toml) | allenai/Olmo-3.1-7B-RL-Zero-Code @ 1ca3bb30 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-3.1-7b-rl-zero-math](fixtures/olmo-3.1-7b-rl-zero-math/sets.toml) | allenai/Olmo-3.1-7B-RL-Zero-Math @ 364820b0 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmo-hybrid-instruct-dpo-7b](fixtures/olmo-hybrid-instruct-dpo-7b/sets.toml) | allenai/Olmo-Hybrid-Instruct-DPO-7B @ ec62da5a | 1 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-hybrid-instruct-sft-7b](fixtures/olmo-hybrid-instruct-sft-7b/sets.toml) | allenai/Olmo-Hybrid-Instruct-SFT-7B @ 15ec262d | 1 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-hybrid-think-sft-7b](fixtures/olmo-hybrid-think-sft-7b/sets.toml) | allenai/Olmo-Hybrid-Think-SFT-7B @ 79d1f7f6 | 1 | 2 | 238490 | 0 | 278476 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [olmocr-2-7b-1025](fixtures/olmocr-2-7b-1025/sets.toml) | allenai/olmOCR-2-7B-1025 @ e52d6f09 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmoe-1b-7b-0125-instruct](fixtures/olmoe-1b-7b-0125-instruct/sets.toml) | allenai/OLMoE-1B-7B-0125-Instruct @ b89a7c4b | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmoe-1b-7b-0125-sft](fixtures/olmoe-1b-7b-0125-sft/sets.toml) | allenai/OLMoE-1B-7B-0125-SFT @ 6f7a5b02 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmoe-1b-7b-0924-instruct](fixtures/olmoe-1b-7b-0924-instruct/sets.toml) | allenai/OLMoE-1B-7B-0924-Instruct @ 7f1c97f4 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [opencodereasoning-nemotron-7b](fixtures/opencodereasoning-nemotron-7b/sets.toml) | nvidia/OpenCodeReasoning-Nemotron-7B @ 3baa2a3d | 7 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openmath-nemotron-14b](fixtures/openmath-nemotron-14b/sets.toml) | nvidia/OpenMath-Nemotron-14B @ b7eeebb0 | 4 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openmath-nemotron-14b-kaggle](fixtures/openmath-nemotron-14b-kaggle/sets.toml) | nvidia/OpenMath-Nemotron-14B-Kaggle @ 735d7fa0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openreasoning-nemotron-1.5b](fixtures/openreasoning-nemotron-1.5b/sets.toml) | nvidia/OpenReasoning-Nemotron-1.5B @ f7e9c457 | 3 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openreasoning-nemotron-14b](fixtures/openreasoning-nemotron-14b/sets.toml) | nvidia/OpenReasoning-Nemotron-14B @ 9e2805b5 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [oreal-32b-sft](fixtures/oreal-32b-sft/sets.toml) | internlm/OREAL-32B-SFT @ 35e32df3 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [oreal-7b](fixtures/oreal-7b/sets.toml) | internlm/OREAL-7B @ 43e8c3b0 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [oreal-7b-sft](fixtures/oreal-7b-sft/sets.toml) | internlm/OREAL-7B-SFT @ 1b5bf328 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [oreal-deepseek-r1-distill-qwen-7b](fixtures/oreal-deepseek-r1-distill-qwen-7b/sets.toml) | internlm/OREAL-DeepSeek-R1-Distill-Qwen-7B @ 09bea68a | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ovis2-4b](fixtures/ovis2-4b/sets.toml) | ATH-MaaS/Ovis2-4B @ 30d3ed68 | 5 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ovis2.5-9b](fixtures/ovis2.5-9b/sets.toml) | ATH-MaaS/Ovis2.5-9B @ d73b2283 | 1 | 3 | 238488 | 184970 | 533011 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ovisocr](fixtures/ovisocr/sets.toml) | ATH-MaaS/OvisOCR @ 5a7a90ff | 1 | 2 | 238488 | 184970 | 533011 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [pacore-8b](fixtures/pacore-8b/sets.toml) | stepfun-ai/PaCoRe-8B @ b49e8427 | 2 | 2 | 531736 | 339548 | 111762 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [paddleocr-vl-1.6](fixtures/paddleocr-vl-1.6/sets.toml) | PaddlePaddle/PaddleOCR-VL-1.6 @ c5630aba | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [paza-phi-4-multimodal-instruct](fixtures/paza-phi-4-multimodal-instruct/sets.toml) | microsoft/paza-Phi-4-multimodal-instruct @ 1e78f4f8 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-3-mini-4k-instruct](fixtures/phi-3-mini-4k-instruct/sets.toml) | microsoft/Phi-3-mini-4k-instruct @ f39ac1d2 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-4-mini-instruct](fixtures/phi-4-mini-instruct/sets.toml) | microsoft/Phi-4-mini-instruct @ cfbefacb | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-4-mini-reasoning](fixtures/phi-4-mini-reasoning/sets.toml) | microsoft/Phi-4-mini-reasoning @ 0e3b1e2d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-4-multimodal-instruct](fixtures/phi-4-multimodal-instruct/sets.toml) | microsoft/Phi-4-multimodal-instruct @ 93f923e1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-4-reasoning](fixtures/phi-4-reasoning/sets.toml) | microsoft/Phi-4-reasoning @ 1de18ec9 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-ground](fixtures/phi-ground/sets.toml) | microsoft/Phi-Ground @ 5eed0cda | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [phi-ground-any](fixtures/phi-ground-any/sets.toml) | microsoft/Phi-Ground-Any @ 08139912 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [phi-tiny-moe-instruct](fixtures/phi-tiny-moe-instruct/sets.toml) | microsoft/Phi-tiny-MoE-instruct @ 2fe50e88 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [pi-mem-35b-a3b](fixtures/pi-mem-35b-a3b/sets.toml) | JetLM/PI-Mem-35B-A3B @ fd87eac3 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [pixtral-12b-fp8-dynamic](fixtures/pixtral-12b-fp8-dynamic/sets.toml) | nm-testing/pixtral-12b-FP8-dynamic @ 6627e5c1 | 1 | 3 | 70812 | 0 | 268922 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, tau2 |
| [points-reader](fixtures/points-reader/sets.toml) | tencent/POINTS-Reader @ 42fbe5b5 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [pp-docbee-2b](fixtures/pp-docbee-2b/sets.toml) | PaddlePaddle/PP-DocBee-2B @ 6e1dccf7 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [pp-docbee-7b](fixtures/pp-docbee-7b/sets.toml) | PaddlePaddle/PP-DocBee-7B @ 486c0c82 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [pp-docbee2-3b](fixtures/pp-docbee2-3b/sets.toml) | PaddlePaddle/PP-DocBee2-3B @ 631d608e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [preferred-medrect-32b](fixtures/preferred-medrect-32b/sets.toml) | pfnet/Preferred-MedRECT-32B @ b992f1db | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qianfan-ocr](fixtures/qianfan-ocr/sets.toml) | baidu/Qianfan-OCR @ 623bf5d2 | 1 | 2 | 531736 | 339546 | 111764 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-agentworld-35b-a3b](fixtures/qwen-agentworld-35b-a3b/sets.toml) | Qwen/Qwen-AgentWorld-35B-A3B @ 60d2b043 | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-drive-1.0-4b](fixtures/qwen-drive-1.0-4b/sets.toml) | Qwen/Qwen-Drive-1.0-4B @ 28484089 | 1 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-image-2.1-pe-i2i](fixtures/qwen-image-2.1-pe-i2i/sets.toml) | Qwen/Qwen-Image-2.1-PE-I2I @ 72927bc0 | 1 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-image-2.1-pe-t2i](fixtures/qwen-image-2.1-pe-t2i/sets.toml) | Qwen/Qwen-Image-2.1-PE-T2I @ f3ed7985 | 1 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-image-bench](fixtures/qwen-image-bench/sets.toml) | Qwen/Qwen-Image-Bench @ 1b77ff83 | 3 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen1.5-moe-a2.7b-chat](fixtures/qwen1.5-moe-a2.7b-chat/sets.toml) | Qwen/Qwen1.5-MoE-A2.7B-Chat @ ec052fda | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2-0-5b-rope8-d-kv-16-refactor](fixtures/qwen2-0-5b-rope8-d-kv-16-refactor/sets.toml) | OpenMOSS-Team/qwen2-0_5B-rope8-d_kv_16-refactor @ 16491216 | 4 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen2-0.5b-instruct](fixtures/qwen2-0.5b-instruct/sets.toml) | Qwen/Qwen2-0.5B-Instruct @ c540970f | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2-5vl-7b-wolfv2-tuned](fixtures/qwen2-5vl-7b-wolfv2-tuned/sets.toml) | Efficient-Large-Model/qwen2_5vl-7b-wolfv2-tuned @ 47c8050e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2-audio-7b-instruct](fixtures/qwen2-audio-7b-instruct/sets.toml) | Qwen/Qwen2-Audio-7B-Instruct @ 0a095220 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2-vl-2b-grpo-8k](fixtures/qwen2-vl-2b-grpo-8k/sets.toml) | lmms-lab/Qwen2-VL-2B-GRPO-8k @ 4668fb4f | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2-vl-2b-instruct](fixtures/qwen2-vl-2b-instruct/sets.toml) | Qwen/Qwen2-VL-2B-Instruct @ 895c3a49 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-0-5b-rope8-d-kv-32-refactor](fixtures/qwen2.5-0-5b-rope8-d-kv-32-refactor/sets.toml) | OpenMOSS-Team/qwen2.5-0_5B-rope8-d_kv_32-refactor @ cbe441c8 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen2.5-0.5b-instruct](fixtures/qwen2.5-0.5b-instruct/sets.toml) | Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760 | 3 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-1.5b-pfn-qfin](fixtures/qwen2.5-1.5b-pfn-qfin/sets.toml) | pfnet/Qwen2.5-1.5B-pfn-qfin @ ee0e12a5 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen2.5-32b-dialoguereason](fixtures/qwen2.5-32b-dialoguereason/sets.toml) | stepfun-ai/Qwen2.5-32B-DialogueReason @ 709df9d0 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-7b-instruct-1m](fixtures/qwen2.5-7b-instruct-1m/sets.toml) | Qwen/Qwen2.5-7B-Instruct-1M @ e28526f7 | 2 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-omni-7b](fixtures/qwen2.5-omni-7b/sets.toml) | Qwen/Qwen2.5-Omni-7B @ ae9e1690 | 2 | 1 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-omni-7b-awq](fixtures/qwen2.5-omni-7b-awq/sets.toml) | Qwen/Qwen2.5-Omni-7B-AWQ @ 848615c2 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-vl-32b-instruct](fixtures/qwen2.5-vl-32b-instruct/sets.toml) | Qwen/Qwen2.5-VL-32B-Instruct @ 7cfb30d7 | 1 | 1 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-vl-7b-instruct](fixtures/qwen2.5-vl-7b-instruct/sets.toml) | Qwen/Qwen2.5-VL-7B-Instruct @ cc594898 | 5 | 1 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-0-6b-uniform-r-16-d-kv-32-refactor](fixtures/qwen3-0-6b-uniform-r-16-d-kv-32-refactor/sets.toml) | OpenMOSS-Team/qwen3-0_6B-uniform_r_16-d_kv_32-refactor @ 32283e48 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3-1.7b-pfn-qfin](fixtures/qwen3-1.7b-pfn-qfin/sets.toml) | pfnet/Qwen3-1.7B-pfn-qfin @ 3e454cbe | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3-30b-a3b](fixtures/qwen3-30b-a3b/sets.toml) | Qwen/Qwen3-30B-A3B @ ad44e777 | 2 | 1 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-30b-a3b-instruct-2507](fixtures/qwen3-30b-a3b-instruct-2507/sets.toml) | Qwen/Qwen3-30B-A3B-Instruct-2507 @ 0d7cf239 | 2 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-30b-a3b-thinking-2507](fixtures/qwen3-30b-a3b-thinking-2507/sets.toml) | Qwen/Qwen3-30B-A3B-Thinking-2507 @ 144afc2f | 3 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-32b-aworld](fixtures/qwen3-32b-aworld/sets.toml) | inclusionAI/Qwen3-32B-AWorld @ ac51f1ef | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-4b-instruct-2507](fixtures/qwen3-4b-instruct-2507/sets.toml) | Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f | 1 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-saferl](fixtures/qwen3-4b-saferl/sets.toml) | Qwen/Qwen3-4B-SafeRL @ 1b95ccb8 | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-sft-executorch-torchao](fixtures/qwen3-4b-sft-executorch-torchao/sets.toml) | imdatta0/qwen3_4b_sft_executorch-torchao @ c7cc0348 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3-4b-thinking-2507](fixtures/qwen3-4b-thinking-2507/sets.toml) | Qwen/Qwen3-4B-Thinking-2507 @ 768f209d | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-8b](fixtures/qwen3-8b/sets.toml) | Qwen/Qwen3-8B @ b968826d | 7 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-8b-abc](fixtures/qwen3-8b-abc/sets.toml) | OpenMOSS-Team/Qwen3-8B-ABC @ fe6d8410 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-8b-fr](fixtures/qwen3-8b-fr/sets.toml) | lightonai/Qwen3-8B-FR @ a53d601e | 16 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-coder-30b-a3b-instruct](fixtures/qwen3-coder-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-Coder-30B-A3B-Instruct @ b2cff646 | 2 | 1 | 238490 | 490451 | 364804 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-coder-next](fixtures/qwen3-coder-next/sets.toml) | Qwen/Qwen3-Coder-Next @ a7fbcb5c | 2 | 1 | 238490 | 490469 | 364786 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-nemotron-235b-a22b-genrm](fixtures/qwen3-nemotron-235b-a22b-genrm/sets.toml) | nvidia/Qwen3-Nemotron-235B-A22B-GenRM @ b127d6ab | 1 | 2 | 531736 | 437023 | 132214 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-nemotron-235b-a22b-genrm-2603](fixtures/qwen3-nemotron-235b-a22b-genrm-2603/sets.toml) | nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603 @ e65a7cb3 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3-nemotron-32b-genrm-principle](fixtures/qwen3-nemotron-32b-genrm-principle/sets.toml) | nvidia/Qwen3-Nemotron-32B-GenRM-Principle @ 9fd58dd6 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-nemotron-32b-rlbff](fixtures/qwen3-nemotron-32b-rlbff/sets.toml) | nvidia/Qwen3-Nemotron-32B-RLBFF @ 9e794640 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-nemotron-8b-brrm](fixtures/qwen3-nemotron-8b-brrm/sets.toml) | nvidia/Qwen3-Nemotron-8B-BRRM @ 00ddfb69 | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-next-80b-a3b-instruct](fixtures/qwen3-next-80b-a3b-instruct/sets.toml) | Qwen/Qwen3-Next-80B-A3B-Instruct @ 9c7f2fbe | 1 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-next-80b-a3b-thinking](fixtures/qwen3-next-80b-a3b-thinking/sets.toml) | Qwen/Qwen3-Next-80B-A3B-Thinking @ e502dd41 | 1 | 1 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-next-moe](fixtures/qwen3-next-moe/sets.toml) | tiny-random/qwen3-next-moe @ fab38e08 | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-omni-30b-a3b-instruct](fixtures/qwen3-omni-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-Omni-30B-A3B-Instruct @ 26291f79 | 1 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-omni-30b-a3b-thinking](fixtures/qwen3-omni-30b-a3b-thinking/sets.toml) | Qwen/Qwen3-Omni-30B-A3B-Thinking @ 2f443cfc | 1 | 1 | 36112 | 54418 | 92 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-vl-235b-a22b-thinking](fixtures/qwen3-vl-235b-a22b-thinking/sets.toml) | Qwen/Qwen3-VL-235B-A22B-Thinking @ 6664affd | 2 | 1 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-vl-30b-a3b-instruct](fixtures/qwen3-vl-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-VL-30B-A3B-Instruct @ 9c4b90e1 | 2 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-vl-8b-instruct](fixtures/qwen3-vl-8b-instruct/sets.toml) | Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0 | 10 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-vl-8b-thinking](fixtures/qwen3-vl-8b-thinking/sets.toml) | Qwen/Qwen3-VL-8B-Thinking @ 92f3c4b4 | 4 | 1 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.5-0.8b-base](fixtures/qwen3.5-0.8b-base/sets.toml) | Qwen/Qwen3.5-0.8B-Base @ dc7cdfe2 | 2 | 1 | 238478 | 0 | 263525 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3.5-27b](fixtures/qwen3.5-27b/sets.toml) | Qwen/Qwen3.5-27B @ fc05daec | 1 | 1 | 27159 | 54179 | 10593 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.5-2b](fixtures/qwen3.5-2b/sets.toml) | Qwen/Qwen3.5-2B @ 15852e8c | 3 | 1 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.5-35b-a3b](fixtures/qwen3.5-35b-a3b/sets.toml) | Qwen/Qwen3.5-35B-A3B @ 59d61f3c | 5 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.5-35b-a3b-base](fixtures/qwen3.5-35b-a3b-base/sets.toml) | Qwen/Qwen3.5-35B-A3B-Base @ 0f081307 | 1 | 1 | 238478 | 0 | 263525 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3.5-9b](fixtures/qwen3.5-9b/sets.toml) | Qwen/Qwen3.5-9B @ c2022362 | 3 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.6-27b](fixtures/qwen3.6-27b/sets.toml) | Qwen/Qwen3.6-27B @ 6a9e13bd | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.6-35b-a3b](fixtures/qwen3.6-35b-a3b/sets.toml) | Qwen/Qwen3.6-35B-A3B @ 995ad96e | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.8-2.4t-a95b](fixtures/qwen3.8-2.4t-a95b/sets.toml) | Qwen/Qwen3.8-2.4T-A95B @ 207bd685 | 1 | 1 | 238477 | 474553 | 349822 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.8-27b](fixtures/qwen3.8-27b/sets.toml) | Qwen/Qwen3.8-27B @ 1d4bf0f2 | 2 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.8-flash-next](fixtures/qwen3.8-flash-next/sets.toml) | Qwen/Qwen3.8-Flash-Next @ de4b8e4d | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3guard-gen-0.6b](fixtures/qwen3guard-gen-0.6b/sets.toml) | Qwen/Qwen3Guard-Gen-0.6B @ fada3b2f | 3 | 1 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwq-32b](fixtures/qwq-32b/sets.toml) | Qwen/QwQ-32B @ 976055f8 | 1 | 1 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [r1-aqa](fixtures/r1-aqa/sets.toml) | mispeech/r1-aqa @ e1068f8e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [recon2reason-reasoning-4b](fixtures/recon2reason-reasoning-4b/sets.toml) | BAAI/Recon2Reason-Reasoning-4B @ f8faf41a | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ring-flash-2.0](fixtures/ring-flash-2.0/sets.toml) | inclusionAI/Ring-flash-2.0 @ a73b44bc | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ring-lite](fixtures/ring-lite/sets.toml) | inclusionAI/Ring-lite @ 237e860a | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ring-lite-2506](fixtures/ring-lite-2506/sets.toml) | inclusionAI/Ring-lite-2506 @ f25035fc | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ring-lite-2507](fixtures/ring-lite-2507/sets.toml) | inclusionAI/Ring-lite-2507 @ 8cf0ec24 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ring-lite-distill-preview](fixtures/ring-lite-distill-preview/sets.toml) | inclusionAI/Ring-lite-distill-preview @ 3c06a3bc | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ring-mini-2.0](fixtures/ring-mini-2.0/sets.toml) | inclusionAI/Ring-mini-2.0 @ e807a1d1 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ring-mini-sparse-2.0-exp](fixtures/ring-mini-sparse-2.0-exp/sets.toml) | inclusionAI/Ring-mini-sparse-2.0-exp @ a69896fe | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [riva-translate-4b-instruct](fixtures/riva-translate-4b-instruct/sets.toml) | nvidia/Riva-Translate-4B-Instruct @ d3250d05 | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [rlpr-gemma2-2b-it](fixtures/rlpr-gemma2-2b-it/sets.toml) | openbmb/RLPR-Gemma2-2B-it @ 0f9270b8 | 1 | 3 | 86317 | 47052 | 530046 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [rlpr-llama3.1-8b-inst](fixtures/rlpr-llama3.1-8b-inst/sets.toml) | openbmb/RLPR-Llama3.1-8B-Inst @ 1463692d | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [rlpr-qwen2.5-7b-base](fixtures/rlpr-qwen2.5-7b-base/sets.toml) | openbmb/RLPR-Qwen2.5-7B-Base @ 0327588f | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [robobrain-dex](fixtures/robobrain-dex/sets.toml) | BAAI/RoboBrain-Dex @ be0b0b2d | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [rubicon-preview](fixtures/rubicon-preview/sets.toml) | inclusionAI/Rubicon-Preview @ ebcbc03b | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sage-mm-molmo2-8b-sft](fixtures/sage-mm-molmo2-8b-sft/sets.toml) | allenai/SAGE-MM-Molmo2-8B-SFT @ 53c7e968 | 1 | 2 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [sage-mm-molmo2-8b-sft-rl](fixtures/sage-mm-molmo2-8b-sft-rl/sets.toml) | allenai/SAGE-MM-Molmo2-8B-SFT_RL @ 55cc7057 | 1 | 2 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [sage-mm-qwen2.5-vl-7b-sft](fixtures/sage-mm-qwen2.5-vl-7b-sft/sets.toml) | allenai/SAGE-MM-Qwen2.5-VL-7B-SFT @ 3c141dd4 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sage-mm-qwen2.5-vl-7b-sft-rl](fixtures/sage-mm-qwen2.5-vl-7b-sft-rl/sets.toml) | allenai/SAGE-MM-Qwen2.5-VL-7B-SFT_RL @ 63930ee3 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sage-mm-qwen3-vl-4b-sft-rl](fixtures/sage-mm-qwen3-vl-4b-sft-rl/sets.toml) | allenai/SAGE-MM-Qwen3-VL-4B-SFT_RL @ ea4dced0 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sage-mm-qwen3-vl-8b-sft](fixtures/sage-mm-qwen3-vl-8b-sft/sets.toml) | allenai/SAGE-MM-Qwen3-VL-8B-SFT @ 983d97eb | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sarvam-m](fixtures/sarvam-m/sets.toml) | sarvamai/sarvam-m @ 01534a53 | 1 | 3 | 238483 | 10886 | 278493 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sarvam-translate](fixtures/sarvam-translate/sets.toml) | sarvamai/sarvam-translate @ e0a7a6bc | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [scalecua-32b](fixtures/scalecua-32b/sets.toml) | OpenGVLab/ScaleCUA-32B @ 9a91c806 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [scalecua-3b](fixtures/scalecua-3b/sets.toml) | OpenGVLab/ScaleCUA-3B @ bb5c5012 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [scalecua-7b](fixtures/scalecua-7b/sets.toml) | OpenGVLab/ScaleCUA-7B @ da549689 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [scicore-mol](fixtures/scicore-mol/sets.toml) | openbmb/SciCore-Mol @ 82831485 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [scithinker-30b](fixtures/scithinker-30b/sets.toml) | OpenMOSS-Team/SciThinker-30B @ cbbfdf4a | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [scithinker-4b](fixtures/scithinker-4b/sets.toml) | OpenMOSS-Team/SciThinker-4B @ ddd5752d | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sdar-30b-a3b-chat](fixtures/sdar-30b-a3b-chat/sets.toml) | JetLM/SDAR-30B-A3B-Chat @ f5add2a1 | 5 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [seed-coder-8b-instruct](fixtures/seed-coder-8b-instruct/sets.toml) | ByteDance-Seed/Seed-Coder-8B-Instruct @ 1c5af72a | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [seed-coder-8b-reasoning](fixtures/seed-coder-8b-reasoning/sets.toml) | ByteDance-Seed/Seed-Coder-8B-Reasoning @ 9927b772 | 1 | 3 | 86317 | 47052 | 530046 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [seed-coder-8b-reasoning-bf16](fixtures/seed-coder-8b-reasoning-bf16/sets.toml) | ByteDance-Seed/Seed-Coder-8B-Reasoning-bf16 @ aa146343 | 1 | 3 | 86317 | 47052 | 530046 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [seed-oss-36b-base](fixtures/seed-oss-36b-base/sets.toml) | ByteDance-Seed/Seed-OSS-36B-Base @ 866afe24 | 1 | 3 | 236463 | 495977 | 373185 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [seed-oss-36b-instruct](fixtures/seed-oss-36b-instruct/sets.toml) | ByteDance-Seed/Seed-OSS-36B-Instruct @ 497f1dca | 2 | 2 | 236463 | 495977 | 373185 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sepsisagent-4b](fixtures/sepsisagent-4b/sets.toml) | FreedomIntelligence/SepsisAgent-4B @ 39193beb | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [sera-32b](fixtures/sera-32b/sets.toml) | allenai/SERA-32B @ b4f53836 | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [setox-qwen2.5-3b](fixtures/setox-qwen2.5-3b/sets.toml) | thu-coai/SeTox-Qwen2.5-3B @ 268ae290 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [shieldagent](fixtures/shieldagent/sets.toml) | thu-coai/ShieldAgent @ a62f8413 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [shieldstral-1.0-3b](fixtures/shieldstral-1.0-3b/sets.toml) | mistralai/Shieldstral-1.0-3B @ 003ec7e2 | 1 | 2 | 238483 | 184968 | 533018 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [shieldvlm-7b-qwen](fixtures/shieldvlm-7b-qwen/sets.toml) | thu-coai/ShieldVLM-7B-qwen @ 5166dfe7 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [singguard-4b](fixtures/singguard-4b/sets.toml) | inclusionAI/SingGuard-4b @ 4e082d06 | 1 | 2 | 531736 | 439594 | 125643 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [singguard-nsfa-4b](fixtures/singguard-nsfa-4b/sets.toml) | inclusionAI/SingGuard-NSFA-4B @ c7991967 | 2 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [skywork-or1-32b-preview](fixtures/skywork-or1-32b-preview/sets.toml) | Skywork/Skywork-OR1-32B-Preview @ ae710ba9 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [skywork-or1-7b](fixtures/skywork-or1-7b/sets.toml) | Skywork/Skywork-OR1-7B @ 5bd323a9 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [skywork-or1-math-7b](fixtures/skywork-or1-math-7b/sets.toml) | Skywork/Skywork-OR1-Math-7B @ fcef970f | 2 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [skywork-r1v2-38b](fixtures/skywork-r1v2-38b/sets.toml) | Skywork/Skywork-R1V2-38B @ 2230f5dc | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [skywork-r1v3-38b](fixtures/skywork-r1v3-38b/sets.toml) | Skywork/Skywork-R1V3-38B @ 85536660 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [skywork-swe-32b](fixtures/skywork-swe-32b/sets.toml) | Skywork/Skywork-SWE-32B @ f39ff490 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [small-qwen3-5-20b](fixtures/small-qwen3-5-20b/sets.toml) | imdatta0/small_qwen3_5_20b @ 801d554c | 1 | 2 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [smollm2-1.7b-instruct-16k](fixtures/smollm2-1.7b-instruct-16k/sets.toml) | HuggingFaceTB/SmolLM2-1.7B-Instruct-16k @ e6176739 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smollm2-1.7b-magpie-ultra-v0.1-attribution](fixtures/smollm2-1.7b-magpie-ultra-v0.1-attribution/sets.toml) | EleutherAI/SmolLM2-1.7B-magpie-ultra-v0.1-attribution @ 9dabc286 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smollm2-1.7b-magpie-ultra-v0.1-attribution-lowest](fixtures/smollm2-1.7b-magpie-ultra-v0.1-attribution-lowest/sets.toml) | EleutherAI/SmolLM2-1.7B-magpie-ultra-v0.1-attribution-lowest @ 69db70bb | 4 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smollm2-1.7b-magpie-ultra-v0.1-classification](fixtures/smollm2-1.7b-magpie-ultra-v0.1-classification/sets.toml) | EleutherAI/SmolLM2-1.7B-magpie-ultra-v0.1-classification @ c53eee0e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smollm2-1.7b-magpie-ultra-v0.1-random](fixtures/smollm2-1.7b-magpie-ultra-v0.1-random/sets.toml) | EleutherAI/SmolLM2-1.7B-magpie-ultra-v0.1-random @ e0e5c140 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smollm2-1.7b-magpie-ultra-v1.0-attribution-lowest](fixtures/smollm2-1.7b-magpie-ultra-v1.0-attribution-lowest/sets.toml) | EleutherAI/SmolLM2-1.7B-magpie-ultra-v1.0-attribution-lowest @ feeabc0e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smollm3-3b](fixtures/smollm3-3b/sets.toml) | HuggingFaceTB/SmolLM3-3B @ a07cc9a0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smolvlm-256m-base](fixtures/smolvlm-256m-base/sets.toml) | HuggingFaceTB/SmolVLM-256M-Base @ 40ead252 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smolvlm-256m-instruct](fixtures/smolvlm-256m-instruct/sets.toml) | HuggingFaceTB/SmolVLM-256M-Instruct @ 7e3e67ed | 1 | 3 | 498897 | 328664 | 144605 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smolvlm-500m-base](fixtures/smolvlm-500m-base/sets.toml) | HuggingFaceTB/SmolVLM-500M-Base @ f76e4be0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smolvlm2-2.2b-base](fixtures/smolvlm2-2.2b-base/sets.toml) | HuggingFaceTB/SmolVLM2-2.2B-Base @ 312a1101 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [smolvlm2-2.2b-instruct](fixtures/smolvlm2-2.2b-instruct/sets.toml) | HuggingFaceTB/SmolVLM2-2.2B-Instruct @ 482adb53 | 1 | 3 | 498897 | 328664 | 144605 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [solar-10.7b-instruct-v1.0](fixtures/solar-10.7b-instruct-v1.0/sets.toml) | upstage/SOLAR-10.7B-Instruct-v1.0 @ 090d405e | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [songprep-7b](fixtures/songprep-7b/sets.toml) | tencent/SongPrep-7B @ 6edc7950 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [spatial-ssrl-qwen3vl-4b](fixtures/spatial-ssrl-qwen3vl-4b/sets.toml) | internlm/Spatial-SSRL-Qwen3VL-4B @ 193be56f | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [stage2-internvideo2-1b-qwen2-5-7b-tome16-mlp](fixtures/stage2-internvideo2-1b-qwen2-5-7b-tome16-mlp/sets.toml) | OpenGVLab/stage2-InternVideo2-1B-Qwen2_5-7B-tome16_mlp @ 5a46f5c7 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [stage2-umt-qwen2-5-7b-1m-tome16-mlp](fixtures/stage2-umt-qwen2-5-7b-1m-tome16-mlp/sets.toml) | OpenGVLab/stage2-UMT-Qwen2_5_7B_1m-tome16_mlp @ 6c2a1ad8 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [stage2-umt-qwen2-7b-tome16-mlp](fixtures/stage2-umt-qwen2-7b-tome16-mlp/sets.toml) | OpenGVLab/stage2-UMT-Qwen2-7B-tome16_mlp @ 0f08243b | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [step-3.5-flash](fixtures/step-3.5-flash/sets.toml) | stepfun-ai/Step-3.5-Flash @ ab446a3d | 1 | 3 | 238490 | 54165 | 278713 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [step-3.5-flash-base-midtrain](fixtures/step-3.5-flash-base-midtrain/sets.toml) | stepfun-ai/Step-3.5-Flash-Base-Midtrain @ c0ca7213 | 1 | 2 | 238490 | 0 | 278476 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [step3](fixtures/step3/sets.toml) | stepfun-ai/step3 @ e2d67fc9 | 1 | 2 | 238490 | 0 | 278476 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [stepfun-formalizer-7b](fixtures/stepfun-formalizer-7b/sets.toml) | stepfun-ai/StepFun-Formalizer-7B @ fb0dc612 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [stepfun-prover-preview-32b](fixtures/stepfun-prover-preview-32b/sets.toml) | stepfun-ai/StepFun-Prover-Preview-32B @ eee45bc5 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [stepfun-prover-preview-7b](fixtures/stepfun-prover-preview-7b/sets.toml) | stepfun-ai/StepFun-Prover-Preview-7B @ d0af60a6 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [swe-dev-32b](fixtures/swe-dev-32b/sets.toml) | zai-org/SWE-Dev-32B @ 6f8669b8 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [swe-dev-7b](fixtures/swe-dev-7b/sets.toml) | zai-org/SWE-Dev-7B @ d04b618d | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [swe-fixer-retriever-7b](fixtures/swe-fixer-retriever-7b/sets.toml) | internlm/SWE-Fixer-Retriever-7B @ f9be4b9f | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [swerankllm-large](fixtures/swerankllm-large/sets.toml) | Salesforce/SweRankLLM-Large @ 282232e6 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [swerankllm-small](fixtures/swerankllm-small/sets.toml) | Salesforce/SweRankLLM-Small @ 5f124b0d | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [synlogic-7b](fixtures/synlogic-7b/sets.toml) | MiniMaxAI/SynLogic-7B @ bfcdfe25 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [tar-1.5b](fixtures/tar-1.5b/sets.toml) | ByteDance-Seed/Tar-1.5B @ d0dfaed9 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tar-7b](fixtures/tar-7b/sets.toml) | ByteDance-Seed/Tar-7B @ b137de64 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [taskmem](fixtures/taskmem/sets.toml) | ByteDance-Seed/TaskMem @ b2b4dff1 | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tcandon-router](fixtures/tcandon-router/sets.toml) | tencent/TCAndon-Router @ 14c2f8fd | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [test-smollm2-135m-instruct](fixtures/test-smollm2-135m-instruct/sets.toml) | EleutherAI/test-SmolLM2-135M-Instruct @ cf998708 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [thyme-rl](fixtures/thyme-rl/sets.toml) | Kwai-Keye/Thyme-RL @ 88bd0bed | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tiny-deepseek-v3-2.5b-0.4b](fixtures/tiny-deepseek-v3-2.5b-0.4b/sets.toml) | imdatta0/tiny_deepseek_v3_2.5B_0.4B @ 38d63864 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tiny-glm4-moe-2.8b-0.7b](fixtures/tiny-glm4-moe-2.8b-0.7b/sets.toml) | imdatta0/tiny_glm4_moe_2.8B_0.7B @ b7c68d97 | 1 | 2 | 238490 | 0 | 278476 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [tiny-qwen3-moe-2.8b-0.7b](fixtures/tiny-qwen3-moe-2.8b-0.7b/sets.toml) | imdatta0/tiny_qwen3_moe_2.8B_0.7B @ d266d1a0 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tiny-qwen35-1.8b-a0.8b](fixtures/tiny-qwen35-1.8b-a0.8b/sets.toml) | imdatta0/tiny_qwen35_1.8B_A0.8B @ bbd8f183 | 1 | 2 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tiny-random-llamaforcausallm](fixtures/tiny-random-llamaforcausallm/sets.toml) | hmellor/tiny-random-LlamaForCausalLM @ 9408c553 | 1 | 3 | 238484 | 0 | 278482 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [tinyllama-1.1b-chat-v1.0](fixtures/tinyllama-1.1b-chat-v1.0/sets.toml) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tmax-8b](fixtures/tmax-8b/sets.toml) | allenai/tmax-8b @ 972af209 | 1 | 2 | 531736 | 589647 | 12 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [tmax-sft-8b](fixtures/tmax-sft-8b/sets.toml) | allenai/tmax-sft-8b @ ec16c2c4 | 1 | 2 | 531736 | 589647 | 12 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [toolmind-web-3b](fixtures/toolmind-web-3b/sets.toml) | Nanbeige/ToolMind-Web-3B @ 9f5aa81c | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [trinity-mini](fixtures/trinity-mini/sets.toml) | arcee-ai/Trinity-Mini @ cdb81e81 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [turntaking-pretraining-it-multilingual-3c](fixtures/turntaking-pretraining-it-multilingual-3c/sets.toml) | fixie-ai/turntaking-pretraining-it-multilingual-3c @ ba1bd74d | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-tars-1.5-7b](fixtures/ui-tars-1.5-7b/sets.toml) | ByteDance-Seed/UI-TARS-1.5-7B @ 683d002d | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-tars-2b-sft](fixtures/ui-tars-2b-sft/sets.toml) | ByteDance-Seed/UI-TARS-2B-SFT @ f366a1db | 3 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-tars-72b-dpo](fixtures/ui-tars-72b-dpo/sets.toml) | ByteDance-Seed/UI-TARS-72B-DPO @ 43a70a29 | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-tars-7b-dpo](fixtures/ui-tars-7b-dpo/sets.toml) | ByteDance-Seed/UI-TARS-7B-DPO @ 727b0df3 | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-venus-ground-72b](fixtures/ui-venus-ground-72b/sets.toml) | inclusionAI/UI-Venus-Ground-72B @ e9d2aa95 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-venus-ground-7b](fixtures/ui-venus-ground-7b/sets.toml) | inclusionAI/UI-Venus-Ground-7B @ cfc2528c | 6 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ui-venus-navi-72b](fixtures/ui-venus-navi-72b/sets.toml) | inclusionAI/UI-Venus-Navi-72B @ 32e9fb08 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ultravox-v0-5-llama-3-1-8b](fixtures/ultravox-v0-5-llama-3-1-8b/sets.toml) | fixie-ai/ultravox-v0_5-llama-3_1-8b @ 94aa77f7 | 2 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ultravox-v0-6-gemma-3-27b](fixtures/ultravox-v0-6-gemma-3-27b/sets.toml) | fixie-ai/ultravox-v0_6-gemma-3-27b @ 17a79bc5 | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ultravox-v0-7-glm-4-6](fixtures/ultravox-v0-7-glm-4-6/sets.toml) | fixie-ai/ultravox-v0_7-glm-4_6 @ a849a095 | 2 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [unireason-med](fixtures/unireason-med/sets.toml) | IQuestLab/UniReason-Med @ 68d7c25b | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [userlm-8b](fixtures/userlm-8b/sets.toml) | microsoft/UserLM-8b @ ce576a23 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [video-xl-2](fixtures/video-xl-2/sets.toml) | BAAI/Video-XL-2 @ 107a6eba | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [video-xl-2-stage3](fixtures/video-xl-2-stage3/sets.toml) | BAAI/Video-XL-2-Stage3 @ 704a77a1 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [videochat-r1-7b-caption](fixtures/videochat-r1-7b-caption/sets.toml) | OpenGVLab/VideoChat-R1_7B_caption @ 0fa455a3 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [vilasr](fixtures/vilasr/sets.toml) | inclusionAI/ViLaSR @ 89f84f29 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [virtuoso-large](fixtures/virtuoso-large/sets.toml) | arcee-ai/Virtuoso-Large @ fd1911bb | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [virtuoso-lite](fixtures/virtuoso-lite/sets.toml) | arcee-ai/Virtuoso-Lite @ d0d005aa | 1 | 3 | 531736 | 328661 | 111769 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [virtuoso-medium-v2](fixtures/virtuoso-medium-v2/sets.toml) | arcee-ai/Virtuoso-Medium-v2 @ bb67df2f | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [vista-9b](fixtures/vista-9b/sets.toml) | inclusionAI/VISTA-9B @ a4e30d1a | 2 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [webworld-32b](fixtures/webworld-32b/sets.toml) | Qwen/WebWorld-32B @ e7fceedc | 3 | 1 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [xgen-small-4b-base-r](fixtures/xgen-small-4b-base-r/sets.toml) | Salesforce/xgen-small-4B-base-r @ d9f205a4 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [xgen-small-4b-instruct-r](fixtures/xgen-small-4b-instruct-r/sets.toml) | Salesforce/xgen-small-4B-instruct-r @ b2b2f609 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [zamba2-1.2b-instruct-v2](fixtures/zamba2-1.2b-instruct-v2/sets.toml) | Zyphra/Zamba2-1.2B-Instruct-v2 @ 960222dd | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [zamba2-7b-instruct-v2](fixtures/zamba2-7b-instruct-v2/sets.toml) | Zyphra/Zamba2-7B-Instruct-v2 @ 54438e69 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [zerogui-androidlab-7b](fixtures/zerogui-androidlab-7b/sets.toml) | OpenGVLab/ZeroGUI-AndroidLab-7B @ c9b1907d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [zerogui-osworld-7b](fixtures/zerogui-osworld-7b/sets.toml) | OpenGVLab/ZeroGUI-OSWorld-7B @ d14c534e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| all | | 1191 | | 336850906 | 246927631 | 101862054 | |

177 more groups (279 checkpoints) have a manifest and nothing recorded yet.

### Corpus

The cases every group is recorded over, by the source they were imported from.

| Source | From | Licenses | Render sets | Render cases | Parse sets | Parse cases | Plain | Stored |
|---|---|---|---:|---:|---:|---:|---:|---|
| bfcl | pypi:bfcl-eval==2026.3.23 | Apache-2.0 | 17 | 4119 | 14 | 2735 | 37.0 MB | 37.0 MB, plain |
| glaive-v2 | hf:datasets/glaiveai/glaive-function-calling-v2@e7f4b645 | Apache-2.0 | 71 | 289494 | 71 | 306240 | 1.24 GB | 81.4 MB, zstd in Git LFS |
| gsm8k | github:openai/grade-school-math@3101c7d5 | MIT | 2 | 8792 | 4 | 17584 | 22.6 MB | 22.6 MB, plain |
| hand-written | written in this repository | | 1 | 24 | 1 | 16 | 0.0 MB | 0.0 MB, plain |
| hermes | hf:datasets/NousResearch/hermes-function-calling-v1@dae3e1d2 | Apache-2.0 | 3 | 20769 | 3 | 20773 | 99.6 MB | 5.8 MB, zstd in Git LFS |
| mgsm | github:google-research/url-nlp@3622039c | CC-BY-4.0 | 10 | 2500 | 12 | 2838 | 4.2 MB | 4.2 MB, plain |
| shapes | github:openai/grade-school-math@3101c7d5 and pypi:bfcl-eval==2026.3.23 | Apache-2.0, MIT | 0 | 0 | 6 | 6000 | 17.9 MB | 17.9 MB, plain |
| swebench | hf:datasets/SWE-bench/SWE-bench@c6fe717f, hf:datasets/SWE-bench/SWE-bench_Verified@78f471bf | 8 licenses | 4 | 2236 | 8 | 4472 | 45.8 MB | 45.8 MB, plain |
| swehero | hf:datasets/nvidia/SWE-Hero-openhands-trajectories@150bc119 | CC-BY-4.0 | 14 | 79627 | 14 | 101568 | 14.84 GB | 882.8 MB, zstd in Git LFS |
| tau2 | github:sierra-research/tau2-bench@4ce7c039 | MIT | 22 | 124175 | 22 | 127433 | 9.77 GB | 30.8 MB, zstd in Git LFS |
| all | | | 144 | 531736 | 155 | 589659 | 26.08 GB | 1.13 GB |

### Images

The images the references of every processor family are recorded over, by the dataset they were imported from; the files are in Git LFS.

| Dataset | From | Licenses | Images | Bytes |
|---|---|---|---:|---:|
| docvqa | hf:datasets/lmms-lab-encoder/DocVQA@539088ef | Apache-2.0 | 1285 | 832.7 MB |
| mmmu | hf:datasets/MMMU/MMMU@876ce5cb | Apache-2.0 | 959 | 340.6 MB |
| all | | | 2244 | 1.17 GB |

<!-- end of what bellwether count --readme writes -->

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

## Checkpoint groups and manifests

```bash
HF_HUB_OFFLINE=1 uv run bellwether manifests
```

Reads the list of checkpoints committed beside the manifests, `fixtures/models.tsv` (`--models` names another file):
one `model<TAB>revision<TAB>downloads<TAB>day<TAB>tier` row per checkpoint, with the Hub's downloads over the last 30
days and the day they were read. It computes each one's oracle inputs (the tokenizer files, the chat templates, and the
few fields of `config.json`, `generation_config.json` and `chat_template.json` that the oracles read) at the pinned
revision from the Hugging Face cache, groups the checkpoints whose inputs are equal, and writes every checkpoint's
`fixtures/<slug>/manifest.toml`. A group is recorded once, under its primary's slug: the recorded one if there is one,
else its most-downloaded member's. The others' manifests name the group and hold no fixtures. It prints each group with
its members and their tiers. Run again over the same list, it writes the same files, so a change to the list is
reviewed as a diff of the list and the manifests. `fixtures/README.md` describes both. `models.jsonl` (below) is
another list: what the engines' registries name, with no revision or downloads until the Hub is read.

## Recording render fixtures

```bash
uv run bellwether record --model Qwen/Qwen3-8B --kind render --oracle reference
```

`record` checks the manifest before it reads the corpus: it refuses any member of a group but its primary, naming the
primary to record instead, and a checkpoint whose oracle inputs at the pinned revision cannot be read or are not the
ones its manifest lists, naming the files that differ.

Finds the manifest whose `model` is the given id (`fixtures/qwen3-8b/manifest.toml`), runs every
case under `corpus/render/` through the checkpoint's own chat template at the pinned revision
(`transformers.apply_chat_template`, each message's content in the format vLLM selects for the
template: a one-item text part list for a template that loops over the content, else the string),
and writes `fixtures/qwen3-8b/render/<set>.jsonl`, or
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

The output's token ids must decode back to its text. When the tokenizer's own ids do not, because its
normalizer applies a Unicode normalization form (Qwen3-8B's NFC writes U+09DF, one code point in the
text, as two), the ids are built from the text as written, by a copy of the tokenizer whose normalizer
leaves out NFC, NFD, NFKC and NFKD and keeps every other step. The run prints how many cases of each
set took that path (#57); `sets.toml` does not record it. A case whose ids do not give back its text
either way is reported and not recorded.

## Recording with the vendor's code

```bash
uv run bellwether sandbox-record --model moonshotai/Kimi-K3 --kind render
uv run bellwether sandbox-record --model moonshotai/Kimi-K3 --kind parse
```

Some checkpoints ship their tokenizer as their own Python class, like Kimi-K3's
`tokenization_kimi.TikTokenTokenizer`. Such a checkpoint is recorded by the vendor-code oracle, which
runs that class only inside a container:
- with no network and a read-only root;
- as a user without privileges;
- holding only the files the manifest lists, each checked against its sha256 before anything runs.

It needs a Docker engine. `record --oracle vendor` refuses outside the container and names this
command. `docs/benchmark-sets.md`, "Running vendor code", has the steps. At benchmark scale the
`sandbox-record` workflow runs it a few sets at a time, with `--set` and a larger `--memory`, on the
`bellwether-sandbox` runners.

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
| `tier` | 1: the maintainer's priority models, in the order given; 2: created in the twelve months before the build; 3: the rest. Within tiers 2 and 3, by 30-day downloads. `null` where the Hub decides the tier and was not asked (`--registry-only`) or gave no answer; those rows come after tier 2, by id |
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
docker/vendor/       the vendor-code oracle's base image (bellwether sandbox-record)
waivers/             engine_defects.toml
tests/
```

## Related

- [smg](https://github.com/smg-project/smg): the gateway under test; `crates/mock_worker` is the replay engine.
- [providers-verifier](https://github.com/smg-project/providers-verifier): record a vendor's API, replay against SMG. Its request catalogue is bellwether's first corpus.
- [deepseek-provider-verifier](https://github.com/smg-project/deepseek-provider-verifier).
- [artifacts](https://github.com/smg-project/artifacts): published evidence.

Apache-2.0.
