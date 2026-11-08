# Recording fixtures

How a checkpoint's fixtures are made: its manifest and group, the render and parse references, the vendor-code
sandbox, and the shape patterns' reference. `README.md` has the vocabulary and the rules on who is right;
`fixtures/README.md` describes the tree the recordings go to.

## Checkpoint groups and manifests

```bash
HF_HUB_OFFLINE=1 uv run bellwether manifests
```

Reads the list of checkpoints committed beside the manifests, `fixtures/models.tsv` (`--models` names another file):
one `model<TAB>revision<TAB>downloads<TAB>day<TAB>tier` row per checkpoint, with the Hub's downloads over the last 30
days and the day they were read. It computes each one's oracle inputs (the tokenizer files, the chat templates, and the
few fields of `config.json`, `generation_config.json`, `chat_template.json` and `processor_config.json` that the oracles
read) at the pinned revision from the Hugging Face cache, groups the checkpoints whose inputs are equal, and writes
every checkpoint's `fixtures/<slug>/manifest.toml`. A group is recorded once, under its primary's slug: the recorded one
if there is one, else its most-downloaded member's. The others' manifests name the group and hold no fixtures. A
checkpoint of an architecture the engine's server renders with its copy of a prompt encoder gets
`encoder = "<the engine's tokenizer mode>"`, which sends its group to the sandbox workflow. It prints each group with its
members and their tiers. Run again over the same list, it writes the same files, so a
change to the list is reviewed as a diff of the list and the manifests. `fixtures/README.md` describes both.
`models.jsonl` (`docs/models.md`) is another list: what the engines' registries name, with no revision or downloads until the Hub
is read.

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
stderr and left out, and the command exits 1. A request without tools renders with the processor's
template when the checkpoint ships one, as the engine's server does, which builds the processor: that
needs the processor's backends installed beside the recorder, as the engine's image ships them, torch,
torchvision and pillow, and what a processor imports itself (SmolVLM's `num2words`); a build that
fails for one of them is the environment's failure, not the checkpoint's, and the run stops naming the
package to install. Engine witnesses (`--oracle vllm|sglang`) run on Linux
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

The round trip proves a call reached the turn by renaming it, so a template that renders a call's
id and never its name (the Kimi K2 line's `<|tool_call_begin|>{{ id }}`; a served model writes the
name inside the id, `functions.<name>:<index>`) would have every hand-written call case refused,
since those cases carry no id. For a checkpoint whose tokenizer is the Kimi line's class, the round
trip first gives the case's calls the ids a served model writes (`record/vendor.py`,
`call_id_form`), the renaming probe renames the id with the call, and the recorded reference keeps
the ids; a template that writes the name (Kimi-K3's) gets none, and its calls stay as the case
gives them.

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

## Recording the shape patterns' reference

```bash
uv run bellwether patterns            # writes patterns/regex.jsonl, search.jsonl, classes.json
uv run bellwether patterns --check    # CI: records again and compares, byte for byte
```

A Hugging Face response template writes the parts of a tool call as regular expressions, which
transformers compiles with Python's `regex` module. Symphony, SMG's parser, lowers each part into a
matcher of its own, so it needs the module's answers to hold itself to. `patterns` records them at
the `regex` version `uv.lock` pins: for every pattern a call shape runs in the public response
templates (the header after a call's marker, the text before its closing marker, a parameter's tag
with the ways its value ends), what `match` gives over written texts, their prefixes and seeded
variations; where `search` closes a region before a marker; and the module's `\w`, `\s` and `\d`
tables with a set of probe characters. The recordings are plain files under `patterns/`, and
`unpack` copies them into the tree consumers read, so Symphony's tests find them beside the models'
sets. `unpack --plain` writes the hand-written sets alone, each `sets.toml` trimmed to them, with nothing
fetched; `--sample-sets N` adds one compressed set in N, whole, each taken by its own id, so a consumer with a
time or a disk budget reads every model's hand-written suite and a steady sample of the benchmark sets: the
same sets on every run of the same tree, and a set one group gains or loses moves no other group's sample. The reference here is the module itself, pinned; a bump of the pin re-records, and the diff is
the review.
