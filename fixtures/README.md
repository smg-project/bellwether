# Fixtures

One directory per model, named by a lowercase slug of the Hugging Face id:

```
fixtures/
  kimi-k3/
    manifest.toml          # model, pinned revision, authority order, SMG and engine parser names
    render/*.jsonl         # request -> prompt token ids
    parse/*.jsonl          # output token ids -> response, whole and per chunk plan
    tokenize/*.jsonl       # text -> ids
    detokenize/*.jsonl     # ids -> incremental text pieces
```

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
canonical JSON, checked against the schema. A re-run with the reference oracle replaces each line's
`reference` and keeps its `witnesses`; an engine oracle does the reverse.

