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

Every line validates against `schemas/case.schema.json`. Fixtures are recorded by
`bellwether record`, never edited by hand; a re-record is a pull request whose diff is the review.
Each reference carries its provenance (oracle versions, Hugging Face revision, vendor file sha,
bellwether commit).
