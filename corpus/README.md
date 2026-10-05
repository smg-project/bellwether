# Corpus

The requests that `bellwether record` runs through the oracles. One JSON Lines file per set:

```
corpus/
  render/
    common.jsonl         # cases every model records
    <slug>/<set>.jsonl   # cases for one model, added to the set of the same name
```

A line is `{"name": "...", "request": {...}, "notes": "..."}`. `name` is a lowercase slug and
becomes the last part of the fixture id (`<slug>/<kind>/<name>`); `request` is an OpenAI chat
completion body without `model` (the manifest supplies it); `notes` says what the case probes.

A parse case adds `"message"`: the assistant message the output must parse to (`content`,
`reasoning_content`, `tool_calls` with `function.name` and `function.arguments` as the JSON string the
parser must return, byte for byte). The round-trip oracle renders that message as the final assistant
turn of `request` through the model's template; the text after the generation prompt is the output.

Cases are inputs, so they may be written by hand or imported from a vendor catalogue; the
recorded results next to them under `fixtures/` may not.
