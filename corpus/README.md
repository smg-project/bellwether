# Corpus

The requests that `bellwether record` runs through the oracles. One JSON Lines file per set:

```
corpus/
  render/
    common.jsonl         # cases every model records
    <slug>/<set>.jsonl   # cases for one model, added to the set of the same name
```

A line is `{"name": "...", "request": {...}, "notes": "..."}`. `name` is a lowercase slug and
becomes the last part of the fixture id (`<slug>/render/<name>`); `request` is an OpenAI chat
completion body without `model` (the manifest supplies it); `notes` says what the case probes.

Cases are inputs, so they may be written by hand or imported from a vendor catalogue; the
recorded results next to them under `fixtures/` may not.
