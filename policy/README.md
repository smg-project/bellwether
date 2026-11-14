# Policy

What SMG's own routing policy code picks, recorded row by row. SMG's gateway chooses a worker for each request
with a load-balancing policy, and its policy controller crate (`crates/policy_controller`) builds those policies a
second time. The crate's test `tests/classic_rows.rs` replays these rows through the new code and fails on a row it
decides differently. The reference here is the gateway's policy code at a pinned commit, run, never read.

```
policy/
  power_of_two.json        one sheet per policy: its recorded rows
  manual.json
  classic_rows.rs          the recorder of power_of_two.json
  classic_manual_rows.rs   the recorder of manual.json
```

## A sheet

A sheet is one JSON object: `recorded_from` (the SMG commit its recorder ran in), `policy`, and `rows`. A row is a
pool of workers, what a router knows of each one (the requests it holds on it, and the worker's load report or its
routing keys), one request shape and what the policy picked for it. There are two kinds of row:

- **sequence row:** a list of steps (a pick, or new facts about every worker) and the worker each pick chose, in
  order. The recorder runs the row twenty times, each time from a fresh policy, and stops if the runs disagree, so a
  sequence row is the same in every recording of one commit.
- **drawn row:** the policy draws, so the row records how often each worker was chosen over 6,000 single decisions,
  each from a fresh policy. The draws are the gateway's own and cannot be seeded, so a drawn row is one sample: a
  second recording of the same commit gives other counts within the same spread. Compare shares, never counts.
  SMG's replay takes two rows as one distribution when a worker never picked on one side is never picked on the
  other and every other share lies within four standard deviations of the difference of two samples of that size.

## The sheets

Each sheet is recorded on its own, so the commit is a column of the sheet's row and never one commit for the
directory. The sha256 is of the file as committed.

| sheet | policy | recorded from (SMG commit) | rows | sha256 | recorder |
|---|---|---|---|---|---|
| `power_of_two.json` | power of two, under each of its four load metrics | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 34: 23 sequence, 11 drawn | `e3a18cd14340a2590bca2327aa163a1ec379f21a5d335adc570abf24a1fff81e` | `classic_rows.rs` |
| `manual.json` | manual, under each of its assignment modes | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 12: 8 sequence, 4 drawn | `335b3506acf0a3223e0dc940b78fb35489c7e21b83f7e3f8e5b1f3b8db295030` | `classic_manual_rows.rs` |

## Recording

A recorder is a test module of SMG's gateway crate. It is not part of the gateway: it lives here, beside the rows
it records. In a checkout of SMG at the sheet's commit:

1. Copy the recorder to `model_gateway/src/policies/` under its own name.
2. Add `#[cfg(test)] mod classic_rows;` (for the manual sheet, `mod classic_manual_rows;`) to
   `model_gateway/src/policies/mod.rs`.
3. Run it, naming the output directory and the commit:

```bash
CLASSIC_ROWS_OUT=<dir> CLASSIC_ROWS_COMMIT=<commit> \
  cargo test -p smg --lib policies::classic_rows -- --ignored --nocapture
```

For the manual sheet the test path is `policies::classic_manual_rows`.

A sheet is never edited by hand. A new recording replaces the whole sheet and its row in the table above, in one
pull request that names the SMG commit and says which rows moved: a sequence row moves only when the policy's
behavior did, and the drawn rows are a new sample.

## Reading

SMG pins this repository by one commit (`.github/versions/bellwether.ref` in SMG). Its replay reads the directory
that `SMG_POLICY_CLASSIC_ROWS` names:

```bash
SMG_POLICY_CLASSIC_ROWS=<this checkout>/policy \
  cargo test -p smg-policy-controller --test classic_rows -- --ignored
```

A few rows of each sheet are also copied into the crate's test source, so every run of its suite replays a subset.

## Limits left

A limit is a recorded row the replay does not decide as recorded, with the reason. The replay names each one and
fails once the reason no longer holds.

- `power_of_two.json`: none. All 34 rows replay.
- `manual.json`: one row is owed, `manual-one-key-pins-apart-under-the-prefill-and-the-decode-role`. The gateway
  keeps a key's pins apart under each role of a prefill and decode pair. The crate refuses a sessions rule under a
  pair plan today, so it cannot build the row; the other 11 rows replay.
