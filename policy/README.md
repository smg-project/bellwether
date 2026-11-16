# Policy

What SMG's own routing policy code picks, recorded row by row. SMG's gateway chooses a worker for each request
with a load-balancing policy, and its policy controller crate (`crates/policy_controller`) builds those policies a
second time. The crate's test `tests/classic_rows.rs` replays these rows through the new code and fails on a row it
decides differently. The reference here is the gateway's policy code at a pinned commit, run, never read.

One sheet records no pick: `pair_index.json` holds which prefill and decode workers the gateway pairs, the lists a
pair's placement then picks from.

One sheet records one rule under every policy: `all_vetoed.json` holds what the gateway answers when every usable
worker of a pool is over its overload thresholds, and what a routing key gets under consistent hashing while its
own worker is over them.

One sheet records a request that carries encode items: `encode_pair.json` holds the pair the gateway picks for it,
the encoders it offers the request's items and the encoder each item takes, or the name of its refusal.

One sheet records the placement of a pair: `pair_select.json` holds which prefill and which decode the gateway
places a request on, over two pools its pair index has paired.

```
policy/
  power_of_two.json        one sheet per policy: its recorded rows
  manual.json
  pair_index.json
  all_vetoed.json
  encode_pair.json
  pair_select.json
  classic_rows.rs               the recorder of power_of_two.json
  classic_manual_rows.rs        the recorder of manual.json
  classic_pair_rows.rs          the recorder of pair_index.json
  classic_all_vetoed_rows.rs    the recorder of all_vetoed.json
  classic_encode_pair_rows.rs   the recorder of encode_pair.json
  classic_pair_select_rows.rs   the recorder of pair_select.json
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

The rows of `all_vetoed.json` are sequence rows and drawn rows of one shape. A row names the policy its pool runs
under and whether shedding is on. Each worker has the requests the router holds on it and a state: `free` (ready,
under its thresholds), `over` (ready, over them), `stalled` (ready, doubted by the liveness tracker) or `down` (not
ready). A step is a pick (with the routing key the request carries, if any, and the workers an earlier attempt
tried), new loads for every worker, or new states. A pick is recorded as the worker's place in the row's list. A
request placed nowhere is a number past the last worker: the count of the workers when the gateway sheds it as an
overload shed, the count plus one when it finds no worker to use.

A ring row carries `ring_key`: its request is placed by that routing key on the hash ring. Two rings put one key on
different workers, so a ring row lists its workers in the key's ring order: the worker the key lands on first, then
the one it passes to when that worker is down, and so on. The recorder finds that order on the gateway's ring by
placing the key with every worker free and then with the members found so far down, and a reader finds it the same
way on its own ring. A ring row's loads, states, tried lists and answers all count in that order, so an answer of 0
is the key's own worker and 1 the next member of the ring, on whatever ring replays the row. A row under consistent
hashing with no `ring_key` records what a request without a routing key gets.

The rows of `encode_pair.json` are sequence rows of their own shape. A row is a fleet: its encoders, its prefill
workers and its decode workers, each named by its place in its list, with what the gateway reads of it from its
registration and the requests the router holds on it (`in_flight`). An encoder has its `runtime`, the `handoff` its
runtime hands an encoded item to a prefill worker in (`push`, `rendezvous`, or none), whether it states the
producer role of a `connector` (`ec_producer`), whether it takes `images`, whether it processes `references` itself
and whether it is ready (`open`). A prefill worker has its consumer role and connector and its `partners`, the
decode workers the gateway pairs it with. A decode worker has whether its registration says it takes images; the
gateway holds the decode workers of one runtime to that and no other's. A row names the policy each side of its
pair is picked by and the policy an item's encoder is picked by. No row draws: under `power_of_two:least_requests`
a row leaves the policy at most two workers, which hold different numbers of requests, and it takes the one that
holds fewer. A request has the key of each of its encode items and what it asks: whether the items are references,
whether they carry images, whether the decode side must take images, and whether each item is encoded on its own
(`fanout`). The answer to a request is `placed` (the prefill and the decode worker, the encoders the items were
`offered`, the `keys` the items were placed by and the encoder each key took) or `refused` (the refusal's `name`; a
refused placement of the pair adds the `role` it names and `why`). Without `fanout` a pushing runtime places a
request's items as one key, a hash of them, so such an answer has one key however many items the request has. Under
`consistent_hashing` a key's encoder is the owner the gateway's ring gives it among the encoders offered; another
ring gives the same key another of them.

The rows of `pair_select.json` are sequence rows and drawn rows whose pick is a pair. A row names the policy each
pool runs under (`power_of_two` carries its load metric after a colon), whether shedding is on and, when the row
runs behind the prefill admission, the most requests that admission lets one prefill hold (`prefill_bound`). Each
worker has the requests the router holds on it, a state as in the all-vetoed sheet and the KV transport its
registration names (`a`, `b` or none): two workers pair when their transports are equal or one of them names none.
A decode may report a running window, with the rooms of it the decode admission has claimed (`rooms`). A row places
its requests one after another, and no request changes what the router holds. An answer is the pair, the prefill's
place in its list and then the decode's, or no pair with the role the gateway named and why: `overloaded` (an
overload shed), `unavailable`, `at_capacity` (the chosen prefill holds its bound), `no_candidates` or `declined`. A
drawn row counts how often each pair took its one request; a pair never placed is not listed.

Two choices keep those rows replayable. No row runs under `round_robin`: the gateway starts the rotation of a
candidate set it has not seen at a shared count, so which pick comes first is not the policy's rule. Where a row
must replay pick for pick, a pool runs under `power_of_two:least_requests`, which takes the fewest requests in
flight with no draw and no state. Where a row must show whether a side's policy was asked at all, that side runs
under `random`: where the gateway places by its own rule, one pair takes every draw.

`pair_index.json` holds a third kind, which picks nothing:

- **pair row:** a prefill pool and a decode pool, with what each worker offers a partner as the gateway derives it
  from the worker's registration (its runtime, an operator's protocol, the KV transport, the engine version, the KV
  layout facts) and the pairing key the gateway names the worker's group by. For each of the gateway's two pair
  wires and each of its three pairing modes the row records the verdict on every prefill and decode (the component
  the pair is refused on, or none), each prefill's partners by index in the decode pool, and the refusal the gateway
  reports when no pair is left. Nothing in a row draws: the recorder builds each row twice and stops if the two
  builds differ, so a pair row is the same in every recording of one commit and replays whole or not at all.

## The sheets

Each sheet is recorded on its own, so the commit is a column of the sheet's row and never one commit for the
directory. The sha256 is of the file as committed.

| sheet | policy | recorded from (SMG commit) | rows | sha256 | recorder |
|---|---|---|---|---|---|
| `power_of_two.json` | power of two, under each of its four load metrics | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 44: 24 sequence, 20 drawn | `602e1fe790cc5368ad05de2297b6e482b7a8893704ee448699ec467df9e0d87c` | `classic_rows.rs` |
| `manual.json` | manual, under each of its assignment modes | `1b141d03b5ca9097f54f0ebc4b95d72e810a62c3` | 14: 9 sequence, 5 drawn | `d34189025a50a7a31f925c632ce1964a09dac0baab89370e6e87c5c842db9a5a` | `classic_manual_rows.rs` |
| `all_vetoed.json` | every policy, over a pool whose every usable worker is over its overload thresholds; a routing key under consistent hashing whose own worker is over them | `d9f8018881cc54c9b0f6e993849a59f8f7c54ab5` | 42: 38 sequence, 4 drawn | `e5496add4cc24e8438e60849570c6f0113f3537eeb2010f448fb8af5a432527b` | `classic_all_vetoed_rows.rs` |
| `pair_index.json` | the prefill and decode pair index, on both pair wires under each pairing mode | `64ff5a437b5007af5079f0492b46f4e5a712b66f` | 23 pair rows | `383783a8a28bdc3d0753111fbb84df4be2713fe332b448938b9cb75f44dec694` | `classic_pair_rows.rs` |
| `encode_pair.json` | a request with encode items: the pair, the encoders offered and each item's encoder | `4ba2a6431357be5411a2383cdc4916b41471d722` | 52: 52 sequence, 0 drawn | `988bf253da9310aeedd08ec2704d8e01ecc4a1210453353723ba0faa9001de33` | `classic_encode_pair_rows.rs` |
| `pair_select.json` | the placement of a prefill and decode pair: each side under its policy, a side over its overload thresholds or doubted whole, a decode's rooms, the prefill admission's bound | `4ba2a6431357be5411a2383cdc4916b41471d722` | 33: 18 sequence, 15 drawn | `1ee16ff0019bd2f57564ef556cc9e8ebebb53143667a72ad30c13a5fdd4805fa` | `classic_pair_select_rows.rs` |

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

For the manual sheet the test path is `policies::classic_manual_rows`; for the all-vetoed sheet the module line is
`mod classic_all_vetoed_rows;` and the test path `policies::classic_all_vetoed_rows`.

The pair sheet's recorder is a child of the module that derives a worker's pairing descriptor, so it records the
descriptor the gateway derived and never a second reading of the registration. Copy it to
`model_gateway/src/worker/pd/pairing/classic_pair_rows.rs`, add `#[cfg(test)] mod classic_pair_rows;` at the end of
`model_gateway/src/worker/pd/pairing.rs`, and run the test path `worker::pd::pairing::classic_pair_rows` with the
same two variables.

The encode sheet's recorder is a child of the gateway's worker selection stage, so it runs the stage's own
selection and reads a worker by the stage's own functions. Copy it to
`model_gateway/src/routers/grpc/common/stages/worker_selection/classic_encode_pair_rows.rs`, add
`#[cfg(test)] mod classic_encode_pair_rows;` at the end of
`model_gateway/src/routers/grpc/common/stages/worker_selection.rs`, and run the test path
`worker_selection::classic_encode_pair_rows` with the same two variables. It wraps the encode policy to write down
each call (the encoders offered and the key) and passes every call on unchanged.

The pair placement sheet's recorder sits beside the placement it calls, and builds its pools, policies and pair
index with the gateway's own types. Copy it to `model_gateway/src/routers/common/classic_pair_select_rows.rs`, add
`#[cfg(test)] mod classic_pair_select_rows;` at the end of `model_gateway/src/routers/common/mod.rs`, and run the
test path `routers::common::classic_pair_select_rows` with the same two variables.

A sheet is never edited by hand. A new recording replaces the whole sheet and its row in the table above, in one
pull request that names the SMG commit and says which rows moved: a sequence row moves only when the policy's
behavior did, and the drawn rows are a new sample.

## Reading

SMG pins this repository by commit, in the files under `.github/versions/` of SMG. Its replay reads the directory
that `SMG_POLICY_CLASSIC_ROWS` names:

```bash
SMG_POLICY_CLASSIC_ROWS=<this checkout>/policy \
  cargo test -p smg-policy-controller --features ingest --test classic_rows -- --ignored
```

A few rows of each sheet are also copied into the crate's test source, so every run of its suite replays a subset.

## Limits left

A limit is a recorded row the replay does not decide as recorded, with the reason. The replay names each one and
fails once the reason no longer holds.

- `power_of_two.json`: none. All 44 rows replay.
- `all_vetoed.json`: four rows are owed, the two rows each of the `prefix_hash` and the `bucket` policy (a pool over
  on every worker, with shedding off and on). The gateway answers them as it answers every other policy; the crate
  has no preset of either name yet, so it cannot build the rows. The other 38 rows replay. Nine of them, the eight
  ring rows and the keyed round robin row, replay with SMG's rule for a routing key under the ring; SMG's replay
  reads them from the commit its pin then names.
- `pair_index.json`: none is recorded yet. The module that replays it arrives in SMG with the crate's pair classes;
  SMG's replay reads this sheet from the commit its pin then names.
- `encode_pair.json`: none is recorded yet. The module that replays it arrives in SMG with the crate's encoder lists
  and its rule for encode items; SMG's replay reads this sheet from the commit its pin then names.
- `pair_select.json`: none is recorded yet. The module that replays it arrives in SMG with the crate's pair
  decision; SMG's replay reads this sheet from the commit its pin then names.
- `manual.json`: one row is owed, `manual-one-key-pins-apart-under-the-prefill-and-the-decode-role`. The gateway
  keeps a key's pins apart under each role of a prefill and decode pair. The crate refuses a sessions rule under a
  pair plan today, so it cannot build the row; the other 13 rows replay.
