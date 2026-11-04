"""PixMo-Cap's pictures as an image set, a window at a time, each picture fetched from the web once and kept: a
snapshot set.

The data is the PixMo-Cap dataset on the Hugging Face Hub at a pinned commit: 4 parquet files of the train split, each
row a picture's URL (``image_url``), a caption and transcripts, and no bytes. The pictures live on the web under their
own terms; the dataset's licence, ODC-By, covers the annotations. So the set is a snapshot: ``bellwether import
pixmocap`` fetches each URL of the window once, writes the picture as the other image sets are written, and writes the
table ``corpus/sources/pixmocap.tsv`` (row, url, fetched, status, sha256, bytes) that records what the web served on
that day, misses included. A row already in the table is never fetched again: its outcome stands, so the names already
written do not move; a later window appends its rows after the earlier ones. ``--check`` compares the committed set,
its files and the licence copy with the table and fetches nothing, so the weekly check needs no web; a re-import that
extends the table needs the set's files fetched from Git LFS first, since it rebuilds every line from the bytes.

The dataset card states ``odc-by``, which the import checks on every run. The dataset ships no licence file, so the
import copies the ODC Attribution License text as the SPDX license list publishes it (``text/ODC-By-1.0.txt``), from
that list's repository at a pinned commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bellwether import storage

from . import github, hf, image_sets

DATASET = "pixmocap"
REPO = "allenai/pixmo-cap"
REVISION = "edce6390d9d5be6c8db0d863fbe62718c88988a4"
SOURCE = hf.source(REPO, REVISION)
# The dataset card (README.md) at the revision, by its sha256, whose front matter states the licence.
CARD_SHA256 = "c8229220121fd298b3e527cccc306c0f5b9ac706988a0d213414e7942ebe228b"
LICENSE = "ODC-By-1.0"
CARD_LICENSE = "odc-by"  # the license key of the reviewed card's YAML front matter; LICENSE is its SPDX name
# The licence text as the SPDX license list publishes it, from the list's repository at one commit.
LICENSE_OWNER, LICENSE_REPO = "spdx", "license-list-data"
LICENSE_COMMIT = "31846481de7ea17d92f2f46fad76abe58b2e9bb0"
LICENSE_PATH = "text/ODC-By-1.0.txt"
LICENSE_SHA256 = "b1c4de4bf91c95eec945ea7cbe0575fd52e7ee8fa430acb01d323d8214b1c482"
LICENSE_COPY = "licenses/pixmocap-LICENSE"
SPLIT = "train"
COLUMN = "image_url"
# The files read so far, by the sha256 of their bytes.
FILES = {
    "data/train-00000-of-00004.parquet": ("6bb4157ea05b975cb6b647f06d2619a65b5bb5dc2f0551b4c051544a2b4827ba"),
}
# The windows of rows that are in the set, in the order the lines are built: a file, its first row and the count. A
# window added later goes last, whether of a new file or of more rows of one already read.
SHARDS = (("data/train-00000-of-00004.parquet", 0, 2000),)
TABLE = "sources/pixmocap.tsv"
TABLE_FIELDS = ("row", "url", "fetched", "status", "sha256", "bytes")
TIMEOUT = 30.0
MAX_BYTES = 64 * 1024 * 1024
USER_AGENT = "bellwether/0.0.1 (+https://github.com/smg-project/bellwether)"
FETCHERS = 8  # URLs fetched at once


def check_license(text: bytes) -> None:
    """Refuse a licence copy that is not the ODC Attribution License.

    ``fetch`` has already held the file to ``LICENSE_SHA256``, the reviewed text. Its first line is checked as well,
    so that moving the pin to a commit whose file is another text cannot pass on an updated hash alone.
    """
    lines = [line.strip() for line in text.decode("utf-8").splitlines() if line.strip()]
    if lines[:1] != ["# ODC Attribution License (ODC-By)"]:
        raise ValueError(f"{LICENSE_PATH} at {LICENSE_COMMIT}: not the ODC Attribution License; review it")


def origin(file: str, row: int, url: str, fetched: str) -> dict:
    """Where a line came from: the dataset, its source, the file the URL was read from with its sha256, the row and
    the column, the URL the bytes were fetched from and the day, and the licence (the annotations'; the picture is a
    web image under its own terms)."""
    return {
        "dataset": DATASET,
        "source": SOURCE,
        "sha256": FILES[file],
        "file": file,
        "row": row,
        "column": COLUMN,
        "url": url,
        "fetched": fetched,
        "license": LICENSE,
    }


def parquet_urls(path: Path, first: int, count: int) -> Iterator[tuple[int, str]]:
    """``(row, url)`` for the rows of the window, in row order; a null or empty URL gives nothing."""
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    row = 0
    for batch in file.iter_batches(batch_size=64, columns=[COLUMN]):
        urls = batch.to_pydict()[COLUMN]
        for index, url in enumerate(urls):
            number = row + index
            if number >= first + count:
                return
            if number >= first and isinstance(url, str) and url:
                yield number, url
        row += batch.num_rows


def fetch_url(url: str) -> tuple[str, bytes | None]:
    """The bytes at ``url``, or why there are none: ``(status, data)`` with ``status`` the HTTP status as a string
    (``200`` with data), or ``timeout``, ``unreachable``, ``too-large``, ``not-an-image``. Redirects are followed as
    urllib follows them; the body is read up to ``MAX_BYTES``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as err:
        return str(err.code), None
    except TimeoutError:
        return "timeout", None
    except (urllib.error.URLError, OSError, ValueError) as err:
        if isinstance(getattr(err, "reason", None), TimeoutError) or "timed out" in str(err):
            return "timeout", None
        return "unreachable", None
    if len(data) > MAX_BYTES:
        return "too-large", None
    if image_sets.format_of(data) is None:
        return "not-an-image", None
    return "200", data


def read_table(path: Path) -> dict[tuple[str, int], dict]:
    """The committed table by (file, row); empty when there is none."""
    rows: dict[tuple[str, int], dict] = {}
    if not path.is_file():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split("\t")) != ("file",) + TABLE_FIELDS:
        raise ValueError(f"{path}: not the pixmocap table (its header is {lines[0] if lines else 'empty'!r})")
    for number, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(TABLE_FIELDS) + 1:
            raise ValueError(f"{path}:{number}: {len(parts)} fields, not {len(TABLE_FIELDS) + 1}")
        entry = dict(zip(("file",) + TABLE_FIELDS, parts, strict=True))
        entry["row"] = int(entry["row"])
        entry["bytes"] = int(entry["bytes"]) if entry["bytes"] else None
        rows[(entry["file"], entry["row"])] = entry
    return rows


def table_text(entries: list[dict]) -> str:
    head = "\t".join(("file",) + TABLE_FIELDS) + "\n"
    body = "".join(
        "\t".join(
            [e["file"], str(e["row"]), e["url"], e["fetched"], e["status"], e["sha256"] or "", str(e["bytes"] or "")]
        )
        + "\n"
        for e in entries
    )
    return head + body


def committed_bytes(corpus_dir: Path, sha256: str, lines: list[dict]) -> bytes:
    """The bytes of the committed picture with ``sha256``, read from its file; a pointer is refused with the fetch
    command that brings the bytes."""
    for line in lines:
        if line.get("sha256") == sha256:
            path = corpus_dir / line["path"]
            data = path.read_bytes() if path.is_file() else b""
            if hashlib.sha256(data).hexdigest() == sha256:
                return data
            raise ValueError(
                f"{path}: not the picture's bytes (a pointer, or missing); fetch the set's files first: "
                f"git lfs pull --include 'corpus/images/{DATASET}/**' --exclude ''"
            )
    raise ValueError(f"the table names sha256 {sha256} and no line of the set carries it")


def read(files: dict[str, Path], corpus_dir: Path, today: str) -> tuple[Iterator[image_sets.Found], list[dict]]:
    """The window's pictures in row order, from the committed table where it has the row and from the web where it
    does not, and the table as it stands after the fetches (written by ``run`` after the set)."""
    table = read_table(corpus_dir / TABLE)
    set_file = image_sets.set_path(corpus_dir, DATASET)
    committed = image_sets.read_set(set_file) if set_file.is_file() else []
    rows = [(file, row, url) for file, first, count in SHARDS for row, url in parquet_urls(files[file], first, count)]
    for file, row, url in rows:
        entry = table.get((file, row))
        if entry is not None and entry["url"] != url:
            raise ValueError(f"{TABLE}: row {row} of {file} names {entry['url']!r}; the pinned file has {url!r}")
    # the rows the table does not have yet are fetched, a few at a time; the lines keep the rows' order whatever the
    # order the web answers in
    wanted = [(file, row, url) for file, row, url in rows if (file, row) not in table]
    fetched: dict[tuple[str, int], tuple[str, bytes | None]] = {}
    with ThreadPoolExecutor(max_workers=FETCHERS) as pool:
        futures = {pool.submit(fetch_url, url): (file, row) for file, row, url in wanted}
        for future in as_completed(futures):
            fetched[futures[future]] = future.result()
    entries: list[dict] = []
    found: list[image_sets.Found] = []
    for file, row, url in rows:
        entry = table.get((file, row))
        if entry is None:
            status, data = fetched[(file, row)]
            entry = {
                "file": file,
                "row": row,
                "url": url,
                "fetched": today,
                "status": status,
                "sha256": hashlib.sha256(data).hexdigest() if data else "",
                "bytes": len(data) if data else None,
            }
        elif entry["sha256"]:
            data = committed_bytes(corpus_dir, entry["sha256"], committed)
        else:
            data = None
        entries.append(entry)
        if data is not None:
            found.append(
                image_sets.Found(
                    data, origin(file, row, url, entry["fetched"]), f"PixMo-Cap {SPLIT} row {row}, {COLUMN}"
                )
            )
    return iter(found), entries


def check(corpus_dir: Path, files: dict[str, Path], copies: dict[str, bytes]) -> list[str]:
    """One line per difference between the committed set, its files, the table and the licence copy; no fetch.

    Every table row of the windows must exist, name the URL the pinned file names, and, when it has a sha256, have
    exactly one line of the set with that sha256 and size, in row order; every line's file must hold its sha256 and
    size (pointer or content), and a file that is content must give the line back through the catalogue; no file
    under the images directory may be unnamed; the licence copy must be the pinned text.
    """
    problems: list[str] = []
    table_path = corpus_dir / TABLE
    set_file = image_sets.set_path(corpus_dir, DATASET)
    if not table_path.is_file():
        return [f"{table_path}: missing"]
    if not set_file.is_file():
        return [f"{set_file}: missing"]
    table = read_table(table_path)
    lines = image_sets.read_set(set_file)
    by_sha = {line.get("sha256"): line for line in lines}
    expected_order: list[str] = []
    seen: set[str] = set()
    for file, first, count in SHARDS:
        for row, url in parquet_urls(files[file], first, count):
            entry = table.get((file, row))
            if entry is None:
                problems.append(f"{table_path}: no row for {file} row {row}")
                continue
            if entry["url"] != url:
                problems.append(f"{table_path}: {file} row {row} names {entry['url']!r}, the pinned file {url!r}")
            if not entry["sha256"] or entry["sha256"] in seen:
                # a miss, or bytes an earlier row already gave: no line of its own, as the writer leaves repeats out
                continue
            seen.add(entry["sha256"])
            line = by_sha.get(entry["sha256"])
            if line is None:
                problems.append(f"{set_file}: no line carries sha256 {entry['sha256']} ({file} row {row})")
                continue
            if (
                line["bytes"] != entry["bytes"]
                or line["origin"].get("row") != row
                or line["origin"].get("file") != file
            ):
                problems.append(f"{set_file}: the line {line['name']} and the table disagree on {file} row {row}")
            expected_order.append(line["name"])
    if [line["name"] for line in lines] != expected_order:
        problems.append(
            f"{set_file}: the lines are not the table's kept rows in order "
            f"({len(lines)} lines, {len(expected_order)} kept rows)"
        )
    expected_files: set[Path] = set()
    for number, line in enumerate(lines, start=1):
        path = corpus_dir / line["path"]
        expected_files.add(path)
        if not path.is_file():
            problems.append(f"{path}: missing")
            continue
        try:
            digest, size = image_sets.identity(path)
        except (OSError, ValueError) as err:
            problems.append(str(err))
            continue
        if (digest, size) != (line["sha256"], line["bytes"]):
            problems.append(f"{path}: holds sha256 {digest}, {size} bytes; the line says {line['sha256']}")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == digest:  # content, not a pointer: the catalogue is checked too
            origin_ = dict(line["origin"])
            fresh = image_sets._line(DATASET, number, image_sets.Found(data, origin_, line["notes"]))[0]
            if fresh != line:
                problems.append(f"{set_file}:{number}: differs from the line its bytes give")
    directory = image_sets.images_dir(corpus_dir, DATASET)
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.is_file() and p not in expected_files):
            problems.append(f"{path}: no row of the table writes it")
    for relative, content in sorted(copies.items()):
        path = corpus_dir / relative
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif path.read_bytes() != content:
            problems.append(f"{path}: differs from the pinned text")
    return problems


def fetch(filename: str, sha256: str, cache: Path) -> Path:
    """The path of a dataset file at the pinned commit, checked against ``sha256``."""
    return hf.fetch(REPO, REVISION, filename, sha256, cache)


def run(args: argparse.Namespace) -> int:
    hf.check_card_license(REPO, REVISION, CARD_SHA256, CARD_LICENSE, args.cache)
    license_text = github.fetch(
        LICENSE_OWNER, LICENSE_REPO, LICENSE_COMMIT, LICENSE_PATH, LICENSE_SHA256, cache=args.cache
    ).read_bytes()
    check_license(license_text)
    files = {name: fetch(name, sha256, args.cache) for name, sha256 in FILES.items()}
    copies = {LICENSE_COPY: license_text}
    if args.check:
        problems = check(args.corpus, files, copies)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the PixMo-Cap set equals its table and the pinned file ({SOURCE})")
        return 1 if problems else 0
    today = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    found, entries = read(files, args.corpus, today)
    _, summary = image_sets.write(DATASET, found, args.corpus, copies)
    storage.write_whole(args.corpus / TABLE, table_text(entries).encode("utf-8"))
    image_sets.report(DATASET, summary, args.corpus)
    misses = [e for e in entries if not e["sha256"]]
    print(f"{args.corpus / TABLE}: {len(entries)} rows, {len(entries) - len(misses)} pictures, {len(misses)} misses")
    return 0
