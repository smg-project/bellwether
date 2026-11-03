import json
import shutil
from datetime import UTC, datetime

import pytest
from transformers.utils import chat_template_utils

from bellwether.record import clock
from bellwether.record.reference import HfTemplateOracle
from bellwether.record.roundtrip import RoundtripOracle

EPOCH = "1791374400"  # 2026-10-07 12:00:00 UTC: an instant SOURCE_DATE_EPOCH names in these tests
DAY = "2026-10-07"
RECORDING_DAY = "2026-10-14"  # the day clock.RECORDING_DAY names


@pytest.fixture
def unpinned(monkeypatch):
    """The module's clock as transformers ships it, restored after the test."""
    monkeypatch.setattr(chat_template_utils, "datetime", chat_template_utils.datetime)


def test_strftime_now_renders_the_pinned_day_once_the_clock_is_pinned(monkeypatch, unpinned):
    monkeypatch.setenv(clock.ENV, EPOCH)
    assert clock.pin_template_clock() == datetime(2026, 10, 7, 12, 0)
    rendered = chat_template_utils._compile_jinja_template("{{ strftime_now('%Y-%m-%d') }}").render()
    assert rendered == DAY


def test_without_the_variable_the_clock_is_the_recording_day(monkeypatch, unpinned):
    monkeypatch.delenv(clock.ENV, raising=False)
    assert clock.pinned_value() == clock.RECORDING_DAY
    assert clock.pin_template_clock() == datetime(2026, 10, 14, 12, 0)
    rendered = chat_template_utils._compile_jinja_template("{{ strftime_now('%Y-%m-%d') }}").render()
    assert rendered == RECORDING_DAY


def test_a_blank_variable_reads_as_unset(monkeypatch):
    monkeypatch.setenv(clock.ENV, "  ")
    assert clock.pinned_value() == clock.RECORDING_DAY


def test_the_recording_day_is_noon_utc_so_every_zone_formats_the_same_day():
    instant = datetime.fromtimestamp(int(clock.RECORDING_DAY), tz=UTC)
    assert instant.strftime("%Y-%m-%d %H:%M") == f"{RECORDING_DAY} 12:00"


def test_the_reference_renders_at_the_epoch_and_its_provenance_names_it(monkeypatch, unpinned, tiny_model, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(tiny_model, checkpoint)
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    config["chat_template"] = (
        "Today: {{ strftime_now('%Y-%m-%d') }}\n{% for m in messages %}{{ m['content'] }}{% endfor %}"
    )
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))
    monkeypatch.setenv(clock.ENV, EPOCH)
    oracle = HfTemplateOracle(str(checkpoint), "local")
    rendered = oracle.render({"messages": [{"role": "user", "content": "Hello"}]})
    assert rendered.text.startswith(f"Today: {DAY}\n")
    assert oracle.provenance()["source_date_epoch"] == EPOCH
    # The provenance names the epoch the oracle was built with, whatever the variable says later.
    monkeypatch.setenv(clock.ENV, "1791979200")
    assert oracle.provenance()["source_date_epoch"] == EPOCH
    monkeypatch.delenv(clock.ENV)
    assert oracle.provenance()["source_date_epoch"] == EPOCH
    # Built without the variable, an oracle renders at the recording day and its provenance names that day.
    unset = HfTemplateOracle(str(checkpoint), "local")
    assert unset.provenance()["source_date_epoch"] == clock.RECORDING_DAY
    rendered = unset.render({"messages": [{"role": "user", "content": "Hello"}]})
    assert rendered.text.startswith(f"Today: {RECORDING_DAY}\n")
    # The first oracle keeps rendering at the instant its provenance names after the second one pinned the clock at
    # another: each render re-points the clock at the oracle's own instant.
    rendered = oracle.render({"messages": [{"role": "user", "content": "Hello"}]})
    assert rendered.text.startswith(f"Today: {DAY}\n")
    assert oracle.provenance()["source_date_epoch"] == EPOCH


def test_the_round_trip_renders_at_its_own_oracles_instant(monkeypatch, unpinned, tiny_model, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(tiny_model, checkpoint)
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    config["chat_template"] = (
        "Today: {{ strftime_now('%Y-%m-%d') }}\n{% for m in messages %}{{ m['content'] }}{% endfor %}"
    )
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))
    monkeypatch.setenv(clock.ENV, EPOCH)
    roundtrip = RoundtripOracle(str(checkpoint), "local")
    monkeypatch.delenv(clock.ENV)
    HfTemplateOracle(str(checkpoint), "local")  # built later, at the recording day: it pins the clock there
    rendered = roundtrip.chat([{"role": "user", "content": "Hello"}], {}, generation_prompt=True)
    assert rendered.startswith(f"Today: {DAY}\n")
    assert roundtrip.renderer.provenance()["source_date_epoch"] == EPOCH
