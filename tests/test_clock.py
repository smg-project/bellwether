import json
import shutil
from datetime import date

import pytest
from transformers.utils import chat_template_utils

from bellwether.record import clock
from bellwether.record.reference import HfTemplateOracle

EPOCH = "1791979200"  # 2026-10-14 12:00:00 UTC
DAY = "2026-10-14"


@pytest.fixture
def unpinned(monkeypatch):
    """The module's clock as transformers ships it, restored after the test."""
    monkeypatch.setattr(chat_template_utils, "datetime", chat_template_utils.datetime)


def test_strftime_now_renders_the_pinned_day_once_the_clock_is_pinned(monkeypatch, unpinned):
    monkeypatch.setenv(clock.ENV, EPOCH)
    assert clock.pin_template_clock() is not None
    rendered = chat_template_utils._compile_jinja_template("{{ strftime_now('%Y-%m-%d') }}").render()
    assert rendered == DAY


def test_without_the_variable_the_clock_stays_the_real_one(monkeypatch, unpinned):
    monkeypatch.delenv(clock.ENV, raising=False)
    assert clock.pin_template_clock() is None
    rendered = chat_template_utils._compile_jinja_template("{{ strftime_now('%Y-%m-%d') }}").render()
    assert rendered == date.today().isoformat()


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
    monkeypatch.setenv(clock.ENV, "1791374400")
    assert oracle.provenance()["source_date_epoch"] == EPOCH
    monkeypatch.delenv(clock.ENV)
    assert oracle.provenance()["source_date_epoch"] == EPOCH
    assert "source_date_epoch" not in HfTemplateOracle(str(checkpoint), "local").provenance()
