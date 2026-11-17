import json
import random
import shutil

import pytest

from bellwether.record import draws
from bellwether.record.roundtrip import RoundtripOracle

# A nonce after every role, drawn with Jinja's ``random`` filter: no two renders of one conversation agree.
DRAWING = (
    "{%- for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + ' ' + (range(100000) | random | string) + '\\n' }}"
    "{{ m['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{ '<|im_start|>assistant ' + (range(100000) | random | string) + '\\n' }}"
    "{%- endif %}"
)
REQUEST = {"messages": [{"role": "user", "content": "Hi"}]}


@pytest.fixture
def drawing_model(tiny_model, tmp_path):
    """The tiny checkpoint with a template that draws."""
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(tiny_model, checkpoint)
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    config["chat_template"] = DRAWING
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))
    return checkpoint


def test_the_round_trip_records_a_turn_of_a_template_that_draws(drawing_model):
    oracle = RoundtripOracle(str(drawing_model), "local")
    assert oracle.render_output(REQUEST, {"content": "Paris."}).text == "Paris."


def test_the_round_trip_gives_the_generator_back_as_it_was(drawing_model):
    oracle = RoundtripOracle(str(drawing_model), "local")
    random.seed(20261117)
    before = random.getstate()
    oracle.render_output(REQUEST, {"content": "Paris."})
    assert random.getstate() == before


def test_two_renders_under_equal_draws_draw_the_same():
    with draws.equal_draws():
        first = random.random()
    with draws.equal_draws():
        second = random.random()
    assert first == second


def test_equal_draws_gives_the_generator_back_when_the_render_raises():
    before = random.getstate()
    with pytest.raises(RuntimeError), draws.equal_draws():
        random.random()
        raise RuntimeError("a render that fails")
    assert random.getstate() == before
