import hashlib
import json
from collections import Counter

import huggingface_hub
import pytest

from bellwether.cli import main
from bellwether.importers import hermes, hf

REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
CARD = "---\nlicense: apache-2.0\ntask_categories:\n- text-generation\n---\n\n# Hermes Function-Calling V1\n"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOLS = json.dumps([WEATHER])
# The dataset's three system prompts, word for word, around a tools element: func_calling and
# func_calling_singleturn use the first two, glaive_func_calling the third.
FUNCTION_CALLING = (
    "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags."
    " You may call one or more functions to assist with the user query. Don't make assumptions about what values to"
    " plug into functions.\n<tools>\n" + TOOLS + "\n</tools>\nFor each function call return a json object with function"
    " name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>\n'
)
EXTRACTION = (
    "You are an expert structured information extraction AI model. You will be provided with documents to extract"
    " information from. You are also provided with the json schema to output extracted information in the function"
    " signatures within XML tags <tools></tools>. Don't make assumptions about what values to plug into json schema."
    " \n<tools>\n" + TOOLS + "\n</tools>\nFor each extraction function call return a json object with function name and"
    " arguments followed by a <tool_call> tag with the following schema:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-dict>}\n</tool_call>'
)
GLAIVE = (
    "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags."
    "You may call one or more functions to assist with the user query. Don't make assumptions about what values to"
    " plug into functions.Here are the available tools:<tools>\n" + TOOLS + "\n</tools>Use the following pydantic"
    " model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties':"
    " {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}},"
    " 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments"
    " within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>"
)


def test_the_hermes_tool_prompt_is_taken_out_of_the_system_message():
    for text in (FUNCTION_CALLING, EXTRACTION, GLAIVE):
        assert hermes.system_message(text, TOOLS) == ""


def test_text_around_the_tool_prompt_stays_as_the_system_message_stripped_at_its_ends():
    text = "You are Bob, a terse assistant.\n\n" + GLAIVE + "\nAnswer in French. "
    assert hermes.system_message(text, TOOLS) == "You are Bob, a terse assistant.\n\n\nAnswer in French."


def test_a_row_without_tools_keeps_its_system_message_as_it_is():
    text = "You are a helpful assistant, with no access to external functions. "
    assert hermes.system_message(text, None) == text


def test_a_system_message_whose_tool_list_is_not_the_rows_tools_is_unmappable():
    # The shape of func_calling row 1109 and 60 others: the field is "[]", the prompt holds a broken list.
    broken = EXTRACTION.replace(TOOLS, '[{"type": "object", "properties": {}}}]')
    with pytest.raises(hermes.Unmappable, match="the system message lists tools the row does not declare"):
        hermes.system_message(broken, None)
    with pytest.raises(hermes.Unmappable, match="the system message carries no Hermes tool prompt with the row's"):
        hermes.system_message(FUNCTION_CALLING, json.dumps([WEATHER, WEATHER]))


PARIS = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
ZURICH = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Z\\u00fcrich"}}\n</tool_call>'


def test_a_turn_of_calls_has_empty_content_and_a_prose_turn_is_its_content_byte_for_byte():
    calls = [
        {"name": "get_weather", "arguments": {"city": "Paris"}},
        {"name": "get_weather", "arguments": {"city": "Zürich"}},
    ]
    assert hermes.split_calls(PARIS + "\n" + ZURICH + "\n") == ("", calls)
    prose = "Here it is:\n- a\n- b \n"
    assert hermes.split_calls(prose) == (prose, [])


def test_prose_before_the_calls_is_the_content_without_the_whitespace_that_separates_it_from_them():
    content, calls = hermes.split_calls("\nLet me check both.\n\n" + PARIS + "\n" + ZURICH)
    assert content == "\nLet me check both." and len(calls) == 2


def test_text_after_a_call_is_unmappable_because_a_message_holds_its_content_before_its_calls():
    for text in (PARIS + "\nDone.", PARIS + "\nand\n" + ZURICH):
        with pytest.raises(hermes.Unmappable, match="text after a <tool_call> block"):
            hermes.split_calls(text)


def test_a_block_that_is_not_json_is_unmappable():
    # The shape of the 793 extraction rows: a backslash and an "n" where the newlines belong, and Python quotes.
    text = '<tool_call>\\n{"arguments": {"queries": [\'How?\']}, "name": "ExpertQAExtractor"}\\n</tool_call>'
    with pytest.raises(hermes.Unmappable, match="a <tool_call> block that is not JSON"):
        hermes.split_calls(text)


def test_a_block_that_is_not_a_name_and_an_arguments_object_is_unmappable():
    for body in (
        '{"arguments": {"queries": ["How?"], "name": "ExpertQAExtractor"}}',  # 6 extraction rows, once unescaped
        '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}',
        '{"name": "get_weather", "arguments": {}, "id": "call_1"}',
        '[{"name": "get_weather", "arguments": {}}]',
    ):
        with pytest.raises(hermes.Unmappable, match='a <tool_call> block that is not {"name": .*, "arguments": {.*}}'):
            hermes.split_calls(f"<tool_call>\n{body}\n</tool_call>")


def test_a_tag_without_its_pair_is_unmappable_rather_than_prose():
    for text in ("Sure.\n<tool_call>\n{}", "Sure.\n</tool_call>", PARIS + "\n<tool_call>"):
        with pytest.raises(hermes.Unmappable, match="a <tool_call> tag without its pair"):
            hermes.split_calls(text)


SUNNY = '<tool_response>\n{"name": "get_weather", "content": {"sky": "sunny", "temp": 18}}\n</tool_response>'
SNOW = '<tool_response>\n{"name": "get_weather", "content": {"sky": "snow", "note": "\\u2744"}}\n</tool_response>'


def test_a_tool_turn_is_the_json_of_each_response_block_in_order():
    assert hermes.responses(SUNNY + "\n" + SNOW + "\n") == [
        {"name": "get_weather", "content": {"sky": "sunny", "temp": 18}},
        {"name": "get_weather", "content": {"sky": "snow", "note": "❄"}},
    ]


def test_a_tool_turn_that_is_not_only_response_blocks_of_json_is_unmappable():
    for text, reason in [
        ("Result:\n" + SUNNY, "text outside the <tool_response> blocks"),
        (SUNNY + "\n<tool_response>\n{}", "a <tool_response> tag without its pair"),
        ("sunny", "a tool turn without a <tool_response> block"),
        ('<tool_response>\n{"temp": 18,}\n</tool_response>', "a <tool_response> block that is not JSON"),
    ]:
        with pytest.raises(hermes.Unmappable, match=reason):
            hermes.responses(text)


def row(*turns: tuple[str, str], tools: str = TOOLS, **extra) -> dict:
    return {
        "id": "4f1c2a",
        "conversations": [{"from": source, "value": value} for source, value in turns],
        "tools": tools,
        "category": "Weather",
        "subcategory": "Forecast",
        "task": "Check the weather",
        **extra,
    }


CONVERSATION = row(
    ("system", GLAIVE),
    ("human", "Weather in Paris and Zürich?"),
    ("gpt", PARIS + "\n" + ZURICH),
    ("tool", SUNNY + "\n" + SNOW),
    ("gpt", "Sunny in Paris, snow in Zürich."),
    ("human", "And tomorrow in Paris?"),
    ("gpt", PARIS + "\n"),
    ("tool", SUNNY + "\n"),
    ("gpt", "Sunny again."),
    ("human", "Thanks!"),
)


def call(city: str, id_: str | None = None) -> dict:
    """A get_weather call as a parse case expects it, or, given ``id_``, as the history holds it."""
    found = {"type": "function", "function": {"name": "get_weather", "arguments": f'{{"city": "{city}"}}'}}
    return found if id_ is None else {"id": id_, **found}


def test_a_conversation_gives_a_render_case_per_answered_user_turn_and_a_parse_case_per_assistant_turn():
    render, parse = hermes.row_cases(CONVERSATION, 7, "glaive_func_calling")
    user = {"role": "user", "content": "Weather in Paris and Zürich?"}
    # A parse case expects its calls as a parser returns them, without ids; the history gives each call the id
    # call_<n>, numbered across the conversation, for the tool message that answers it to name.
    both = {"content": "", "tool_calls": [call("Paris"), call("Zürich")]}
    made = {"role": "assistant", "content": "", "tool_calls": [call("Paris", "call_0"), call("Zürich", "call_1")]}
    results = [
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": '{"name": "get_weather", "content": {"sky": "sunny", "temp": 18}}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"name": "get_weather", "content": {"sky": "snow", "note": "❄"}}',
        },
    ]
    answer = {"content": "Sunny in Paris, snow in Zürich."}
    again = {"role": "user", "content": "And tomorrow in Paris?"}
    tomorrow = {"content": "", "tool_calls": [call("Paris")]}
    tomorrow_made = {"role": "assistant", "content": "", "tool_calls": [call("Paris", "call_2")]}
    result = {"role": "tool", "tool_call_id": "call_2", "content": results[0]["content"]}
    history = [user, made, *results, {"role": "assistant", **answer}, again]

    def request(messages):
        # No add_generation_prompt: the oracles add the generation prompt when a request does not say otherwise, and a
        # request that sets the default differs by its bytes from another importer's that does not.
        return {"messages": messages, "tools": [WEATHER]}

    def origin(turn, written=False):
        found = {
            "dataset": "hermes",
            "source": f"hf:datasets/NousResearch/hermes-function-calling-v1@{REVISION}",
            "sha256": "b98eb3f160359f27ad15018e974ce6db444f566eb5be4aa9e4aa690b34d50832",
            "file": "glaive-function-calling-5k.json",
            "row": 7,
            "row_id": "4f1c2a",
            "turn": turn,
            "license": "Apache-2.0",
        }
        # A case whose request holds a call holds ids bellwether wrote, and its origin says so.
        return {**found, "written": ["tool call ids"]} if written else found

    notes = "Hermes glaive_func_calling row 7 turn {}: Weather / Forecast"
    assert render == [
        {
            "name": "hermes-glaive-func-calling-7-1",
            "request": request([user]),
            "notes": notes.format(1),
            "origin": origin(1),
        },
        {
            "name": "hermes-glaive-func-calling-7-5",
            "request": request(history),
            "notes": notes.format(5),
            "origin": origin(5, written=True),
        },
    ]
    assert [(line["name"], line["request"]["messages"], line["message"]) for line in parse] == [
        ("hermes-glaive-func-calling-7-2", [user], both),
        ("hermes-glaive-func-calling-7-4", [user, made, *results], answer),
        ("hermes-glaive-func-calling-7-6", history, tomorrow),
        ("hermes-glaive-func-calling-7-8", [*history, tomorrow_made, result], {"content": "Sunny again."}),
    ]
    assert [line["origin"] for line in parse] == [origin(2), origin(4, True), origin(6, True), origin(8, True)]
    assert parse[0] == {
        "name": "hermes-glaive-func-calling-7-2",
        "request": request([user]),
        "message": both,
        "notes": notes.format(2),
        "origin": origin(2),
    }


def test_a_response_without_a_call_or_a_call_without_a_response_is_unmappable():
    ask = ("human", "Weather in Paris?")
    for turns, reason in [
        ([ask, ("tool", SUNNY), ("gpt", "Sunny.")], "a response without a call"),  # 10 func_calling rows
        ([ask, ("gpt", PARIS), ("tool", SUNNY + "\n" + SNOW), ("gpt", "Sunny.")], "a response without a call"),
        ([ask, ("gpt", PARIS + "\n" + ZURICH), ("tool", SUNNY), ("gpt", "Sunny.")], "a call without a response"),
        ([ask, ("gpt", PARIS), ask, ("gpt", "Sunny.")], "a call without a response"),
    ]:
        with pytest.raises(hermes.Unmappable, match=reason):
            hermes.row_cases(row(("system", GLAIVE), *turns), 0, "func_calling")
    # A row may end on its calls, as every func_calling_singleturn row does: no later request holds them.
    render, parse = hermes.row_cases(row(("system", FUNCTION_CALLING), ask, ("gpt", PARIS)), 0, "func_calling")
    assert len(render) == len(parse) == 1


def test_a_call_to_a_function_the_row_does_not_declare_is_unmappable():
    movie = '<tool_call>\n{"name": "get_movie_details", "arguments": {"title": "The Holiday"}}\n</tool_call>'
    with pytest.raises(hermes.Unmappable, match="a call to get_movie_details, which the row's tools do not declare"):
        turns = [("system", GLAIVE), ("human", "Tell me about The Holiday."), ("gpt", movie)]
        hermes.row_cases(row(*turns), 0, "glaive_func_calling")


def test_a_tool_that_is_not_an_openai_function_tool_is_unmappable():
    joke = {"name": "get_random_joke", "description": "Get a random joke", "parameters": None}  # glaive row 1288
    tools = json.dumps([WEATHER, joke])
    turns = [("system", GLAIVE.replace(TOOLS, tools)), ("human", "A joke?"), ("gpt", "Why did the cloud stay home?")]
    with pytest.raises(hermes.Unmappable, match="a tool that is not an OpenAI function tool"):
        hermes.row_cases(row(*turns, tools=tools), 0, "glaive_func_calling")


def test_a_row_whose_tools_declare_one_name_twice_is_unmappable():
    # 44 glaive_func_calling rows taken do, 42 of them with two different definitions: a call to that name could be
    # held to either, and no engine is asked to choose.
    fahrenheit = {**WEATHER, "function": {**WEATHER["function"], "description": "Weather in a city, in Fahrenheit."}}
    for tools in (json.dumps([WEATHER, WEATHER]), json.dumps([WEATHER, fahrenheit])):
        turns = [("system", GLAIVE.replace(TOOLS, tools)), ("human", "Weather in Paris?"), ("gpt", PARIS)]
        with pytest.raises(hermes.Unmappable, match="two tools under one name"):
            hermes.row_cases(row(*turns, tools=tools), 0, "glaive_func_calling")


def test_a_row_without_tools_sends_none_and_keeps_its_system_message():
    plain = "You are a helpful assistant, with no access to external functions."  # 865 glaive_func_calling rows
    turns = [("system", plain), ("human", "Hi"), ("gpt", "Hello!")]
    render, parse = hermes.row_cases(row(*turns, tools="null"), 3, "glaive_func_calling")
    messages = [{"role": "system", "content": plain}, {"role": "user", "content": "Hi"}]
    assert render[0]["request"] == {"messages": messages}
    assert parse[0]["request"] == render[0]["request"] and parse[0]["message"] == {"content": "Hello!"}


def test_a_turn_from_an_unknown_speaker_is_unmappable_rather_than_dropped():
    turns = [("system", GLAIVE), ("human", "Hi"), ("observation", "noted"), ("gpt", "Hello!")]
    with pytest.raises(hermes.Unmappable, match="a turn from 'observation'"):
        hermes.row_cases(row(*turns), 0, "glaive_func_calling")


def ask(city: str) -> dict:
    call = PARIS.replace("Paris", city) + "\n"
    return row(("system", FUNCTION_CALLING), ("human", f"Weather in {city}?"), ("gpt", call))


ORPHAN = row(("system", FUNCTION_CALLING), ("human", "Weather in Paris?"), ("tool", SUNNY), ("gpt", "Sunny."))


def test_sets_take_every_stride_row_from_the_configs_first_row_and_name_each_row_they_skip():
    # func_calling's rows begin as func_calling_singleturn's of the same index, so it takes the odd rows.
    assert hermes.STRIDE == 2
    skipped: list = []
    rows = {
        "func_calling_singleturn": [ask("Paris"), ORPHAN, ORPHAN, ask("Oslo"), ask("Rome")],
        "func_calling": [ORPHAN, ask("Lima"), ask("Quito"), ORPHAN],
    }
    sets = hermes.build_sets(rows, stride=2, skipped=skipped)
    names = ["hermes-func-calling-singleturn-0-1", "hermes-func-calling-singleturn-4-1"]
    assert [line["name"] for line in sets[("render", "hermes-func-calling-singleturn")]] == names
    assert [line["name"] for line in sets[("parse", "hermes-func-calling-singleturn")]] == [n[:-1] + "2" for n in names]
    assert [line["name"] for line in sets[("render", "hermes-func-calling")]] == ["hermes-func-calling-1-1"]
    assert [line["name"] for line in sets[("parse", "hermes-func-calling")]] == ["hermes-func-calling-1-2"]
    assert skipped == [
        ("func_calling_singleturn", 2, "a response without a call"),
        ("func_calling", 3, "a response without a call"),
    ]


def test_a_case_that_repeats_an_earlier_one_of_its_kind_is_left_out_and_named_with_the_case_it_repeats():
    def case(name: str, user: str, answer: str | None = None) -> dict:
        line = {"name": name, "request": {"messages": [{"role": "user", "content": user}]}}
        return line if answer is None else {**line, "message": {"content": answer}}

    sets = {
        ("render", "hermes-a"): [case("a-1", "Hi"), case("a-3", "Hi")],
        ("parse", "hermes-a"): [case("a-2", "Hi", "Hello!"), case("a-4", "Hi", "Hey!")],
        ("render", "hermes-b"): [case("b-1", "Hi"), case("b-3", "Bye")],
        ("parse", "hermes-b"): [case("b-2", "Hi", "Hello!")],
    }
    repeated: list = []
    kept = hermes.leave_out_repeats(sets, repeated)
    # A render and a parse case may share a request, and two parse cases a request with different messages. A
    # repeat is left out in any set, and a set left with no case is dropped.
    assert {key: [line["name"] for line in lines] for key, lines in kept.items()} == {
        ("render", "hermes-a"): ["a-1"],
        ("parse", "hermes-a"): ["a-2", "a-4"],
        ("render", "hermes-b"): ["b-3"],
    }
    assert repeated == [("a-3", "a-1"), ("b-1", "a-1"), ("b-2", "a-2")]


def test_rows_that_open_with_the_same_turns_give_the_cases_of_those_turns_once():
    # Six pairs of the glaive_func_calling rows taken open with the same turns and part later, as rows 48 and 622 do.
    hello = row(("system", GLAIVE), ("human", "Hi"), ("gpt", "Hello!"))
    hey = row(("system", GLAIVE), ("human", "Hi"), ("gpt", "Hey there!"))
    repeated: list = []
    sets = hermes.build_sets({"glaive_func_calling": [hello, hey, hello]}, stride=1, repeated=repeated)
    name = "hermes-glaive-func-calling-{}".format
    assert [line["name"] for line in sets[("render", "hermes-glaive-func-calling")]] == [name("0-1")]
    assert [line["name"] for line in sets[("parse", "hermes-glaive-func-calling")]] == [name("0-2"), name("1-2")]
    assert repeated == [(name("1-1"), name("0-1")), (name("2-1"), name("0-1")), (name("2-2"), name("0-2"))]
    # Calls carry the same ids in both rows, so the cases at and after them repeat too.
    opening = [(turn["from"], turn["value"]) for turn in CONVERSATION["conversations"][:5]]
    rome = row(*opening, ("human", "And in Rome?"), ("gpt", "Rain."))
    repeated = []
    sets = hermes.build_sets({"glaive_func_calling": [CONVERSATION, rome]}, stride=1, repeated=repeated)
    assert repeated == [(name("1-1"), name("0-1")), (name("1-2"), name("0-2")), (name("1-4"), name("0-4"))]
    assert [line["name"] for line in sets[("render", "hermes-glaive-func-calling")]][-1] == name("1-5")


def test_written_sets_check_clean_and_a_changed_or_stale_hermes_file_is_reported(tmp_path):
    sets, corpus = hermes.build_sets({"glaive_func_calling": [CONVERSATION]}, stride=1), tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    for name in ("common", "bfcl-simple-python", "hermes-old"):
        (corpus / "render" / f"{name}.jsonl").write_text("{}\n")
    hermes.write_sets(sets, corpus)
    assert not (corpus / "render" / "hermes-old.jsonl").exists()
    for name in ("common", "bfcl-simple-python"):
        assert (corpus / "render" / f"{name}.jsonl").read_text() == "{}\n"
    text = (corpus / "parse" / "hermes-glaive-func-calling.jsonl").read_bytes().decode("utf-8")
    assert "Zürich" in text and text.endswith("\n") and len(text.splitlines()) == 4
    assert hermes.check_sets(sets, corpus) == []
    (corpus / "parse" / "hermes-glaive-func-calling.jsonl").write_text("{}\n")
    (corpus / "render" / "hermes-stale.jsonl").write_text("{}\n")
    assert hermes.check_sets(sets, corpus) == [
        f"{corpus / 'parse' / 'hermes-glaive-func-calling.jsonl'}: differs from a fresh import",
        f"{corpus / 'render' / 'hermes-stale.jsonl'}: no Hermes config writes it",
    ]


def serve(tmp_path, monkeypatch, card: str = CARD, **rows: list[dict]) -> list[tuple]:
    """``hf_hub_download`` served from files written here, each pinned by its sha256 in place of the dataset's: the
    card, and each config's rows (no config given: none). Returns the downloads asked for, as (repo, file, options).
    """
    paths = {hf.CARD: tmp_path / "hub" / hf.CARD}
    paths[hf.CARD].parent.mkdir(parents=True)
    paths[hf.CARD].write_text(card, encoding="utf-8")
    monkeypatch.setattr(hermes, "CARD_SHA256", hashlib.sha256(paths[hf.CARD].read_bytes()).hexdigest())
    for config, (filename, _) in list(hermes.CONFIGS.items()):
        paths[filename] = tmp_path / "hub" / filename
        paths[filename].write_text(json.dumps(rows.get(config, [])), encoding="utf-8")
        sha256 = hashlib.sha256(paths[filename].read_bytes()).hexdigest()
        monkeypatch.setitem(hermes.CONFIGS, config, (filename, sha256))
    calls: list[tuple] = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(paths[filename])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def test_the_command_writes_names_every_row_it_skips_and_every_case_it_leaves_out_then_checks(
    tmp_path, monkeypatch, capsys
):
    hello = row(("system", GLAIVE), ("human", "Hi"), ("gpt", "Hello!"))
    hey = row(("system", GLAIVE), ("human", "Hi"), ("gpt", "Hey there!"))
    calls = serve(
        tmp_path,
        monkeypatch,
        func_calling_singleturn=[ask("Paris"), ORPHAN, ORPHAN, ask("Oslo"), ask("Rome"), ORPHAN, ORPHAN],
        func_calling=[ORPHAN, ask("Lima"), ask("Quito")],
        glaive_func_calling=[hello, hello, hey],
    )
    corpus = tmp_path / "corpus"
    assert main(["import", "hermes", "--corpus", str(corpus), "--check"]) == 1
    assert main(["import", "hermes", "--corpus", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'render' / 'hermes-func-calling-singleturn.jsonl'}: 2 cases" in out
    assert f"{corpus / 'render' / 'hermes-func-calling.jsonl'}: 1 cases" in out
    assert f"{corpus / 'parse' / 'hermes-glaive-func-calling.jsonl'}: 2 cases" in out
    assert "func_calling_singleturn: 4 of 7 rows, from row 0 in steps of 2" in out
    assert "func_calling: 1 of 3 rows, from row 1 in steps of 2" in out
    assert "no cases for 2 row(s) of func_calling_singleturn (2, 6): a response without a call" in out
    assert "no case hermes-glaive-func-calling-2-1: it repeats hermes-glaive-func-calling-0-1" in out
    assert main(["import", "hermes", "--corpus", str(corpus), "--check"]) == 0
    assert f"{corpus}: the Hermes sets equal a fresh import of hf:datasets/" in capsys.readouterr().out
    # Every file comes from the dataset at the pinned commit.
    assert {(repo, options["revision"]) for repo, _, options in calls} == {
        ("NousResearch/hermes-function-calling-v1", REVISION)
    }


def test_the_command_checks_the_cards_license_before_it_reads_any_row_and_writes_nothing_on_a_refusal(
    tmp_path, monkeypatch
):
    calls = serve(tmp_path, monkeypatch, card=CARD.replace("apache-2.0", "mit"), func_calling=[ask("Paris")])
    corpus = tmp_path / "corpus"
    refusal = "the card's license is 'mit', not the reviewed 'apache-2.0'; review it before importing"
    with pytest.raises(ValueError, match=f"NousResearch/hermes-function-calling-v1 README.md: {refusal}"):
        main(["import", "hermes", "--corpus", str(corpus)])
    assert [filename for _, filename, _ in calls] == ["README.md"] and not corpus.exists()


def real_rows(config: str) -> list[dict]:
    """The pinned file of ``config`` from the Hugging Face cache; skipped when it is not there (no download)."""
    filename, sha256 = hermes.CONFIGS[config]
    cached = huggingface_hub.try_to_load_from_cache(
        hermes.REPO, filename, revision=hermes.REVISION, repo_type="dataset"
    )
    if not isinstance(cached, str):
        pytest.skip(f"{filename} is not in the Hugging Face cache; `bellwether import hermes` downloads it")
    return hermes.read_rows(hf.fetch(hermes.REPO, hermes.REVISION, filename, sha256))


def system_outcome(row: dict) -> str:
    tools = row["tools"] if json.loads(row["tools"]) else None
    try:
        return hermes.system_message(row["conversations"][0]["value"], tools)
    except hermes.Unmappable as err:
        return f"refused: {err}"


def test_on_the_real_rows_every_tool_prompt_comes_out_whole_and_only_the_rows_without_tools_keep_one():
    refused = "refused: the system message lists tools the row does not declare"
    for config in ("func_calling_singleturn", "func_calling"):
        assert Counter(system_outcome(row) for row in real_rows(config)) == {"": 1832, refused: 61}
    assert Counter(system_outcome(row) for row in real_rows("glaive_func_calling")) == {
        "": 4344,
        "You are a helpful assistant, with no access to external functions.": 865,
    }
