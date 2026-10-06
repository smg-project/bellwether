import hashlib
import json
import re
from types import SimpleNamespace

import huggingface_hub
import pytest

from bellwether.cli import main
from bellwether.importers import github, glaive_v2, hf

LEAD_IN = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -"

# Row 2144's system field: two functions, each pretty-printed, after the lead-in sentence.
SYSTEM_2144 = (
    LEAD_IN
    + '\n{\n    "name": "get_random_joke",\n    "description": "Get a random joke",\n    "parameters": {}\n}\n\n'
    + '{\n    "name": "get_random_fact",\n    "description": "Get a random fact",\n    "parameters": {}\n}\n\n'
)


def test_the_function_list_becomes_tools_and_leaves_no_system_message():
    system, tools = glaive_v2.system_and_tools(SYSTEM_2144)
    assert system is None
    assert tools == [
        {
            "type": "function",
            "function": {"name": "get_random_joke", "description": "Get a random joke", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "get_random_fact", "description": "Get a random fact", "parameters": {}},
        },
    ]
    assert [list(tool["function"]) for tool in tools] == [["name", "description", "parameters"]] * 2


def test_a_system_prompt_without_functions_stays_as_the_system_message():
    # The system field of every row without functions (34,598 at the pinned revision).
    system = "SYSTEM: You are a helpful assistant, with no access to external functions.\n\n"
    assert glaive_v2.system_and_tools(system) == (
        "You are a helpful assistant, with no access to external functions.",
        [],
    )


def test_text_after_the_function_list_stays_as_the_system_message():
    # No row at the pinned revision has any; the rule keeps every word that is neither the lead-in nor a function.
    system = LEAD_IN + '\n{\n    "name": "ping",\n    "description": "Pings.",\n    "parameters": {}\n}\n\nBe brief.\n'
    ping = {"type": "function", "function": {"name": "ping", "description": "Pings.", "parameters": {}}}
    assert glaive_v2.system_and_tools(system) == ("Be brief.", [ping])


def test_quoted_arguments_become_one_line_of_json_with_raw_unicode():
    # The text between <functioncall> and <|endoftext|> in rows 25475, 61004 and 19583.
    assert glaive_v2.parse_call(
        """ {"name": "detect_language", "arguments": '{"text": "Je suis un étudiant"}'} """
    ) == (
        "detect_language",
        '{"text": "Je suis un étudiant"}',
    )
    pretty = """ {"name": "get_lyrics", "arguments": '{\n  "title": "Shape of You",\n  "artist": "Ed Sheeran"\n}'} """
    assert glaive_v2.parse_call(pretty) == ("get_lyrics", '{"title": "Shape of You", "artist": "Ed Sheeran"}')
    apostrophe = (
        """ {"name": "send_sms", "arguments": '{"phone_number": "1234567890", "message": "Hey, let's meet"}'} """
    )
    assert glaive_v2.parse_call(apostrophe) == (
        "send_sms",
        '{"phone_number": "1234567890", "message": "Hey, let\'s meet"}',
    )


def test_arguments_written_as_an_object_are_taken_as_they_are():
    # Rows 111754 and 128: the call is valid JSON, its arguments an object rather than a quoted string.
    assert glaive_v2.parse_call(' {"name": "generate_random_id", "arguments": {}} ') == ("generate_random_id", "{}")
    bmi = ' {"name": "calculate_bmi", "arguments": {\n  "weight": 70,\n  "height": 1.75\n}} '
    assert glaive_v2.parse_call(bmi) == ("calculate_bmi", '{"weight": 70, "height": 1.75}')


@pytest.mark.parametrize(
    "text, reason",
    [
        # row 71039: \' is not an escape JSON knows
        (""" {"name": "analyze_sentiment", "arguments": '{"text": "life\\'s full"}'} """, "ARGUMENTS_NOT_JSON"),
        # row 26684, shortened: Infinity is not JSON, though Python's json reads it
        (""" {"name": "calculate_tax", "arguments": '{"range": [70001, Infinity]}'} """, "ARGUMENTS_NOT_JSON"),
        # row 65421
        (""" {"name": "generate_report", "arguments": '{"data": [User provided data]}'} """, "ARGUMENTS_NOT_JSON"),
        # rows 81297 and 81998
        (""" {"name": "calculate_average", "arguments": '[5, 10, 15, 20, 25]'} """, "ARGUMENTS_NOT_OBJECT"),
        (' {"name": "get_current_time", "arguments": null} ', "ARGUMENTS_NOT_OBJECT"),
        # row 81082: parameters, not arguments
        (' {"name": "get_random_quote", "parameters": {}} ', "CALL_FORM"),
        # row 34221: the call stops inside its arguments
        (""" {"name": "get_song_lyrics", "arguments": '{"artist": "Ed Sheeran", "title": " """, "CALL_FORM"),
        # row 57572, shortened: one brace too many
        (""" {"name": "send_email", "arguments": '{"subject": "Agenda"}'}} """, "CALL_FORM"),
    ],
)
def test_a_call_that_cannot_be_read_is_refused_with_its_reason(text, reason):
    with pytest.raises(glaive_v2.Unmappable) as refused:
        glaive_v2.parse_call(text)
    assert str(refused.value) == getattr(glaive_v2, reason)


ROW_107105 = {
    "system": "SYSTEM: You are a helpful assistant, with no access to external functions.\n\n",
    "chat": "USER: Assign this occupation to the appropriate category\nNurse\n\nASSISTANT: Healthcare <|endoftext|>\n\n"
    "USER: What about an electrician?\n\nASSISTANT: Skilled Trades <|endoftext|>",
}


MESSAGES_107105 = [
    {"role": "system", "content": "You are a helpful assistant, with no access to external functions."},
    {"role": "user", "content": "Assign this occupation to the appropriate category\nNurse"},
    {"role": "assistant", "content": "Healthcare"},
    {"role": "user", "content": "What about an electrician?"},
    {"role": "assistant", "content": "Skilled Trades"},
]


def test_a_row_without_functions_becomes_its_system_prompt_and_turns():
    assert glaive_v2.messages_for(ROW_107105) == (MESSAGES_107105, [])


# Row 25475 exactly as the file holds it.
ROW_25475 = json.loads(
    r"""{"system": "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{\n    \"name\": \"detect_language\",\n    \"description\": \"Detect the language of a given text\",\n    \"parameters\": {\n        \"type\": \"object\",\n        \"properties\": {\n            \"text\": {\n                \"type\": \"string\",\n                \"description\": \"The text to detect the language of\"\n            }\n        },\n        \"required\": [\n            \"text\"\n        ]\n    }\n}\n", "chat": "USER: I have a text here and I am not sure what language it is. Can you help me identify it?\n\n\nASSISTANT: Of course, I can help with that. Please provide me with the text. <|endoftext|>\n\n\nUSER: Here it is - \"Je suis un étudiant\"\n\n\nASSISTANT: <functioncall> {\"name\": \"detect_language\", \"arguments\": '{\"text\": \"Je suis un étudiant\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {\"language\": \"French\"}\n\n\nASSISTANT: The text you provided is in French. <|endoftext|>\n\n\n"}"""  # noqa: E501
)
DETECT_LANGUAGE = {
    "type": "function",
    "function": {
        "name": "detect_language",
        "description": "Detect the language of a given text",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to detect the language of"}},
            "required": ["text"],
        },
    },
}
CALL_25475 = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "detect_language", "arguments": '{"text": "Je suis un étudiant"}'},
}
# The call as a parse case expects it: a parser makes up its own id, so the expected call holds none.
EXPECTED_25475 = {"type": "function", "function": CALL_25475["function"]}
MESSAGES_25475 = [
    {
        "role": "user",
        "content": "I have a text here and I am not sure what language it is. Can you help me identify it?",
    },
    {"role": "assistant", "content": "Of course, I can help with that. Please provide me with the text."},
    {"role": "user", "content": 'Here it is - "Je suis un étudiant"'},
    {"role": "assistant", "content": "", "tool_calls": [CALL_25475]},
    {"role": "tool", "tool_call_id": "call_0", "content": '{"language": "French"}'},
    {"role": "assistant", "content": "The text you provided is in French."},
]


def test_a_call_gets_an_id_and_the_function_response_answers_it():
    messages, tools = glaive_v2.messages_for(ROW_25475)
    assert tools == [DETECT_LANGUAGE]
    assert messages == MESSAGES_25475
    assert list(messages[3]["tool_calls"][0]) == ["id", "type", "function"]
    assert list(messages[4]) == ["role", "tool_call_id", "content"]


def news(country: str) -> str:
    """An assistant turn calling get_news_headlines as row 1 does."""
    arguments = f"""'{{"country": "{country}"}}'"""
    return f'ASSISTANT: <functioncall> {{"name": "get_news_headlines", "arguments": {arguments}}} <|endoftext|>'


NEWS_SYSTEM = (
    LEAD_IN + '\n{\n    "name": "get_news_headlines",\n    "description": "Get news",\n    "parameters": {}\n}\n'
)


def test_call_ids_count_the_calls_of_a_row_and_each_response_answers_the_call_before_it():
    # Row 1's shape, shortened: two rounds of a call, its response and prose.
    chat = "\n\n\n".join(
        [
            "USER: News for the US?",
            news("United States"),
            'FUNCTION RESPONSE: {"headlines": ["A"]}',
            "ASSISTANT: A. <|endoftext|>",
            "USER: And France?",
            news("France"),
            'FUNCTION RESPONSE: {"headlines": ["B"]}',
            "ASSISTANT: B. <|endoftext|>",
        ]
    )
    messages, _ = glaive_v2.messages_for({"system": NEWS_SYSTEM, "chat": chat + "\n\n\n"})
    ids = [m["tool_calls"][0]["id"] if "tool_calls" in m else m.get("tool_call_id") for m in messages]
    assert ids == [None, "call_0", "call_0", None, None, "call_1", "call_1", None]


def test_the_same_chat_in_another_row_gives_the_same_requests_and_messages():
    # The call ids leave the row out, so a chat that recurs in another row repeats its cases there; only the case
    # names, notes and origins tell the two rows apart.
    def cases(index: int) -> list[dict]:
        render, parse = glaive_v2.cases_for(index, *glaive_v2.messages_for(ROW_25475))
        return [{key: line[key] for key in ("request", "message") if key in line} for line in render + parse]

    assert cases(31) == cases(25475)
    _, parse = glaive_v2.cases_for(31, *glaive_v2.messages_for(ROW_25475))
    assert [line["name"] for line in parse] == ["glaive-v2-31-1", "glaive-v2-31-3", "glaive-v2-31-5"]
    assert [line["origin"]["row"] for line in parse] == [31, 31, 31]


def one_function(name: str) -> str:
    return LEAD_IN + f'\n{{\n    "name": "{name}",\n    "description": "Does it.",\n    "parameters": {{}}\n}}\n'


@pytest.mark.parametrize(
    "row, reason",
    [
        # row 17357: the assistant calls in another syntax, so the function response follows no call
        (
            {
                "system": one_function("get_song_lyrics"),
                "chat": 'USER: Hi, I need the lyrics of the song "Imagine" by John Lennon.\n\n\nASSISTANT: Sure, let me'
                ' fetch that for you.\nAI to= get_song_lyrics: {"song_title": "Imagine", "artist": "John Lennon"}'
                ' <|endoftext|>\n\n\nFUNCTION RESPONSE: {"lyrics": "',
            },
            "RESPONSE_WITHOUT_CALL",
        ),
        # row 21816: words before the call
        (
            {
                "system": one_function("search_products"),
                "chat": "USER: I am looking for a new laptop.\n\n\nASSISTANT: Of course, I can help you with that."
                """ Let's start the search.\n<functioncall> {"name": "search_products", "arguments":"""
                """ '{"keyword": "laptop"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {"products": []}\n\n\n"""
                "ASSISTANT: None found. <|endoftext|>",
            },
            "TEXT_BEFORE_CALL",
        ),
        # row 2289: "AI:" in the source became "<|endoftext|>\n\n\nASSISTANT:" inside one turn
        (
            {
                "system": one_function("search_news"),
                "chat": "USER: News on AI?\n\n\nASSISTANT: Here are some articles on recent developments in"
                " <|endoftext|>\n\n\nASSISTANT:\n1. [AI and Machine Learning](https://www.example.com/ai-ml-2021)"
                " <|endoftext|>\n\n\n",
            },
            "ASSISTANT_END",
        ),
        # row 9046: the same damage inside a function response
        (
            {
                "system": one_function("search_news"),
                "chat": "USER: News on AI?\n\n\n"
                + news("AI").replace("get_news_headlines", "search_news")
                + '\n\n\nFUNCTION RESPONSE: {"articles": [{"title": "The Next Big Thing in \nASSISTANT: Top Trends"}]}'
                " <|endoftext|>\n\n\nASSISTANT: Found one. <|endoftext|>\n\n\n",
            },
            "STRAY_END",
        ),
        # row 44369: and inside a user turn
        (
            {
                "system": one_function("add_to_shopping_cart"),
                "chat": 'USER: Hey, I need to add a book to my shopping cart. The book is "\nASSISTANT: A Modern'
                ' Approach", I need 2 copies and each costs $30. <|endoftext|>\n\n\n'
                "ASSISTANT: Sure. <|endoftext|>\n\n\n",
            },
            "STRAY_END",
        ),
        # row 3871: the call's name differs from the declared create_to-do_list
        (
            {
                "system": one_function("create_to-do_list"),
                "chat": 'USER: The title should be "Weekend Chores".\n\n\nASSISTANT: <functioncall> {"name":'
                """ "create_to_do_list", "arguments": '{"title": "Weekend Chores"}'} <|endoftext|>\n\n\n"""
                'FUNCTION RESPONSE: {"status": "success"}\n\n\nASSISTANT: Done. <|endoftext|>\n\n\n',
            },
            "UNDECLARED",
        ),
        # row 168, shortened: convert_currency is declared twice, with two definitions, so a call to it could be held
        # to either; 351 rows of the file declare a name twice, 338 of them with two different definitions
        (
            {
                "system": LEAD_IN
                + '\n{\n    "name": "convert_currency",\n    "description": "Convert one currency to another",\n'
                + '    "parameters": {}\n}\n\n'
                + '{\n    "name": "convert_currency",\n    "description": "Convert currency",\n'
                + '    "parameters": {"type": "object"}\n}\n',
                "chat": 'USER: Convert 500 USD to EUR.\n\n\nASSISTANT: <functioncall> {"name": "convert_currency",'
                """ "arguments": '{"amount": 500}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {"result": 425.5}\n\n\n"""
                "ASSISTANT: 425.50 EUR. <|endoftext|>\n\n\n",
            },
            "NAME_TWICE",
        ),
        # the same with one definition written twice (13 rows of the file): the name rule leaves it out too
        (
            {
                "system": one_function("get_random_joke") + one_function("get_random_joke").removeprefix(LEAD_IN),
                "chat": "USER: A joke?\n\nASSISTANT: Not now. <|endoftext|>",
            },
            "NAME_TWICE",
        ),
    ],
)
def test_a_row_that_cannot_be_mapped_is_refused_with_its_reason(row, reason):
    with pytest.raises(glaive_v2.Unmappable) as refused:
        glaive_v2.messages_for(row)
    assert str(refused.value) == getattr(glaive_v2, reason)


def origin(row: int, turn: int, written: bool = False) -> dict:
    found = {
        "dataset": "glaive-v2",
        "source": "hf:datasets/glaiveai/glaive-function-calling-v2@e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
        "sha256": "e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad",
        "file": "glaive-function-calling-v2.json",
        "row": row,
        "turn": turn,
        "license": "Apache-2.0",
    }
    # A case whose request holds a call holds ids bellwether wrote, and its origin says so.
    return {**found, "written": ["tool call ids"]} if written else found


def test_each_assistant_turn_is_a_parse_case_and_each_answered_user_turn_a_render_case():
    render, parse = glaive_v2.cases_for(25475, MESSAGES_25475, [DETECT_LANGUAGE])
    notes = "glaive-function-calling-v2 row 25475 turn {}".format
    assert render == [
        {
            "name": "glaive-v2-25475-0",
            "request": {"messages": MESSAGES_25475[:1], "tools": [DETECT_LANGUAGE]},
            "notes": notes(0),
            "origin": origin(25475, 0),
        },
        {
            "name": "glaive-v2-25475-2",
            "request": {"messages": MESSAGES_25475[:3], "tools": [DETECT_LANGUAGE]},
            "notes": notes(2),
            "origin": origin(25475, 2),
        },
    ]
    assert parse == [
        {
            "name": "glaive-v2-25475-1",
            "request": {"messages": MESSAGES_25475[:1], "tools": [DETECT_LANGUAGE]},
            "message": {"content": "Of course, I can help with that. Please provide me with the text."},
            "notes": notes(1),
            "origin": origin(25475, 1),
        },
        {
            "name": "glaive-v2-25475-3",
            "request": {"messages": MESSAGES_25475[:3], "tools": [DETECT_LANGUAGE]},
            "message": {"content": "", "tool_calls": [EXPECTED_25475]},
            "notes": notes(3),
            "origin": origin(25475, 3),
        },
        {
            "name": "glaive-v2-25475-5",
            "request": {"messages": MESSAGES_25475[:5], "tools": [DETECT_LANGUAGE]},
            "message": {"content": "The text you provided is in French."},
            "notes": notes(5),
            "origin": origin(25475, 5, written=True),
        },
    ]
    assert [list(line) for line in render] == [["name", "request", "notes", "origin"]] * 2
    assert [list(line) for line in parse] == [["name", "request", "message", "notes", "origin"]] * 3
    assert list(render[0]["origin"]) == ["dataset", "source", "sha256", "file", "row", "turn", "license"]
    assert list(parse[2]["origin"]) == ["dataset", "source", "sha256", "file", "row", "turn", "license", "written"]
    assert list(parse[1]["message"]["tool_calls"][0]) == ["type", "function"]


def test_a_case_whose_request_holds_a_call_says_its_ids_are_written_and_expects_calls_without_ids():
    # Row 1's shape: the second round's cases carry the first round's call in their history, with the id call_0 that
    # bellwether wrote; the second call is expected without an id, and holds call_1 once it is history.
    chat = "\n\n\n".join(
        [
            "USER: News for the US?",
            news("United States"),
            'FUNCTION RESPONSE: {"headlines": ["A"]}',
            "ASSISTANT: A. <|endoftext|>",
            "USER: And France?",
            news("France"),
            'FUNCTION RESPONSE: {"headlines": ["B"]}',
            "ASSISTANT: B. <|endoftext|>",
        ]
    )
    render, parse = glaive_v2.cases_for(1, *glaive_v2.messages_for({"system": NEWS_SYSTEM, "chat": chat}))
    assert [(line["name"], "written" in line["origin"]) for line in render] == [
        ("glaive-v2-1-0", False),
        ("glaive-v2-1-4", True),
    ]
    assert [(line["name"], line["origin"].get("written")) for line in parse] == [
        ("glaive-v2-1-1", None),
        ("glaive-v2-1-3", ["tool call ids"]),
        ("glaive-v2-1-5", ["tool call ids"]),
        ("glaive-v2-1-7", ["tool call ids"]),
    ]
    france = {"name": "get_news_headlines", "arguments": '{"country": "France"}'}
    assert parse[2]["message"] == {"content": "", "tool_calls": [{"type": "function", "function": france}]}
    history = parse[3]["request"]["messages"]
    assert [m["tool_calls"][0]["id"] for m in history if "tool_calls" in m] == ["call_0", "call_1"]
    assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == ["call_0", "call_1"]


def test_a_row_without_functions_sends_no_tools_and_counts_turns_after_its_system_message():
    render, parse = glaive_v2.cases_for(107105, MESSAGES_107105, [])
    assert [(line["name"], line["request"]) for line in render] == [
        ("glaive-v2-107105-0", {"messages": MESSAGES_107105[:2]}),
        ("glaive-v2-107105-2", {"messages": MESSAGES_107105[:4]}),
    ]
    assert [(line["name"], line["request"], line["message"]) for line in parse] == [
        ("glaive-v2-107105-1", {"messages": MESSAGES_107105[:2]}, {"content": "Healthcare"}),
        ("glaive-v2-107105-3", {"messages": MESSAGES_107105[:4]}, {"content": "Skilled Trades"}),
    ]


def test_a_user_turn_no_assistant_answers_is_no_render_case():
    # Row 40468 has a user turn followed by another; 1,599 rows end on a user turn.
    messages = [
        {"role": "user", "content": "Thank you. I will start now.\n\n[After a while]"},
        {"role": "user", "content": "I am done with the recording. Can you stop it now?"},
        {"role": "assistant", "content": "Stopped."},
        {"role": "user", "content": "Thank you."},
    ]
    render, parse = glaive_v2.cases_for(40468, messages, [])
    assert [line["name"] for line in render] == ["glaive-v2-40468-1"]
    assert [line["name"] for line in parse] == ["glaive-v2-40468-2"]


NO_FUNCTIONS = "SYSTEM: You are a helpful assistant, with no access to external functions.\n\n"


def chat_row(*rounds: str) -> dict:
    """A row without functions: per round, a user turn and the assistant's answer."""
    turns = [turn for text in rounds for turn in (f"USER: {text}?", f"ASSISTANT: {text}. <|endoftext|>")]
    return {"system": NO_FUNCTIONS, "chat": "\n\n".join(turns)}


BROKEN = {"system": NO_FUNCTIONS, "chat": "USER: Hi <|endoftext|>\n\nASSISTANT: Hello. <|endoftext|>"}


def names(sets: dict) -> dict:
    return {key: [line["name"] for line in lines] for key, lines in sets.items()}


def test_the_sample_is_every_kth_row_and_refused_rows_are_counted_across_the_whole_file():
    rows = [chat_row("a", "b"), BROKEN, chat_row("c"), chat_row("d"), BROKEN, chat_row("e"), chat_row("f")]
    skipped: list[tuple[int, str]] = []
    sets = glaive_v2.build_sets(rows, step=3, set_size=100, skipped=skipped)
    assert names(sets) == {
        ("render", "glaive-v2-00"): ["glaive-v2-0-0", "glaive-v2-0-2", "glaive-v2-3-0", "glaive-v2-6-0"],
        ("parse", "glaive-v2-00"): ["glaive-v2-0-1", "glaive-v2-0-3", "glaive-v2-3-1", "glaive-v2-6-1"],
    }
    assert skipped == [(1, glaive_v2.STRAY_END), (4, glaive_v2.STRAY_END)]


def test_a_set_holds_whole_rows_and_no_more_cases_of_either_kind_than_the_set_size():
    # Row 111754's shape: two assistant turns answer one user turn, so the row has one render and two parse cases.
    two_answers = {
        "system": NO_FUNCTIONS,
        "chat": "USER: g?\n\nASSISTANT: g. <|endoftext|>\n\nASSISTANT: h. <|endoftext|>",
    }
    sets = glaive_v2.build_sets([chat_row("a", "b"), two_answers, chat_row("c")], step=1, set_size=3)
    assert names(sets) == {
        ("render", "glaive-v2-00"): ["glaive-v2-0-0", "glaive-v2-0-2"],
        ("parse", "glaive-v2-00"): ["glaive-v2-0-1", "glaive-v2-0-3"],
        ("render", "glaive-v2-01"): ["glaive-v2-1-0", "glaive-v2-2-0"],
        ("parse", "glaive-v2-01"): ["glaive-v2-1-1", "glaive-v2-1-2", "glaive-v2-2-1"],
    }
    assert glaive_v2.build_sets([BROKEN], step=1, set_size=3) == {}


# The opening of the Apache License 2.0 as the Apache Software Foundation publishes it.
APACHE = b"\n                                 Apache License\n                           Version 2.0, January 2004\n"


def test_written_sets_check_clean_and_a_changed_missing_or_stale_file_is_reported(tmp_path):
    sets, corpus = glaive_v2.build_sets([ROW_25475], step=1, set_size=10), tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    for name in ("common", "bfcl-simple-python", "glaive-v2-07"):
        (corpus / "render" / f"{name}.jsonl").write_text("{}\n")
    written = glaive_v2.write_sets(sets, corpus, APACHE)
    kinds = ("parse", "render")
    assert written == [
        *(corpus / kind / "glaive-v2-00.jsonl" for kind in kinds),
        corpus / "licenses" / "glaive-v2-LICENSE",
    ]
    assert sorted(path.name for path in (corpus / "render").iterdir()) == [
        "bfcl-simple-python.jsonl",
        "common.jsonl",
        "glaive-v2-00.jsonl",
    ]
    assert (corpus / "render" / "common.jsonl").read_text() == "{}\n"
    text = (corpus / "parse" / "glaive-v2-00.jsonl").read_bytes().decode("utf-8")
    assert "étudiant" in text and text.endswith("\n")
    assert [json.loads(line) for line in text.split("\n")[:-1]] == sets[("parse", "glaive-v2-00")]
    assert glaive_v2.check_sets(sets, corpus, APACHE) == []
    (corpus / "parse" / "glaive-v2-00.jsonl").write_text("{}\n")
    (corpus / "render" / "glaive-v2-00.jsonl").unlink()
    (corpus / "render" / "glaive-v2-09.jsonl").write_text("{}\n")
    assert glaive_v2.check_sets(sets, corpus, APACHE) == [
        f"{corpus / 'parse' / 'glaive-v2-00.jsonl'}: differs from a fresh import",
        f"{corpus / 'render' / 'glaive-v2-00.jsonl'}: missing",
        f"{corpus / 'render' / 'glaive-v2-09.jsonl'}: no slice of the glaive-v2 sample writes it",
    ]


# The dataset card (README.md) at the pinned revision, all 106 bytes of it.
CARD = (
    "---\nlicense: apache-2.0\ntask_categories:\n- text-generation\nlanguage:\n- en\nsize_categories:\n- 100K<n<1M\n---"
)


def test_the_card_at_the_pin_states_the_reviewed_license():
    assert hf.card_license(CARD) == glaive_v2.CARD_LICENSE


def asked(cache) -> list[tuple]:
    """What the importer asks hf_hub_download for: the card, then the data, from the dataset at the pinned revision,
    into the Hugging Face layout under the import's ``--cache``, with no token."""
    pinned = {
        "repo_type": "dataset",
        "revision": "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
        "cache_dir": cache / "huggingface",
        "force_download": False,
        "token": False,
    }
    repo = "glaiveai/glaive-function-calling-v2"
    return [(repo, "README.md", pinned), (repo, "glaive-function-calling-v2.json", pinned)]


def serve(monkeypatch, tmp_path, rows: list[dict], card: str = CARD, license_text: bytes = APACHE) -> SimpleNamespace:
    """``hf_hub_download`` as the Hub's cache answers it, with these rows as the data file and ``card`` as the card, and
    ``github.fetch`` serving ``license_text``.

    The importer's pins are set to these files' sha256, so ``hf.fetch`` passes them. Returns the downloads asked for:
    ``hub`` as (repo, file, options), ``github`` as (owner, repo, commit, path, sha256).
    """
    files = {"README.md": card.encode("utf-8"), "glaive-function-calling-v2.json": json.dumps(rows).encode("utf-8")}
    for pin, filename in (("CARD_SHA256", "README.md"), ("DATA_SHA256", "glaive-function-calling-v2.json")):
        monkeypatch.setattr(glaive_v2, pin, hashlib.sha256(files[filename]).hexdigest())
    downloads: list = []

    def download(repo_id, filename, **kwargs):
        downloads.append((repo_id, filename, kwargs))
        path = tmp_path / "hub" / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(files[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    fetched: list = []

    def fetch(owner, repo, commit, path, sha256, cache=None):
        fetched.append((owner, repo, commit, path, sha256))
        served = tmp_path / "github" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(license_text)
        return served

    monkeypatch.setattr(github, "fetch", fetch)
    return SimpleNamespace(hub=downloads, github=fetched)


def test_the_command_writes_then_checks(tmp_path, monkeypatch, capsys):
    served = serve(monkeypatch, tmp_path, [chat_row("a"), BROKEN, chat_row("b"), chat_row("c")])
    monkeypatch.setattr(glaive_v2, "STEP", 2)
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    argv = ["import", "glaive-v2", "--corpus", str(corpus), "--cache", str(cache)]
    assert main([*argv, "--check"]) == 1
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    assert served.hub == asked(cache) * 3
    out = capsys.readouterr().out
    assert f"{corpus / 'render' / 'glaive-v2-00.jsonl'}: 2 cases" in out
    assert [line["name"] for line in map(json.loads, (corpus / "parse" / "glaive-v2-00.jsonl").open())] == [
        "glaive-v2-0-1",
        "glaive-v2-2-1",
    ]


def test_the_command_names_the_rows_it_refuses_by_reason_across_the_whole_file(tmp_path, monkeypatch, capsys):
    unended = {"system": NO_FUNCTIONS, "chat": "USER: Hi\n\nASSISTANT: Hello."}
    serve(monkeypatch, tmp_path, [chat_row("a"), *[BROKEN] * 52, unended, chat_row("b")])
    monkeypatch.setattr(glaive_v2, "STEP", 2)
    assert main(["import", "glaive-v2", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")]) == 0
    out = capsys.readouterr().out.splitlines()
    first_50 = ", ".join(str(row) for row in range(1, 51))
    assert f"no case for 52 row(s), the first 50 ({first_50}), 26 in the sample: {glaive_v2.STRAY_END}" in out
    assert f"no case for 1 row(s) (53), 0 in the sample: {glaive_v2.ASSISTANT_END}" in out
    assert "sample: every row whose index is a multiple of 2, 28 of 55 rows: 2 render and 2 parse cases" in out


def test_the_command_refuses_a_card_under_another_license_and_writes_nothing(tmp_path, monkeypatch):
    served = serve(monkeypatch, tmp_path, [chat_row("a")], card=CARD.replace("apache-2.0", "mit"))
    refused = (
        "hf:datasets/glaiveai/glaive-function-calling-v2@e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac README.md:"
        " the card's license is 'mit', not the reviewed 'apache-2.0'; review it before importing"
    )
    with pytest.raises(ValueError, match=re.escape(refused)):
        main(["import", "glaive-v2", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert served.hub == asked(tmp_path / "cache")[:1]
    assert not (tmp_path / "corpus").exists()


def test_the_command_refuses_a_data_file_that_is_not_the_pinned_one_and_writes_nothing(tmp_path, monkeypatch):
    serve(monkeypatch, tmp_path, [chat_row("a")])
    monkeypatch.setattr(glaive_v2, "DATA_SHA256", "0" * 64)
    # hf.fetch downloads a file that is not the pinned one once more, then refuses it by its path.
    refused = re.escape(f"{tmp_path / 'hub' / 'glaive-function-calling-v2.json'}: sha256 ")
    with pytest.raises(ValueError, match=refused + "[0-9a-f]{64} is not the pinned 0{64}"):
        main(["import", "glaive-v2", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert not (tmp_path / "corpus").exists()


def test_the_command_writes_the_apache_license_next_to_the_sets(tmp_path, monkeypatch):
    served = serve(monkeypatch, tmp_path, [chat_row("a")])
    corpus = tmp_path / "corpus"
    assert main(["import", "glaive-v2", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]) == 0
    assert (corpus / "licenses" / "glaive-v2-LICENSE").read_bytes() == APACHE
    # The dataset ships no LICENSE or NOTICE file; its card names apache-2.0, so the copy is the License as the Apache
    # Software Foundation publishes it, from its website's repository at the one commit that file has.
    commit, sha256 = "01b1be9fbc5cd93b6794f5653a58b9b863807f84", glaive_v2.LICENSE_SHA256
    assert served.github == [("apache", "www-site", commit, "content/licenses/LICENSE-2.0.txt", sha256)]
    assert sha256 == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def test_check_names_the_license_copy_when_it_is_missing_or_differs(tmp_path, monkeypatch, capsys):
    serve(monkeypatch, tmp_path, [chat_row("a")])
    corpus = tmp_path / "corpus"
    copy = corpus / "licenses" / "glaive-v2-LICENSE"
    argv = ["import", "glaive-v2", "--corpus", str(corpus), "--cache", str(tmp_path / "cache")]
    assert main(argv) == 0
    copy.unlink()
    assert main([*argv, "--check"]) == 1
    copy.write_bytes(APACHE + b"Additional terms apply.\n")
    assert main([*argv, "--check"]) == 1
    copy.write_bytes(APACHE)
    assert main([*argv, "--check"]) == 0
    assert capsys.readouterr().err.splitlines() == [f"{copy}: missing", f"{copy}: differs from a fresh import"]


def test_the_command_refuses_a_license_text_that_is_not_the_apache_license_and_writes_nothing(tmp_path, monkeypatch):
    serve(monkeypatch, tmp_path, [chat_row("a")], license_text=b"MIT License\n\nCopyright (c) 2024\n")
    with pytest.raises(ValueError, match="not the Apache License 2.0; review it before importing"):
        main(["import", "glaive-v2", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert not (tmp_path / "corpus").exists()
