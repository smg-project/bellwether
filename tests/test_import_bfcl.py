import json
import zipfile

import pytest

from bellwether.cli import main
from bellwether.importers import bfcl, github, pypi
from bellwether.record.corpus import read_cases

WHEEL = "pkg-1.0-py3-none-any.whl"


class Response:
    def __init__(self, content: bytes = b"", payload: dict | None = None):
        self.content, self.payload = content, payload

    def raise_for_status(self):
        return self

    def json(self):
        return self.payload


def serve(data: bytes, calls: list[str]):
    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("/json"):
            return Response(payload={"urls": [{"filename": WHEEL, "url": "https://files.example/pkg.whl"}]})
        return Response(content=data)

    return get


def test_fetch_uses_a_cached_wheel_whose_hash_matches_without_the_network(tmp_path, monkeypatch):
    (tmp_path / WHEEL).write_bytes(b"wheel bytes")
    monkeypatch.setattr(pypi.httpx, "get", lambda *a, **k: pytest.fail("no download expected"))
    path = pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert path.read_bytes() == b"wheel bytes"


def test_fetch_downloads_the_named_file_and_caches_it(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(pypi.httpx, "get", serve(b"wheel bytes", calls))
    path = pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert path == tmp_path / WHEEL and path.read_bytes() == b"wheel bytes"
    assert calls == ["https://pypi.org/pypi/pkg/1.0/json", "https://files.example/pkg.whl"]


def test_fetch_replaces_a_cached_wheel_whose_hash_does_not_match(tmp_path, monkeypatch):
    (tmp_path / WHEEL).write_bytes(b"stale")
    monkeypatch.setattr(pypi.httpx, "get", serve(b"wheel bytes", []))
    path = pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert path.read_bytes() == b"wheel bytes"


def test_fetch_rejects_a_download_that_is_not_the_pinned_one_and_caches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(pypi.httpx, "get", serve(b"tampered", []))
    with pytest.raises(ValueError, match="is not the pinned"):
        pypi.fetch("pkg", "1.0", WHEEL, pypi.sha256_of(b"wheel bytes"), cache=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_names_a_file_pypi_does_not_have(tmp_path, monkeypatch):
    monkeypatch.setattr(pypi.httpx, "get", lambda url, **k: Response(payload={"urls": []}))
    with pytest.raises(ValueError, match="has no file pkg-1.0-py3-none-any.whl"):
        pypi.fetch("pkg", "1.0", WHEEL, "0" * 64, cache=tmp_path)


def fn(name: str, properties: dict, **extra) -> dict:
    return {"name": name, "description": "Does it.", "parameters": {"type": "dict", "properties": properties}, **extra}


def test_every_function_gets_its_language_hint():
    for category, hint in [
        ("simple_python", " Note that the provided function is in Python 3 syntax."),
        ("live_multiple", " Note that the provided function is in Python 3 syntax."),
        ("simple_java", " Note that the provided function is in Java 8 SDK syntax."),
        ("simple_javascript", " Note that the provided function is in JavaScript syntax."),
    ]:
        assert bfcl.prepare_functions([fn("f", {})], category)[0]["description"] == "Does it." + hint


def test_java_parameters_become_strings_that_say_their_type():
    properties = {
        "controller": {"type": "any", "description": "The controller."},
        "count": {"type": "integer", "description": "How many."},
        "names": {"type": "ArrayList", "description": "The names.", "items": {"type": "String"}},
    }
    [prepared] = bfcl.prepare_functions([fn("f", properties)], "simple_java")
    assert prepared["parameters"]["properties"] == {
        "controller": {
            "type": "string",
            "description": "The controller. This parameter can be of any type of Java object in string representation.",
        },
        "count": {
            "type": "string",
            "description": "How many. This is Java integer type parameter in string representation.",
        },
        "names": {
            "type": "string",
            "description": "The names. This is Java ArrayList type parameter in string representation."
            " The list elements are of type String; they are not in string representation.",
        },
    }


def test_a_javascript_dict_parameter_carries_its_schema_in_the_description():
    schema = {"x": {"type": "integer", "description": "X."}}
    point = {"type": "dict", "description": "A point.", "properties": schema}
    [prepared] = bfcl.prepare_functions([fn("f", {"point": point})], "simple_javascript")
    assert prepared["parameters"]["properties"]["point"] == {
        "type": "string",
        "description": "A point. This is JavaScript dict type parameter in string representation."
        " The dictionary entries have the following schema; they are not in string representation."
        ' {"x": {"type": "integer", "description": "X."}}',
    }


def test_a_float_becomes_a_number_with_its_format_and_a_note():
    [tool] = bfcl.to_tools([fn("f", {"rate": {"type": "float", "description": "The rate."}})])
    rate = tool["function"]["parameters"]["properties"]["rate"]
    assert rate == {"type": "number", "description": "The rate. This is a float type value.", "format": "float"}
    assert list(rate) == ["type", "description", "format"]


def test_types_map_to_openapi_and_anything_else_becomes_a_string():
    properties = {
        "pair": {"type": "tuple", "description": "P."},
        "big": {"type": "Bigint", "description": "B."},
        "odd": {"type": "complex", "description": "O."},
        "bare": {"description": "No type."},
    }
    [tool] = bfcl.to_tools([fn("f", properties)])
    cast = tool["function"]["parameters"]["properties"]
    assert [cast[k]["type"] for k in ("pair", "big", "odd", "bare")] == ["array", "integer", "string", "string"]
    assert list(cast["bare"]) == ["description", "type"]


def test_nested_properties_and_items_are_cast_as_bfcl_casts_them():
    float_w = {"type": "float", "description": "W."}
    float_v = {"type": "float", "description": "V."}
    properties = {
        "spec": {"type": "dict", "description": "S.", "properties": {"w": float_w}},
        "rows": {"type": "list", "description": "R.", "items": {"type": "dict", "properties": {"v": float_v}}},
        "grid": {"type": "array", "description": "G.", "items": {"type": "list", "items": {"type": "float"}}},
    }
    [tool] = bfcl.to_tools([fn("f", properties)])
    cast = tool["function"]["parameters"]["properties"]
    note = " This is a float type value."
    assert cast["spec"]["properties"]["w"] == {"type": "number", "description": "W." + note, "format": "float"}
    assert cast["rows"]["items"] == {
        "type": "object",
        "properties": {"v": {"type": "number", "description": "V." + note, "format": "float"}},
    }
    assert cast["grid"]["items"] == {"type": "array", "items": {"type": "number"}}


def test_dotted_names_get_underscores_and_the_parameters_become_an_object():
    original = fn("Geometry.createPresentation", {}, response={"type": "dict"})
    [tool] = bfcl.to_tools([original])
    assert tool == {
        "type": "function",
        "function": {
            "name": "Geometry_createPresentation",
            "description": "Does it.",
            "parameters": {"type": "object", "properties": {}},
            "response": {"type": "dict"},
        },
    }
    assert original["name"] == "Geometry.createPresentation" and original["parameters"]["type"] == "dict"


def test_pick_takes_the_first_value_that_is_not_the_omission_marker():
    assert bfcl.pick(["", 0]) == 0
    assert bfcl.pick(["units", ""]) == "units"
    assert bfcl.pick([None, ""]) is None
    assert bfcl.pick([[[10, 20], [30, 40]]]) == [[10, 20], [30, 40]]
    assert bfcl.pick([""]) is bfcl.OMIT


def test_a_dict_option_takes_one_value_per_key_and_leaves_out_keys_that_can_only_be_omitted():
    option = {
        "size": ["large"],
        "note": [""],
        "temperature": ["", "hot"],
        "pos": [{"lateral": 10.5, "longitudinal": 50}],
    }
    assert bfcl.realize(option) == {"size": "large", "temperature": "hot", "pos": {"lateral": 10.5, "longitudinal": 50}}


def test_a_list_of_dict_options_realizes_each_dict():
    option = [{"item": ["burgers"], "quantity": [5]}, {"item": ["chicken wings"], "quantity": [6]}]
    assert bfcl.realize(option) == [{"item": "burgers", "quantity": 5}, {"item": "chicken wings", "quantity": 6}]


def test_a_dict_option_whose_values_are_not_lists_is_refused():
    with pytest.raises(ValueError, match="not a list of acceptable values"):
        bfcl.realize({"size": "large"})


CAFE_FUNCTIONS = [
    fn("Cafe.order", {"drink": {"type": "string"}, "count": {"type": "integer"}, "note": {"type": "string"}}),
    fn("ping", {}),
]


def test_the_message_has_one_call_per_ground_truth_entry_with_json_arguments():
    answer = {
        "id": "x",
        "ground_truth": [{"Cafe.order": {"drink": ["Café ☕"], "count": ["", 3], "note": [""]}}, {"ping": {}}],
    }
    assert bfcl.message_for(answer, CAFE_FUNCTIONS) == {
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "Cafe_order", "arguments": '{"drink": "Café ☕", "count": 3}'}},
            {"type": "function", "function": {"name": "ping", "arguments": "{}"}},
        ],
    }


def fake_wheel(tmp_path, members: dict[str, list[dict] | str]):
    """A wheel holding each member as JSON Lines, or as the text given."""
    path = tmp_path / "bfcl.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        metadata = "Metadata-Version: 2.1\nName: bfcl-eval\nLicense: Apache 2.0\n"
        wheel.writestr("bfcl_eval-2026.3.23.dist-info/METADATA", metadata)
        for name, rows in members.items():
            text = (
                rows if isinstance(rows, str) else "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            wheel.writestr(name, text)
    return path


DRINK = {"drink": {"type": "string", "description": "D."}}
SIMPLE = {
    "id": "simple_python_0",
    "question": [[{"role": "user", "content": "Order a Café ☕"}]],
    "function": [
        {
            "name": "Cafe.order",
            "description": "Orders.",
            "parameters": {"type": "dict", "properties": DRINK, "required": ["drink"]},
        }
    ],
}
ANSWER = {"id": "simple_python_0", "ground_truth": [{"Cafe.order": {"drink": ["Café ☕"]}}]}
IRRELEVANT = {"id": "irrelevance_0", "question": [[{"role": "user", "content": "Hi"}]], "function": []}


def build(tmp_path):
    path = fake_wheel(
        tmp_path,
        {
            "bfcl_eval/data/BFCL_v4_simple_python.json": [SIMPLE],
            "bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json": [ANSWER],
            "bfcl_eval/data/BFCL_v4_irrelevance.json": [IRRELEVANT],
        },
    )
    with zipfile.ZipFile(path) as wheel:
        return bfcl.build_sets(wheel, categories=("simple_python", "irrelevance"))


def test_render_sets_for_every_category_and_parse_sets_where_an_answer_exists(tmp_path):
    sets = build(tmp_path)
    assert sorted(sets) == [
        ("parse", "bfcl-simple-python"),
        ("render", "bfcl-irrelevance"),
        ("render", "bfcl-simple-python"),
    ]
    origin = {
        "dataset": "bfcl",
        "source": "pypi:bfcl-eval==2026.3.23",
        "sha256": "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
        "file": "bfcl_eval/data/BFCL_v4_simple_python.json",
        "row": "simple_python_0",
        "license": "Apache-2.0",
    }
    request = {
        "messages": [{"role": "user", "content": "Order a Café ☕"}],
        "temperature": 0.001,
        "store": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Cafe_order",
                    "description": "Orders. Note that the provided function is in Python 3 syntax.",
                    "parameters": {"type": "object", "properties": DRINK, "required": ["drink"]},
                },
            }
        ],
    }
    assert sets[("render", "bfcl-simple-python")] == [
        {
            "name": "bfcl-simple-python-0",
            "request": request,
            "notes": "BFCL simple_python simple_python_0",
            "origin": origin,
        }
    ]
    [parse] = sets[("parse", "bfcl-simple-python")]
    assert list(parse) == ["name", "request", "message", "notes", "origin"]
    assert parse["origin"] == {
        **{k: origin[k] for k in ("dataset", "source", "sha256", "file")},
        "answer_file": "bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json",
        "row": "simple_python_0",
        "license": "Apache-2.0",
    }
    assert list(parse["origin"]) == ["dataset", "source", "sha256", "file", "answer_file", "row", "license"]
    assert parse["message"]["tool_calls"][0]["function"] == {"name": "Cafe_order", "arguments": '{"drink": "Café ☕"}'}
    assert sets[("render", "bfcl-irrelevance")][0]["request"] == {
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.001,
        "store": False,
    }


def test_a_row_whose_text_holds_unicode_line_breaks_is_read_intact(tmp_path):
    text = "Order a Café\u0085now"
    row = dict(SIMPLE, question=[[{"role": "user", "content": text}]])
    path = fake_wheel(tmp_path, {"bfcl_eval/data/BFCL_v4_simple_python.json": [row]})
    with zipfile.ZipFile(path) as wheel:
        [line] = bfcl.build_sets(wheel, categories=("simple_python",))[("render", "bfcl-simple-python")]
    assert line["request"]["messages"] == [{"role": "user", "content": text}]


def test_two_rows_with_one_case_name_stop_the_import(tmp_path):
    twin = dict(SIMPLE, id="simple-python-0")
    path = fake_wheel(tmp_path, {"bfcl_eval/data/BFCL_v4_simple_python.json": [SIMPLE, twin]})
    with zipfile.ZipFile(path) as wheel, pytest.raises(ValueError, match="bfcl-simple-python-0"):
        bfcl.build_sets(wheel, categories=("simple_python",))


# The heading of the Apache License 2.0, laid out as the gorilla repository's LICENSE lays it out.
APACHE = b"                                 Apache License\n                           Version 2.0, January 2004\n"
# The LICENSE the import copies: the gorilla repository's root LICENSE at the commit the wheel was built from.
LICENSE_PIN = (
    "ShishirPatil",
    "gorilla",
    "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    "LICENSE",
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
)


def test_a_single_turn_row_with_more_than_one_turn_stops_the_import(tmp_path):
    row = dict(SIMPLE, question=[[{"role": "user", "content": "Order"}], [{"role": "user", "content": "Again"}]])
    path = fake_wheel(tmp_path, {"bfcl_eval/data/BFCL_v4_simple_python.json": [row]})
    with zipfile.ZipFile(path) as wheel:
        with pytest.raises(ValueError, match="simple_python_0: a single-turn row with 2 turns"):
            bfcl.build_sets(wheel, categories=("simple_python",))


def test_written_sets_check_clean_and_a_changed_or_stale_file_is_reported(tmp_path):
    sets, corpus = build(tmp_path), tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    (corpus / "render" / "common.jsonl").write_text("{}\n")
    (corpus / "render" / "bfcl-old.jsonl").write_text("{}\n")
    bfcl.write_sets(sets, corpus, APACHE)
    assert not (corpus / "render" / "bfcl-old.jsonl").exists()
    assert (corpus / "render" / "common.jsonl").read_text() == "{}\n"
    text = (corpus / "render" / "bfcl-simple-python.jsonl").read_bytes().decode("utf-8")
    assert "Café ☕" in text and text.endswith("\n")
    assert bfcl.check_sets(sets, corpus, APACHE) == []
    (corpus / "parse" / "bfcl-simple-python.jsonl").write_text("{}\n")
    (corpus / "render" / "bfcl-stale.jsonl").write_text("{}\n")
    assert bfcl.check_sets(sets, corpus, APACHE) == [
        f"{corpus / 'parse' / 'bfcl-simple-python.jsonl'}: differs from a fresh import",
        f"{corpus / 'render' / 'bfcl-stale.jsonl'}: no BFCL category writes it",
    ]


def serve_pins(tmp_path, monkeypatch, wheel, license_text: bytes = APACHE) -> None:
    """Stand in for both pinned fetches, so tests never touch the network: the wheel, and the LICENSE it lacks."""
    monkeypatch.setattr(pypi, "fetch", lambda *a, **k: wheel)

    def fetch(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path, sha256) == LICENSE_PIN
        served = tmp_path / "served" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(license_text)
        return served

    monkeypatch.setattr(github, "fetch", fetch)


def simple_wheel(tmp_path):
    return fake_wheel(
        tmp_path,
        {
            "bfcl_eval/data/BFCL_v4_simple_python.json": [SIMPLE],
            "bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json": [ANSWER],
        },
    )


def test_the_command_writes_then_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bfcl, "CATEGORIES", ("simple_python",))
    serve_pins(tmp_path, monkeypatch, simple_wheel(tmp_path))
    corpus = tmp_path / "corpus"
    assert main(["import", "bfcl", "--corpus", str(corpus), "--check"]) == 1
    assert main(["import", "bfcl", "--corpus", str(corpus)]) == 0
    assert main(["import", "bfcl", "--corpus", str(corpus), "--check"]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'render' / 'bfcl-simple-python.jsonl'}: 1 cases" in out


def test_the_command_writes_the_pinned_license_next_to_the_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "CATEGORIES", ("simple_python",))
    serve_pins(tmp_path, monkeypatch, simple_wheel(tmp_path))
    corpus = tmp_path / "corpus"
    assert main(["import", "bfcl", "--corpus", str(corpus)]) == 0
    assert (corpus / "licenses" / "bfcl-LICENSE").read_bytes() == APACHE


def test_check_names_the_license_copy_when_it_is_missing_or_differs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bfcl, "CATEGORIES", ("simple_python",))
    serve_pins(tmp_path, monkeypatch, simple_wheel(tmp_path))
    corpus = tmp_path / "corpus"
    copy = corpus / "licenses" / "bfcl-LICENSE"
    argv = ["import", "bfcl", "--corpus", str(corpus)]
    assert main(argv) == 0
    copy.unlink(missing_ok=True)
    assert main([*argv, "--check"]) == 1
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(APACHE + b"Additional terms apply.\n")
    assert main([*argv, "--check"]) == 1
    copy.write_bytes(APACHE)
    assert main([*argv, "--check"]) == 0
    assert capsys.readouterr().err.splitlines() == [f"{copy}: missing", f"{copy}: differs from a fresh import"]


def test_the_command_refuses_a_license_file_that_is_not_the_apache_license_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl, "CATEGORIES", ("simple_python",))
    serve_pins(tmp_path, monkeypatch, simple_wheel(tmp_path), license_text=b"MIT License\n\nCopyright (c) 2021\n")
    with pytest.raises(ValueError, match="not the Apache License 2.0"):
        main(["import", "bfcl", "--corpus", str(tmp_path / "corpus")])
    assert not (tmp_path / "corpus").exists()


def test_the_command_leaves_out_cases_that_repeat_earlier_ones_and_names_what_they_repeat(
    tmp_path, monkeypatch, capsys
):
    # As live_irrelevance_118-7-8 sends what live_simple_29-7-2 sends, irrelevance_1 sends what simple_python_0 sends.
    path = fake_wheel(
        tmp_path,
        {
            "bfcl_eval/data/BFCL_v4_simple_python.json": [SIMPLE],
            "bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json": [ANSWER],
            "bfcl_eval/data/BFCL_v4_irrelevance.json": [IRRELEVANT, dict(SIMPLE, id="irrelevance_1")],
        },
    )
    monkeypatch.setattr(bfcl, "CATEGORIES", ("simple_python", "irrelevance"))
    serve_pins(tmp_path, monkeypatch, path)
    corpus = tmp_path / "corpus"
    assert main(["import", "bfcl", "--corpus", str(corpus)]) == 0
    assert main(["import", "bfcl", "--corpus", str(corpus), "--check"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "no case bfcl-irrelevance-1: it repeats bfcl-simple-python-0" in out
    assert f"{corpus / 'render' / 'bfcl-irrelevance.jsonl'}: 1 cases, 1 left out as repeats" in out
    assert f"{corpus / 'parse' / 'bfcl-simple-python.jsonl'}: 1 cases, 1 distinct messages" in out
    assert f"{corpus}: 3 cases in the 3 BFCL sets, 1 left out as repeats, 1 distinct messages" in out
    assert [case.name for case in read_cases(corpus / "render" / "bfcl-irrelevance.jsonl")] == ["bfcl-irrelevance-0"]


def metadata_wheel(tmp_path, text: str):
    path = tmp_path / "metadata.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("bfcl_eval-2026.3.23.dist-info/METADATA", text)
    return path


def test_the_reviewed_license_passes(tmp_path):
    text = "Metadata-Version: 2.1\nName: bfcl-eval\nLicense: Apache 2.0\n"
    with zipfile.ZipFile(metadata_wheel(tmp_path, text)) as wheel:
        bfcl.check_license(wheel)


def test_a_wheel_under_another_license_is_refused(tmp_path):
    text = "Metadata-Version: 2.1\nName: bfcl-eval\nLicense: MIT\n"
    with zipfile.ZipFile(metadata_wheel(tmp_path, text)) as wheel, pytest.raises(ValueError, match="license 'MIT'"):
        bfcl.check_license(wheel)


def test_a_license_line_in_the_description_does_not_count(tmp_path):
    text = "Metadata-Version: 2.1\nName: bfcl-eval\n\nLicense: Apache 2.0\n"
    with zipfile.ZipFile(metadata_wheel(tmp_path, text)) as wheel, pytest.raises(ValueError, match="license None"):
        bfcl.check_license(wheel)


JAVA_FN = [
    {
        "name": "Box.make",
        "description": "Makes.",
        "parameters": {
            "type": "dict",
            "properties": {
                "size": {"type": "integer", "description": "S."},
                "label": {"type": "String", "description": "L."},
                "meta": {"type": "HashMap", "description": "M."},
            },
        },
    }
]


def java_sets(tmp_path, answers: list[dict], skipped: list | None = None):
    rows = [{"id": a["id"], "question": [[{"role": "user", "content": "Box"}]], "function": JAVA_FN} for a in answers]
    path = fake_wheel(
        tmp_path,
        {
            "bfcl_eval/data/BFCL_v4_simple_java.json": rows,
            "bfcl_eval/data/possible_answer/BFCL_v4_simple_java.json": answers,
        },
    )
    with zipfile.ZipFile(path) as wheel:
        return bfcl.build_sets(wheel, categories=("simple_java",), skipped=skipped)


def test_the_command_names_every_row_it_gives_no_parse_case(tmp_path, monkeypatch, capsys):
    ids = [f"simple_java_{i}" for i in range(1, 6)]
    rows = [{"id": i, "question": [[{"role": "user", "content": "Box"}]], "function": JAVA_FN} for i in ids]
    answers = [{"id": i, "ground_truth": [{"Box.make": {"size": [5]}}]} for i in ids]
    path = fake_wheel(
        tmp_path,
        {
            "bfcl_eval/data/BFCL_v4_simple_java.json": rows,
            "bfcl_eval/data/possible_answer/BFCL_v4_simple_java.json": answers,
        },
    )
    monkeypatch.setattr(bfcl, "CATEGORIES", ("simple_java",))
    serve_pins(tmp_path, monkeypatch, path)
    assert main(["import", "bfcl", "--corpus", str(tmp_path / "corpus")]) == 0
    out = capsys.readouterr().out
    assert f"no parse case for 5 row(s) ({', '.join(ids)}): {bfcl.NOT_STRINGS}" in out


def test_java_rows_whose_values_are_not_all_strings_get_no_parse_case(tmp_path):
    sets = java_sets(
        tmp_path,
        [
            {"id": "simple_java_1", "ground_truth": [{"Box.make": {"size": [5], "label": ["big"]}}]},
            {"id": "simple_java_2", "ground_truth": [{"Box.make": {"label": ["big"], "size": [""]}}]},
            {"id": "simple_java_3", "ground_truth": [{"Box.make": {"meta": [{"format": "epoch_millis"}]}}]},
        ],
    )
    assert [line["name"] for line in sets[("render", "bfcl-simple-java")]] == [
        "bfcl-simple-java-1",
        "bfcl-simple-java-2",
        "bfcl-simple-java-3",
    ]
    [parse] = sets[("parse", "bfcl-simple-java")]
    assert parse["name"] == "bfcl-simple-java-2"
    skipped: list[tuple[str, str]] = []
    java_sets(tmp_path, [{"id": "simple_java_1", "ground_truth": [{"Box.make": {"size": [5]}}]}], skipped)
    assert skipped == [("simple_java_1", bfcl.NOT_STRINGS)]
    assert parse["message"]["tool_calls"][0]["function"]["arguments"] == '{"label": "big"}'


def test_a_parameter_the_function_does_not_declare_is_left_out_when_it_may_be_omitted():
    answer = {"id": "x", "ground_truth": [{"ping": {"stray": ["", 0.1]}}]}
    message = bfcl.message_for(answer, CAFE_FUNCTIONS)
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_a_ground_truth_no_call_can_satisfy_is_unanswerable():
    required = [
        fn(
            "record",
            {"start": {"type": "string"}},
        )
    ]
    required[0]["parameters"]["required"] = ["start"]
    for params, reason in [
        ({"start": []}, "record requires start"),
        ({"start": [""]}, "record requires start"),
        ({"question": ["hi"]}, "record has no parameter 'question'"),
    ]:
        with pytest.raises(bfcl.Unanswerable, match=reason):
            bfcl.message_for({"id": "x", "ground_truth": [{"record": params}]}, required)
    with pytest.raises(bfcl.Unanswerable, match="calls missing, which the row does not define"):
        bfcl.message_for({"id": "x", "ground_truth": [{"missing": {}}]}, required)


def test_rows_without_a_parse_case_are_reported_with_their_reason(tmp_path):
    record = fn("record", {"start": {"type": "string"}})
    record["parameters"]["required"] = ["start"]
    row = {"id": "live_simple_1", "question": [[{"role": "user", "content": "Go"}]], "function": [record]}
    answer = {"id": "live_simple_1", "ground_truth": [{"record": {"start": []}}]}
    path = fake_wheel(
        tmp_path,
        {
            "bfcl_eval/data/BFCL_v4_live_simple.json": [row],
            "bfcl_eval/data/possible_answer/BFCL_v4_live_simple.json": [answer],
        },
    )
    skipped: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as wheel:
        sets = bfcl.build_sets(wheel, categories=("live_simple",), skipped=skipped)
    assert ("parse", "bfcl-live-simple") not in sets and len(sets[("render", "bfcl-live-simple")]) == 1
    assert skipped == [("live_simple_1", "record requires start, which the ground truth gives no value")]


BACKEND_CONFIG = "bfcl_eval/constants/executable_backend_config.py"
# Running this module would exit: the importer must read the maps from its source, never import it.
BACKEND = (
    'raise SystemExit("the backend config was run")\n'
    'MULTI_TURN_FUNC_DOC_FILE_MAPPING = {"Mail": "mail.json", "Cafe": "cafe.json"}\n'
    'BACKEND_PATH_PREFIX = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"\n'
    'CLASS_FILE_PATH_MAPPING = {"Mail": f"{BACKEND_PATH_PREFIX}.mail", "Cafe": f"{BACKEND_PATH_PREFIX}.cafe"}\n'
)
SOURCES = "bfcl_eval/eval_checker/multi_turn_eval/func_source_code"


def test_the_class_to_file_map_is_read_from_the_backend_source_without_running_it(tmp_path):
    with zipfile.ZipFile(fake_wheel(tmp_path, {BACKEND_CONFIG: BACKEND})) as wheel:
        assert bfcl.func_doc_files(wheel) == {"Mail": "mail.json", "Cafe": "cafe.json"}


def test_the_class_to_module_map_is_read_from_the_backend_source_without_running_it(tmp_path):
    with zipfile.ZipFile(fake_wheel(tmp_path, {BACKEND_CONFIG: BACKEND})) as wheel:
        assert bfcl.class_files(wheel) == {"Mail": f"{SOURCES}/mail.py", "Cafe": f"{SOURCES}/cafe.py"}


def test_a_backend_config_without_the_maps_stops_the_import(tmp_path):
    config = 'BACKEND_PATH_PREFIX = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"\n'
    with zipfile.ZipFile(fake_wheel(tmp_path, {BACKEND_CONFIG: config})) as wheel:
        with pytest.raises(ValueError, match="assigns no MULTI_TURN_FUNC_DOC_FILE_MAPPING"):
            bfcl.func_doc_files(wheel)
        with pytest.raises(ValueError, match="assigns no CLASS_FILE_PATH_MAPPING"):
            bfcl.class_files(wheel)


def test_a_multi_turn_row_offers_its_classes_functions_less_those_held_back():
    docs = {"Cafe": [fn("order", {}), fn("pay", {})], "Mail": [fn("send", {}), fn("sort", {})]}
    row = {"id": "multi_turn_miss_func_0", "involved_classes": ["Mail", "Cafe"], "missed_function": {"2": ["sort"]}}
    assert [f["name"] for f in bfcl.first_turn_functions(row, docs)] == ["send", "order", "pay"]
    row["missed_function"]["1"] = ["pay"]
    assert [f["name"] for f in bfcl.first_turn_functions(row, docs)] == ["send", "order"]
    with pytest.raises(ValueError, match="holds back pay at the first turn"):
        bfcl.first_turn_functions(dict(row, missed_function={"0": ["pay"]}), docs)


def test_a_held_back_name_takes_out_only_the_first_function_of_that_name():
    # BFCL takes out the first doc of each held-back name and stops looking (bfcl_eval/utils.py:793-799).
    first, second = fn("pay", {}), fn("pay", {"to": {"type": "string"}})
    docs = {"Cafe": [fn("order", {}), first], "Mail": [fn("send", {}), second]}
    row = {"id": "multi_turn_miss_func_0", "involved_classes": ["Cafe", "Mail"], "missed_function": {"1": ["pay"]}}
    assert bfcl.first_turn_functions(row, docs) == [fn("order", {}), fn("send", {}), second]


def test_a_held_back_name_the_row_does_not_offer_takes_nothing_out():
    docs = {"Cafe": [fn("order", {}), fn("pay", {})]}
    row = {"id": "multi_turn_miss_func_0", "involved_classes": ["Cafe"], "missed_function": {"1": ["refund"]}}
    assert bfcl.first_turn_functions(row, docs) == [fn("order", {}), fn("pay", {})]


SENT = {"type": "dict", "properties": {"sent": {"type": "boolean", "description": "Sent."}}}
MAIL_DOCS = [
    {
        "name": "send",
        "description": "Sends a message.",
        "parameters": {
            "type": "dict",
            "properties": {"to": {"type": "string", "description": "T."}},
            "required": ["to"],
        },
        "response": SENT,
    },
    {"name": "sort", "description": "Sorts.", "parameters": {"type": "dict", "properties": {}, "required": []}},
]
DRINK_SIZE = {"drink": {"type": "string", "description": "D."}, "size": {"type": "float", "description": "S."}}
CAFE_DOCS = [
    {
        "name": "order",
        "description": "Orders a drink.",
        "parameters": {"type": "dict", "properties": DRINK_SIZE, "required": ["drink"]},
    }
]
MT_ROW = {
    "id": "multi_turn_miss_func_0",
    "question": [
        [{"role": "user", "content": "Order a Café ☕, then mail Bo"}],
        [],
        [{"role": "user", "content": "Go"}],
    ],
    "initial_config": {"Mail": {"inbox": []}},
    "path": ["Cafe.order", "Mail.send", "Mail.sort"],
    "involved_classes": ["Cafe", "Mail"],
    "missed_function": {"1": ["sort"]},
}
# The classes' sources, which exit when run: the importer must read their defs with ast, never import them.
NEVER_RUN = 'raise SystemExit("the class source was run")\n'
MAIL_SOURCE = NEVER_RUN + "class Mail:\n    def send(self, to): ...\n    def sort(self): ...\n"
CAFE_SOURCE = NEVER_RUN + "class Cafe:\n    def order(self, drink, size=None): ...\n"


def multi_turn_wheel(
    tmp_path, category: str, rows: list[dict], answers: list[dict] | None = None, sources: dict[str, str] | None = None
):
    members = {
        BACKEND_CONFIG: BACKEND,
        "bfcl_eval/data/multi_turn_func_doc/mail.json": MAIL_DOCS,
        "bfcl_eval/data/multi_turn_func_doc/cafe.json": CAFE_DOCS,
        f"{SOURCES}/mail.py": MAIL_SOURCE,
        f"{SOURCES}/cafe.py": CAFE_SOURCE,
        **(sources or {}),
        f"bfcl_eval/data/BFCL_v4_{category}.json": rows,
    }
    if answers is not None:
        members[f"bfcl_eval/data/possible_answer/BFCL_v4_{category}.json"] = answers
    return fake_wheel(tmp_path, members)


def test_a_multi_turn_row_renders_its_first_turn_with_the_functions_it_offers(tmp_path):
    with zipfile.ZipFile(multi_turn_wheel(tmp_path, "multi_turn_miss_func", [MT_ROW])) as wheel:
        sets = bfcl.build_sets(wheel, categories=("multi_turn_miss_func",))
    hint = " Note that the provided function is in Python 3 syntax."
    size = {"type": "number", "description": "S. This is a float type value.", "format": "float"}
    order = {
        "name": "order",
        "description": "Orders a drink." + hint,
        "parameters": {
            "type": "object",
            "properties": {"drink": DRINK_SIZE["drink"], "size": size},
            "required": ["drink"],
        },
    }
    send = {**MAIL_DOCS[0], "description": "Sends a message." + hint}
    send["parameters"] = {**send["parameters"], "type": "object"}
    assert sets == {
        ("render", "bfcl-multi-turn-miss-func"): [
            {
                "name": "bfcl-multi-turn-miss-func-0",
                "request": {
                    "messages": [{"role": "user", "content": "Order a Café ☕, then mail Bo"}],
                    "temperature": 0.001,
                    "store": False,
                    "tools": [{"type": "function", "function": order}, {"type": "function", "function": send}],
                },
                "notes": "BFCL multi_turn_miss_func multi_turn_miss_func_0, first turn",
                "origin": {
                    "dataset": "bfcl",
                    "source": "pypi:bfcl-eval==2026.3.23",
                    "sha256": "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
                    "file": "bfcl_eval/data/BFCL_v4_multi_turn_miss_func.json",
                    "row": "multi_turn_miss_func_0",
                    "license": "Apache-2.0",
                },
            }
        ]
    }


NOTE = fn("note", {"tags": {"type": "array"}, "due": {"type": "dict"}, "pin": {"type": "boolean"}})
# The parameters each method's def takes by position, as read_func_defs gives them.
DEFS = {"order": ["drink", "size"], "send": ["to"], "sort": [], "note": ["tags", "due", "pin"]}


def test_the_first_turn_gold_calls_become_one_message_in_their_order():
    answer = {
        "id": "multi_turn_base_0",
        "ground_truth": [
            ["order('Café ☕', size=-1.5)", "note(['a', 'b'], pin=True, due={'day': 2, 'by': None})", "send(to='Bo')"],
            ["sort()"],
        ],
    }
    assert bfcl.first_turn_message(answer, CAFE_DOCS + MAIL_DOCS + [NOTE], DEFS) == {
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "order", "arguments": '{"drink": "Café ☕", "size": -1.5}'}},
            {
                "type": "function",
                "function": {
                    "name": "note",
                    "arguments": '{"tags": ["a", "b"], "pin": true, "due": {"day": 2, "by": null}}',
                },
            },
            {"type": "function", "function": {"name": "send", "arguments": '{"to": "Bo"}'}},
        ],
    }


def test_a_first_turn_call_that_breaks_the_checker_rules_is_unanswerable():
    for call, reason in [
        ("sort()", "the ground truth calls sort, which the first turn does not offer"),
        ("order(drink='tea', milk=True)", "order has no parameter 'milk', which the ground truth requires"),
        ("order(size=2.0)", "order requires drink, which the ground truth gives no value"),
        ("order('tea', 2.0, 3)", "the def of order takes 2 parameters, and the ground truth passes 3 by position"),
        ("order('tea', drink='tea')", "order gets drink both by position and by name"),
    ]:
        with pytest.raises(bfcl.Unanswerable, match=reason):
            bfcl.first_turn_message({"id": "x", "ground_truth": [[call]]}, CAFE_DOCS, DEFS)


BASE_ROW = {key: value for key, value in MT_ROW.items() if key != "missed_function"} | {"id": "multi_turn_base_0"}


def first_turn_calls(tmp_path, call: str, sources: dict[str, str]) -> list[dict]:
    answer = {"id": "multi_turn_base_0", "ground_truth": [[call]]}
    with zipfile.ZipFile(multi_turn_wheel(tmp_path, "multi_turn_base", [BASE_ROW], [answer], sources)) as wheel:
        [parse] = bfcl.build_sets(wheel, categories=("multi_turn_base",))[("parse", "bfcl-multi-turn-base")]
    return parse["message"]["tool_calls"]


def test_values_passed_by_position_take_the_parameters_of_the_def_not_of_the_doc(tmp_path):
    # BFCL's executor runs the call on the class, so its def binds the values. The doc lists drink first and the def
    # size first, as TravelAPI.purchase_insurance's doc and def order booking_id and insurance_cost differently.
    swapped = NEVER_RUN + "class Cafe:\n    def order(self, size, drink): ...\n"
    calls = first_turn_calls(tmp_path, "order(1.5, 'tea')", {f"{SOURCES}/cafe.py": swapped})
    assert calls == [{"type": "function", "function": {"name": "order", "arguments": '{"size": 1.5, "drink": "tea"}'}}]


def test_a_value_passed_by_position_to_a_function_without_a_def_stops_the_import(tmp_path):
    no_send = NEVER_RUN + "class Mail:\n    def sort(self): ...\n"
    with pytest.raises(ValueError, match="no def of send"):
        first_turn_calls(tmp_path, "send('Bo')", {f"{SOURCES}/mail.py": no_send})


def test_a_ground_truth_entry_that_is_not_a_call_to_a_named_function_stops_the_import(tmp_path):
    for entry in ["Mail.send(to='Bo')", "send"]:
        with pytest.raises(ValueError, match="is not a call to a named function"):
            first_turn_calls(tmp_path, entry, {})


def test_the_command_writes_first_turn_parse_cases_and_names_the_rows_without_a_first_call(
    tmp_path, monkeypatch, capsys
):
    row = {key: value for key, value in MT_ROW.items() if key != "missed_function"}
    # The first turns differ, so that neither row's case repeats the other's.
    other = dict(row, question=[[{"role": "user", "content": "Mail Bo"}], *row["question"][1:]])
    rows = [dict(row, id="multi_turn_miss_param_0"), dict(other, id="multi_turn_miss_param_1")]
    answers = [
        {"id": "multi_turn_miss_param_0", "ground_truth": [["send(to='Bo')"], [], ["sort()"]]},
        {"id": "multi_turn_miss_param_1", "ground_truth": [[], ["send(to='Bo')"], ["sort()"]]},
    ]
    path = multi_turn_wheel(tmp_path, "multi_turn_miss_param", rows, answers)
    monkeypatch.setattr(bfcl, "CATEGORIES", ("multi_turn_miss_param",))
    monkeypatch.setattr(pypi, "fetch", lambda *a, **k: path)
    corpus = tmp_path / "corpus"
    assert main(["import", "bfcl", "--corpus", str(corpus)]) == 0
    out = capsys.readouterr().out
    assert f"{corpus / 'render' / 'bfcl-multi-turn-miss-param.jsonl'}: 2 cases" in out
    assert f"{corpus / 'parse' / 'bfcl-multi-turn-miss-param.jsonl'}: 1 cases" in out
    assert f"no parse case for 1 row(s) (multi_turn_miss_param_1): {bfcl.NO_FIRST_CALL}" in out
    text = (corpus / "parse" / "bfcl-multi-turn-miss-param.jsonl").read_text(encoding="utf-8")
    [parse] = [json.loads(line) for line in text.splitlines()]
    assert parse["name"] == "bfcl-multi-turn-miss-param-0"
    assert parse["message"] == {
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "send", "arguments": '{"to": "Bo"}'}}],
    }
    assert parse["origin"]["answer_file"] == "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_miss_param.json"
    assert list(parse["origin"]) == ["dataset", "source", "sha256", "file", "answer_file", "row", "license"]
    assert main(["import", "bfcl", "--corpus", str(corpus), "--check"]) == 0


# The categories smg's weekly run sends, in its order (.github/workflows/nightly-bfcl.yml in smg).
WEEKLY = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def test_the_import_takes_the_weekly_categories_in_order_but_leaves_long_context_out():
    assert bfcl.CATEGORIES == tuple(category for category in WEEKLY if category != "multi_turn_long_context")
