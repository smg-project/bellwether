import pytest

from bellwether.importers import bfcl, pypi

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


def test_the_message_has_one_call_per_ground_truth_entry_with_json_arguments():
    answer = {
        "id": "x",
        "ground_truth": [{"Cafe.order": {"drink": ["Café ☕"], "count": ["", 3], "note": [""]}}, {"ping": {}}],
    }
    assert bfcl.message_for(answer) == {
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "Cafe_order", "arguments": '{"drink": "Café ☕", "count": 3}'}},
            {"type": "function", "function": {"name": "ping", "arguments": "{}"}},
        ],
    }
