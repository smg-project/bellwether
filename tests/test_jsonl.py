import json

import pytest

from bellwether import jsonl


def test_only_a_newline_ends_a_line_so_unicode_line_breaks_and_crlf_inside_strings_read_back_intact():
    # json.dumps with ensure_ascii=False, as every writer here calls it, writes U+2028 and U+0085 raw, and
    # str.splitlines() ends a line at either; it escapes "\r\n", which must read back as written.
    values = [{"text": "one two"}, {"text": "three\u0085four"}, {"text": "five\r\nsix"}]
    first, second, third = (json.dumps(value, ensure_ascii=False) for value in values)
    assert " " in first and "\u0085" in second
    text = f"{first}\n{second}\n\n{third}\n"
    assert list(jsonl.loads(text, "set.jsonl")) == [(1, values[0]), (2, values[1]), (4, values[2])]


def test_a_line_that_is_not_json_is_named_by_where_it_came_from_and_its_number():
    with pytest.raises(ValueError, match=r"^set\.jsonl:2: not JSON: "):
        list(jsonl.loads('{"text": "one"}\n{"text": \n', "set.jsonl"))
