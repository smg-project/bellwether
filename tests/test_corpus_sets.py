from bellwether.importers import corpus_sets


def render(name: str, request: dict) -> dict:
    """A render line as an importer builds it; its notes and origin differ from every other line's."""
    return {"name": name, "request": request, "notes": f"row {name}", "origin": {"dataset": "test", "row": name}}


def parse(name: str, request: dict, message: dict) -> dict:
    origin = {"dataset": "test", "row": name}
    return {"name": name, "request": request, "message": message, "notes": f"row {name}", "origin": origin}


def ask(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def names(sets: dict) -> dict:
    return {key: [line["name"] for line in lines] for key, lines in sets.items()}


def test_a_render_case_whose_request_equals_an_earlier_ones_is_left_out_naming_the_case_kept():
    sets = {
        ("render", "s"): [
            render("a", ask("Hi")),
            render("b", ask("Hi")),
            render("c", ask("Bye")),
            render("d", ask("Hi")),
        ]
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert kept == {("render", "s"): [render("a", ask("Hi")), render("c", ask("Bye"))]}
    assert repeats == [("b", "a"), ("d", "a")]
    assert names(sets) == {("render", "s"): ["a", "b", "c", "d"]}


def test_a_parse_case_repeats_an_earlier_one_only_when_its_message_is_equal_too():
    sets = {
        ("parse", "s"): [
            parse("a", ask("1 + 1?"), {"content": "2"}),
            parse("b", ask("1 + 1?"), {"content": "two"}),
            parse("c", ask("1 + 1?"), {"content": "2"}),
        ]
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert names(kept) == {("parse", "s"): ["a", "b"]}
    assert repeats == [("c", "a")]


def tool(*params: str) -> dict:
    properties = {param: {"type": "string"} for param in params}
    return {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": properties}}}


def test_key_order_counts_since_a_template_can_see_it():
    sets = {
        ("render", "s"): [
            render("a", {**ask("Hi"), "tools": [tool("x", "y")]}),
            render("b", {**ask("Hi"), "tools": [tool("y", "x")]}),
            render("c", {**ask("Hi"), "tools": [tool("x", "y")]}),
        ],
        ("parse", "s"): [
            parse("a", ask("Hi"), {"reasoning_content": "Greet.", "content": "Hello"}),
            parse("b", ask("Hi"), {"content": "Hello", "reasoning_content": "Greet."}),
        ],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert names(kept) == {("render", "s"): ["a", "b"], ("parse", "s"): ["a", "b"]}
    assert repeats == [("c", "a")]


def test_the_set_built_first_keeps_the_case_whatever_the_sets_are_named():
    sets = {
        ("render", "zeta"): [render("z", ask("Hi"))],
        ("render", "alpha"): [render("a", ask("Hi")), render("b", ask("Bye"))],
        ("render", "beta"): [render("c", ask("Bye"))],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert list(kept) == list(sets)
    assert names(kept) == {("render", "zeta"): ["z"], ("render", "alpha"): ["b"], ("render", "beta"): []}
    assert repeats == [("a", "z"), ("c", "b")]


def test_a_render_case_and_a_parse_case_never_repeat_each_other():
    sets = {("render", "s"): [render("a", ask("Hi"))], ("parse", "s"): [parse("a", ask("Hi"), {"content": "Hello"})]}
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert kept == sets and repeats == []


def test_when_nothing_repeats_the_sets_come_back_whole_and_the_list_is_empty():
    sets = {
        ("render", "s"): [render("a", ask("Hi")), render("b", ask("Bye"))],
        ("parse", "s"): [parse("c", ask("1 + 1?"), {"content": "2"}), parse("d", ask("2 + 2?"), {"content": "4"})],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    assert kept == sets and list(kept) == list(sets) and repeats == []


def test_the_report_names_each_repeat_then_each_set_by_kind_and_name_with_its_count_then_the_total(tmp_path, capsys):
    sets = {
        ("render", "zeta"): [render("a", ask("Hi")), render("b", ask("Hi"))],
        ("render", "alpha"): [render("c", ask("Hi")), render("d", ask("Bye"))],
        ("parse", "zeta"): [parse("e", ask("Hi"), {"content": "Hello"}), parse("f", ask("Hi"), {"content": "Hello"})],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    corpus_sets.report("Test", sets, kept, repeats, tmp_path)
    assert capsys.readouterr().out.splitlines() == [
        "no case b: it repeats a",
        "no case c: it repeats a",
        "no case f: it repeats e",
        f"{tmp_path / 'parse' / 'zeta.jsonl'}: 1 cases, 1 left out as repeats, 1 distinct messages",
        f"{tmp_path / 'render' / 'alpha.jsonl'}: 1 cases, 1 left out as repeats",
        f"{tmp_path / 'render' / 'zeta.jsonl'}: 1 cases, 1 left out as repeats",
        f"{tmp_path}: 3 cases in the 3 Test sets, 3 left out as repeats, 1 distinct messages",
    ]


def test_the_report_counts_distinct_messages_as_written_per_parse_set_and_once_across_the_sets(tmp_path, capsys):
    # Cases that are not repeats can carry the same message, which is all a parser sees: the counts keep them apart.
    sets = {
        ("parse", "s"): [
            parse("a", ask("Hi"), {"content": "Hello"}),
            parse("b", ask("Hey"), {"content": "Hello"}),
            parse("c", ask("Hi"), {"reasoning_content": "Greet.", "content": "Hello"}),
            parse("d", ask("Hi"), {"content": "Hello", "reasoning_content": "Greet."}),
        ],
        ("parse", "t"): [parse("e", ask("Yo"), {"content": "Hello"}), parse("f", ask("Yo"), {"content": "Bye"})],
    }
    kept, repeats = corpus_sets.leave_out_repeats(sets)
    corpus_sets.report("Test", sets, kept, repeats, tmp_path)
    assert capsys.readouterr().out.splitlines() == [
        f"{tmp_path / 'parse' / 's.jsonl'}: 4 cases, 3 distinct messages",
        f"{tmp_path / 'parse' / 't.jsonl'}: 2 cases, 2 distinct messages",
        f"{tmp_path}: 6 cases in the 2 Test sets, 0 left out as repeats, 4 distinct messages",
    ]


def test_each_reason_is_printed_once_naming_every_row_it_left_out_and_what_the_rows_do_not_get(capsys):
    skipped = [(f"row {row}", "no final answer" if row == 2 else "empty") for row in range(1, 6)]
    corpus_sets.report_skipped(skipped)
    corpus_sets.report_skipped([("simple_java_1", "not strings")], "parse case")
    assert capsys.readouterr().out.splitlines() == [
        "no case for 4 row(s) (row 1, row 3, row 4, row 5): empty",
        "no case for 1 row(s) (row 2): no final answer",
        "no parse case for 1 row(s) (simple_java_1): not strings",
    ]
