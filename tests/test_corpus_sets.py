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
