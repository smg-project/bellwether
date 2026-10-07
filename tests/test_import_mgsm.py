import re
import unicodedata

import pytest

from bellwether.cli import main
from bellwether.importers import github, mgsm
from bellwether.record.corpus import read_cases

CC_BY = (
    b"Creative Commons Attribution 4.0 International Public License (CC-BY)\n"
    b"\n   By exercising the Licensed Rights (defined below), You accept and agree\n"
)


def test_the_reviewed_license_passes():
    mgsm.check_license(CC_BY)


def test_a_license_whose_first_line_is_not_cc_by_4_is_refused():
    share_alike = b"Creative Commons Attribution-ShareAlike 4.0 International Public License (CC-BY-SA)\n"
    title_later = b"MIT License\n\nCreative Commons Attribution 4.0 International Public License (CC-BY)\n"
    for other in [share_alike, title_later]:
        with pytest.raises(ValueError, match="not the CC-BY-4.0 license"):
            mgsm.check_license(other)


def tsv(lines: list[str]) -> bytes:
    """A language's file: no header, each line ending in "\\n"."""
    return "".join(line + "\n" for line in lines).encode("utf-8")


def test_lines_split_at_their_tabs_only_in_file_order_with_their_line_index():
    lines = [
        'Mientras está en la tienda "Todo por un dólar", Sloane cuenta 100 clientes.\t100',
        '"Quoted" at the start, ""doubled"" inside\t7',
        "Clive opens a box.\u2028It holds 6 balls.\t6",
        "",
        "one\ttwo\tthree",
    ]
    assert mgsm.read_rows(tsv(lines)) == [
        (0, ['Mientras está en la tienda "Todo por un dólar", Sloane cuenta 100 clientes.', "100"]),
        (1, ['"Quoted" at the start, ""doubled"" inside', "7"]),
        (2, ["Clive opens a box.\u2028It holds 6 balls.", "6"]),
        (4, ["one", "two", "three"]),
    ]


def test_a_line_that_is_not_a_question_a_tab_and_an_integer_is_unusable():
    for fields, why in [
        (["How many?"], "the line holds 0 tabs, not the one between the question and the final answer"),
        (["How", "many?", "18"], "the line holds 2 tabs, not the one between the question and the final answer"),
        ([" ", "18"], "the question is empty"),
        (["How many?", "18.5"], "the final answer '18.5' is not an integer"),
        (["How many?", "eighteen"], "the final answer 'eighteen' is not an integer"),
        (["How many?", " 18"], "the final answer ' 18' is not an integer"),
        (["How many?", "1_000"], "the final answer '1_000' is not an integer"),
        (["How many?", ""], "the final answer '' is not an integer"),
    ]:
        with pytest.raises(mgsm.Unusable, match=re.escape(why)):
            mgsm.question_and_final_answer(fields)
    assert mgsm.question_and_final_answer(["Janet’s ducks lay 16 eggs. How many?", "18"]) == (
        "Janet’s ducks lay 16 eggs. How many?",
        "18",
    )
    assert mgsm.question_and_final_answer(["How far below?", "-3"]) == ("How far below?", "-3")


def test_a_final_answer_whose_thousands_are_grouped_by_commas_is_an_integer_kept_as_written():
    for final_answer in ["2,125", "114,200", "276,000", "1,000,000", "-5,600"]:
        assert mgsm.question_and_final_answer(["How much?", final_answer]) == ("How much?", final_answer)
    for final_answer in ["2,12", "21,25", "1234,567", ",125", "2,125,", "2,,125"]:
        with pytest.raises(mgsm.Unusable, match="is not an integer"):
            mgsm.question_and_final_answer(["How much?", final_answer])


JA = "ジャネットのアヒルは1日に16個の卵を生みます。彼女は毎日市場でいくら手に入れていますか？"
JA_ORIGIN = {
    "dataset": "mgsm",
    "source": "github:google-research/url-nlp@3622039cf51f7eeffa58b957332a8e8c337981d7",
    "sha256": "59a2b50debe77981fd784cb3b2bef1505e3abf2a37116dc9d7a366ab029b4637",
    "file": "mgsm/mgsm_ja.tsv",
    "row": 0,
    "license": "CC-BY-4.0",
}
JA_REQUEST = {"messages": [{"role": "user", "content": JA}]}


def test_a_render_case_is_the_question_as_one_user_turn_with_its_origin():
    sets = mgsm.language_sets({"ja": tsv([f"{JA}\t18"])})
    assert sets[("render", "mgsm-ja")] == [
        {"name": "mgsm-ja-0", "request": JA_REQUEST, "notes": "MGSM ja row 0", "origin": JA_ORIGIN}
    ]
    [line] = sets[("render", "mgsm-ja")]
    assert list(line) == ["name", "request", "notes", "origin"]
    assert list(line["origin"]) == ["dataset", "source", "sha256", "file", "row", "license"]


def test_the_content_parse_case_holds_the_final_answer_as_written_as_its_content():
    sets = mgsm.language_sets({"ja": tsv([f"{JA}\t18", f"{JA}\t2,125"])})
    tail = {"notes": "MGSM ja row 1", "origin": {**JA_ORIGIN, "row": 1}}
    assert sets[("parse", "mgsm-ja-content")] == [
        {
            "name": "mgsm-ja-content-0",
            "request": JA_REQUEST,
            "message": {"content": "18"},
            "notes": "MGSM ja row 0",
            "origin": JA_ORIGIN,
        },
        {"name": "mgsm-ja-content-1", "request": JA_REQUEST, "message": {"content": "2,125"}, **tail},
    ]
    assert list(sets[("parse", "mgsm-ja-content")][0]) == ["name", "request", "message", "notes", "origin"]


def test_english_gets_a_content_set_and_no_render_set_since_its_questions_are_gsm8k_test_questions():
    sets = mgsm.language_sets({"en": tsv(["How many eggs does she sell?\t18"]), "ja": tsv([f"{JA}\t18"])})
    assert list(sets) == [("parse", "mgsm-en-content"), ("render", "mgsm-ja"), ("parse", "mgsm-ja-content")]
    [line] = sets[("parse", "mgsm-en-content")]
    assert (line["name"], line["message"]) == ("mgsm-en-content-0", {"content": "18"})


def test_rows_that_cannot_become_a_case_are_left_out_of_both_sets_of_their_language_with_their_reason():
    skipped: list[tuple[str, str]] = []
    ja = tsv([f"{JA}\t18", "How\tmany?\t18", f"{JA}\t18 dollars", f"{JA}\t18"])
    sets = mgsm.language_sets({"ja": ja, "th": tsv(["\t5", "มีกี่ลูก\t5"])}, skipped=skipped)
    assert list(sets) == [
        ("render", "mgsm-ja"),
        ("parse", "mgsm-ja-content"),
        ("render", "mgsm-th"),
        ("parse", "mgsm-th-content"),
    ]
    for key in [("render", "mgsm-ja"), ("parse", "mgsm-ja-content")]:
        assert [line["origin"]["row"] for line in sets[key]] == [0, 3]
    assert [line["name"] for line in sets[("parse", "mgsm-th-content")]] == ["mgsm-th-content-1"]
    [th] = sets[("render", "mgsm-th")]
    assert (th["origin"]["file"], th["origin"]["sha256"]) == (
        "mgsm/mgsm_th.tsv",
        "f3932dc5ad8e9d0ea82b017adc1e1461dd647af861e7166d6741986602a0cfd6",
    )
    assert skipped == [
        ("ja row 1", "the line holds 2 tabs, not the one between the question and the final answer"),
        ("ja row 2", "the final answer '18 dollars' is not an integer"),
        ("th row 0", "the question is empty"),
    ]


EN_QUESTION = (
    "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. "
    "How many tennis balls does he have now?"
)
EN_SOLUTION = "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11."
EN_FINAL = "The answer is 11."
EN_1 = {"q": "Question: " + EN_QUESTION, "a": "Step-by-Step Answer: " + EN_SOLUTION + " " + EN_FINAL}
JA_QUESTION = (
    "ロジャーは5個のテニスボールがあります。テニスボールの缶を2つ追加で買います。"
    "彼は今いくつのテニスボールがありますか？"
)
JA_SOLUTION = (
    "ロジャーは最初5個のボールがありました。テニスボール3個入りの缶が2つあれば、テニスボールは6個あります。5+6=11。"
)
JA_FINAL = "答えは11です。"
JA_1 = {"q": "問題：" + JA_QUESTION, "a": "ステップごとの答え：" + JA_SOLUTION + JA_FINAL}


def exemplars_py(exemplars: dict[str, dict[str, dict[str, str]]], final_answers: list[int], before: str = "") -> bytes:
    """An exemplars.py as the repository writes it: literals, with non-ASCII text as escapes and each string in two."""
    lines = ['"""Prompts for mgsm."""', before, f"EXEMPLAR_NUMBER_ANSWERS = {final_answers!r}", "MGSM_EXEMPLARS = {"]
    for lang, items in exemplars.items():
        lines.append(f"    {lang!r}: {{")
        for key, exemplar in items.items():
            fields = ", ".join(f"{name!r}: {ascii(text[:9])}\n {ascii(text[9:])}" for name, text in exemplar.items())
            lines.append(f"        {key!r}: {{{fields}}},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines).encode("utf-8")


def test_exemplars_py_is_read_as_literals_and_never_run(tmp_path):
    ran = tmp_path / "ran"
    source = exemplars_py({"en": {"1": EN_1}, "ja": {"1": JA_1}}, [11], before=f"open({str(ran)!r}, 'w').write('ran')")
    assert b"\\u" in source
    assert mgsm.read_exemplars(source) == ({"en": {"1": EN_1}, "ja": {"1": JA_1}}, [11])
    assert not ran.exists()
    with pytest.raises(ValueError):
        mgsm.read_exemplars(b"EXEMPLAR_NUMBER_ANSWERS = [11]\nMGSM_EXEMPLARS = dict(en={})\n")
    with pytest.raises(ValueError, match="no MGSM_EXEMPLARS"):
        mgsm.read_exemplars(b"EXEMPLAR_NUMBER_ANSWERS = [11]\n")


DE_QUESTION = "Es waren neun Computer im Serverraum. Wie viele Computer sind jetzt im Serverraum?"
DE_SOLUTION = (
    "Von Montag bis Donnerstag sind es 4 Tage. Am Anfang waren es 9 Computer, also sind es jetzt 9 + 20 =29 Computer."
)
DE_FINAL = "Die Antwort lautet 29."
DE_2 = {"q": "Frage: " + DE_QUESTION, "a": "Schritt-für-Schritt-Antwort: " + DE_SOLUTION + " " + DE_FINAL}
TH_QUESTION = "โรเจอร์มีลูกเทนนิส 5 ลูก เขาซื้อลูกเทนนิสเพิ่มอีก 2 กระป๋อง ตอนนี้เขามีลูกเทนนิสกี่ลูก"
TH_SOLUTION = "โรเจอร์เริ่มด้วยการมีลูกเทนนิส 5 ลูก กระป๋อง 2 ใบที่มีลูกเทนนิสใบละ 3 ลูก ดังนั้น 5 + 6 = 11"
TH_FINAL = "คำตอบคือ 11"
TH_1 = {"q": "โจทย์: " + TH_QUESTION, "a": "คำตอบทีละขั้นตอน: " + TH_SOLUTION + " " + TH_FINAL}
ZWNJ = chr(0x200C)  # the zero-width non-joiner, which Telugu text carries
TE_QUESTION = f"లియా వద్ద 32 చాక్లెట్{ZWNJ}లు మరియు ఆమె సోదరి వద్ద 42 ఉన్నాయి. వారి వద్ద ఎన్ని పీస్{ZWNJ}లు మిగిలి ఉన్నాయి?"
TE_SOLUTION = f"అందువల్ల వారి వద్ద మొత్తం 74-35= 39 చాక్లెట్{ZWNJ}లు ఉన్నాయి."
TE_FINAL = f"సమాధానం 39.చాక్లెట్{ZWNJ}లుచాక్లెట్{ZWNJ}లు"  # as the pinned file writes it: words follow the sentence
TE_3 = {"q": "ప్రశ్న: " + TE_QUESTION, "a": "దశలవారీగా సమాధానం: " + TE_SOLUTION + " " + TE_FINAL}


def test_an_exemplar_splits_into_its_question_solution_and_final_sentence_by_its_language_wording():
    assert mgsm.split_exemplar("en", EN_1, 11) == (EN_QUESTION, EN_SOLUTION, EN_FINAL)
    assert mgsm.split_exemplar("ja", JA_1, 11) == (JA_QUESTION, JA_SOLUTION, JA_FINAL)
    assert mgsm.split_exemplar("de", DE_2, 29) == (DE_QUESTION, DE_SOLUTION, DE_FINAL)
    assert mgsm.split_exemplar("th", TH_1, 11) == (TH_QUESTION, TH_SOLUTION, TH_FINAL)
    assert mgsm.split_exemplar("te", TE_3, 39) == (TE_QUESTION, TE_SOLUTION, TE_FINAL)


def test_the_final_sentence_is_the_last_one_its_opening_words_start():
    solution = "The answer is not 5 + 5 = 10. 5 + 6 = 11."
    exemplar = dict(EN_1, a=f"Step-by-Step Answer: {solution} {EN_FINAL}")
    assert mgsm.split_exemplar("en", exemplar, 11) == (EN_QUESTION, solution, EN_FINAL)


def test_an_exemplar_that_does_not_fit_its_language_wording_is_unusable():
    answer = "Step-by-Step Answer: "
    for lang, exemplar, final_answer, why in [
        ("xx", EN_1, 11, "no wording is known for the language 'xx'"),
        ("en", {"a": EN_1["a"]}, 11, "the exemplar is not {'q': question, 'a': answer}"),
        ("en", dict(EN_1, q=EN_QUESTION), 11, "the question does not start with 'Question: '"),
        ("en", dict(EN_1, q="Question: "), 11, "the question is empty"),
        ("en", dict(EN_1, a=EN_SOLUTION + " " + EN_FINAL), 11, "answer does not start with 'Step-by-Step Answer: '"),
        ("en", dict(EN_1, a=answer + EN_SOLUTION), 11, "answer has no final sentence opening with 'The answer is '"),
        ("en", dict(EN_1, a=answer + EN_FINAL), 11, "the answer has no worked solution before its final sentence"),
        ("en", EN_1, 12, "the final sentence 'The answer is 11.' does not state the exemplar's final answer 12"),
        (
            "en",
            dict(EN_1, a=f"{answer}5 + 6 = 11. The answer is 111."),
            11,
            "does not state the exemplar's final answer 11",
        ),
        ("en", dict(EN_1, a=f"{answer}5 + 6 = 11. The answer is eleven."), 11, "does not state the exemplar's"),
    ]:
        with pytest.raises(mgsm.Unusable, match=re.escape(why)):
            mgsm.split_exemplar(lang, exemplar, final_answer)


EXEMPLAR_ORIGIN = {
    "dataset": "mgsm",
    "source": "github:google-research/url-nlp@3622039cf51f7eeffa58b957332a8e8c337981d7",
    "sha256": "239dda1557bb2ba76b71e5ef744ddd0b454da0b453e5f8b909498ceff690a919",
    "file": "mgsm/exemplars.py",
    "row": "MGSM_EXEMPLARS['ja']['1']",
    "license": "CC-BY-4.0",
}


def test_an_exemplar_parse_case_holds_the_solution_as_reasoning_and_the_final_sentence_as_content():
    [line] = mgsm.exemplar_set(exemplars_py({"ja": {"1": JA_1}}, [11]))
    assert line == {
        "name": "mgsm-exemplars-ja-1",
        "request": {"messages": [{"role": "user", "content": JA_QUESTION}]},
        "message": {"reasoning_content": JA_SOLUTION, "content": JA_FINAL},
        "notes": "MGSM ja exemplar 1",
        "origin": EXEMPLAR_ORIGIN,
    }
    assert list(line) == ["name", "request", "message", "notes", "origin"]
    assert list(line["message"]) == ["reasoning_content", "content"]
    assert list(line["origin"]) == ["dataset", "source", "sha256", "file", "row", "license"]


def test_an_exemplar_takes_the_final_answer_of_its_key_and_one_that_cannot_become_a_case_is_skipped_with_its_reason():
    skipped: list[tuple[str, str]] = []
    exemplars = {"en": {"1": EN_1, "2": EN_1}, "de": {"2": DE_2, "9": DE_2}, "te": {"3": TE_3}, "xx": {"1": EN_1}}
    lines = mgsm.exemplar_set(exemplars_py(exemplars, [11, 29, 39]), skipped=skipped)
    assert [line["name"] for line in lines] == ["mgsm-exemplars-en-1", "mgsm-exemplars-de-2", "mgsm-exemplars-te-3"]
    assert lines[2]["origin"]["row"] == "MGSM_EXEMPLARS['te']['3']"
    assert skipped == [
        ("exemplar en 2", "the final sentence 'The answer is 11.' does not state the exemplar's final answer 29"),
        ("exemplar de 9", "EXEMPLAR_NUMBER_ANSWERS has no final answer for the key '9'"),
        ("exemplar xx 1", "no wording is known for the language 'xx'"),
    ]


def test_written_sets_are_raw_unicode_that_reads_back_whole_and_a_rewrite_is_byte_identical(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "render").mkdir(parents=True)
    (corpus / "render" / "mgsm-old.jsonl").write_text("{}\n")
    (corpus / "render" / "gsm8k-test.jsonl").write_text("{}\n")
    ja_5 = dict(JA_1, q="問題：ロジャーは5個の\nテニスボールがあります。")  # as exemplar ja 5 is, with a line break

    def sets():
        return mgsm.build_sets({"te": tsv([f"{TE_QUESTION}\t39"])}, exemplars_py({"ja": {"1": ja_5}}, [11]))

    written = mgsm.write_sets(sets(), corpus, CC_BY)
    assert sorted(path.relative_to(corpus).as_posix() for path in written) == [
        "licenses/mgsm-LICENSE",
        "parse/mgsm-exemplars.jsonl",
        "parse/mgsm-te-content.jsonl",
        "render/mgsm-te.jsonl",
    ]
    assert not (corpus / "render" / "mgsm-old.jsonl").exists()
    assert (corpus / "render" / "gsm8k-test.jsonl").read_text() == "{}\n"
    first = {path: path.read_bytes() for path in written}
    assert TE_QUESTION.encode("utf-8") in first[corpus / "render" / "mgsm-te.jsonl"]
    [exemplar] = read_cases(corpus / "parse" / "mgsm-exemplars.jsonl")
    assert exemplar.request["messages"][0]["content"] == "ロジャーは5個の\nテニスボールがあります。"
    for path in [corpus / "render" / "mgsm-te.jsonl", corpus / "parse" / "mgsm-te-content.jsonl"]:
        assert [case.request["messages"][0]["content"] for case in read_cases(path)] == [TE_QUESTION]
    mgsm.write_sets(sets(), corpus, CC_BY)
    assert {path: path.read_bytes() for path in written} == first


def test_check_passes_on_a_fresh_import_and_names_each_set_file_that_differs(tmp_path):
    sets = mgsm.build_sets({"ja": tsv([f"{JA}\t18"])}, exemplars_py({"en": {"1": EN_1}}, [11]))
    corpus = tmp_path / "corpus"
    mgsm.write_sets(sets, corpus, CC_BY)
    assert mgsm.check_sets(sets, corpus, CC_BY) == []
    (corpus / "parse" / "mgsm-ja-content.jsonl").write_text("{}\n")
    (corpus / "render" / "mgsm-ja.jsonl").unlink()
    (corpus / "render" / "mgsm-stale.jsonl").write_text("{}\n")
    (corpus / "render" / "gsm8k-other.jsonl").write_text("{}\n")
    assert mgsm.check_sets(sets, corpus, CC_BY) == [
        f"{corpus / 'parse' / 'mgsm-ja-content.jsonl'}: differs from a fresh import",
        f"{corpus / 'render' / 'mgsm-ja.jsonl'}: missing",
        f"{corpus / 'render' / 'mgsm-stale.jsonl'}: no MGSM file writes it",
    ]


# Bengali as MGSM writes it, from the pinned files: U+09DC BENGALI LETTER RRA and U+09DF BENGALI LETTER YYA as one code
# point each, which NFC writes as two. BN_QUESTION is row 2's first sentence and its question; the exemplar is exemplar
# 8, whose final sentence gains the word for its cars so that it holds RRA too.
RRA, YYA = "\N{BENGALI LETTER RRA}", "\N{BENGALI LETTER YYA}"
BN_QUESTION = f"জোশ একটি বা{RRA}ি ফ্লিপ করার সিদ্ধান্ত নি{YYA}েছেন। তিনি কত ডলার লাভ করলেন?"
BN_EXEMPLAR_QUESTION = f"পার্কিং লটে যদি 3 টি গা{RRA}ি থাকে এবং আরও 2টি গা{RRA}ি আসে, তাহলে পার্কিং লটটিতে কতগুলি গা{RRA}ি আছে?"
BN_SOLUTION = f"শুরুতে সেখানে 3টি গা{RRA}ি ছিল, আরও 2টি গা{RRA}ি আসে, তাহলে বর্তমানে 3 +2 = 5টি গা{RRA}ি থাকা উচিৎ।"
BN_FINAL = f"উত্তর হল 5টি গা{RRA}ি।"


def test_a_case_is_named_by_its_row_not_by_its_place_among_the_rows_kept():
    sets = mgsm.language_sets({"ja": tsv(["How\tmany?\t18", f"{JA}\t18"])}, skipped=[])
    [render] = sets[("render", "mgsm-ja")]
    [content] = sets[("parse", "mgsm-ja-content")]
    assert (render["name"], render["notes"], render["origin"]["row"]) == ("mgsm-ja-1", "MGSM ja row 1", 1)
    assert (content["name"], content["notes"], content["origin"]["row"]) == ("mgsm-ja-content-1", "MGSM ja row 1", 1)


def test_text_that_is_not_in_nfc_reaches_the_written_sets_as_written(tmp_path):
    for text in [BN_QUESTION, BN_EXEMPLAR_QUESTION, BN_SOLUTION, BN_FINAL]:
        assert not unicodedata.is_normalized("NFC", text)
    question_label, answer_label, _ = mgsm.WORDING["bn"]
    exemplar = {"q": question_label + BN_EXEMPLAR_QUESTION, "a": f"{answer_label}{BN_SOLUTION} {BN_FINAL}"}
    exemplars = exemplars_py({"bn": {"8": exemplar}}, [11, 29, 39, 9, 33, 8, 8, 5])
    mgsm.write_sets(mgsm.build_sets({"bn": tsv([f"{BN_QUESTION}\t70000"])}, exemplars), tmp_path, CC_BY)
    for kind, name in [("render", "mgsm-bn"), ("parse", "mgsm-bn-content")]:
        [case] = read_cases(tmp_path / kind / f"{name}.jsonl")
        assert case.request == {"messages": [{"role": "user", "content": BN_QUESTION}]}
    [case] = read_cases(tmp_path / "parse" / "mgsm-exemplars.jsonl")
    assert case.request == {"messages": [{"role": "user", "content": BN_EXEMPLAR_QUESTION}]}
    assert case.message == {"reasoning_content": BN_SOLUTION, "content": BN_FINAL}


def test_a_line_of_whitespace_alone_is_a_row_named_with_its_reason_not_a_line_dropped():
    skipped: list[tuple[str, str]] = []
    sets = mgsm.language_sets({"ja": tsv([f"{JA}\t18", "   ", "\t", f"{JA}\t5"])}, skipped=skipped)
    assert [line["origin"]["row"] for line in sets[("parse", "mgsm-ja-content")]] == [0, 3]
    assert skipped == [
        ("ja row 1", "the line holds 0 tabs, not the one between the question and the final answer"),
        ("ja row 2", "the question is empty"),
    ]


def test_whitespace_at_the_edges_of_a_question_or_an_exemplar_is_kept_as_written():
    question = f"  {JA} "
    sets = mgsm.language_sets({"ja": tsv([f"{question}\t18"])})
    for key in [("render", "mgsm-ja"), ("parse", "mgsm-ja-content")]:
        [line] = sets[key]
        assert line["request"] == {"messages": [{"role": "user", "content": question}]}
    exemplar = {"q": f"Question:  {EN_QUESTION}\n", "a": f"Step-by-Step Answer:  {EN_SOLUTION} {EN_FINAL} "}
    [line] = mgsm.exemplar_set(exemplars_py({"en": {"1": exemplar}}, [11]))
    assert line["request"] == {"messages": [{"role": "user", "content": f" {EN_QUESTION}\n"}]}
    assert line["message"] == {"reasoning_content": f" {EN_SOLUTION}", "content": f"{EN_FINAL} "}


# 18 in Bengali, Devanagari, Arabic-Indic, fullwidth and Thai digits, and an ASCII 1 before a Bengali 8: int() reads
# each as 18, and the regular expression \d matches each.
OTHER_DIGITS = ["\u09e7\u09ee", "\u0967\u096e", "\u0661\u0668", "\uff11\uff18", "\u0e51\u0e58", "1\u09ee"]


def test_a_final_answer_in_digits_other_than_ascii_is_unusable_in_a_row_or_a_final_sentence():
    for final_answer in OTHER_DIGITS:
        assert int(final_answer) == 18
        with pytest.raises(mgsm.Unusable, match="is not an integer"):
            mgsm.question_and_final_answer(["How many?", final_answer])
    eleven = "\u09e7\u09e7"  # 11 in Bengali digits
    exemplar = dict(EN_1, a=f"Step-by-Step Answer: {EN_SOLUTION} The answer is {eleven}.")
    with pytest.raises(mgsm.Unusable, match="does not state the exemplar's final answer 11"):
        mgsm.split_exemplar("en", exemplar, 11)


LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def serve(tmp_path, monkeypatch, files: dict[str, bytes]) -> None:
    """Stand in for ``github.fetch``: each pinned path is served from ``files``, and nothing is downloaded."""
    pins = {"mgsm/LICENSE": mgsm.LICENSE_SHA256, "mgsm/exemplars.py": mgsm.EXEMPLARS_SHA256}
    pins.update({f"mgsm/mgsm_{lang}.tsv": sha256 for lang, sha256 in mgsm.SHA256.items()})

    def fetch(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit) == ("google-research", "url-nlp", mgsm.COMMIT)
        assert (sha256, cache) == (pins[path], tmp_path)
        served = tmp_path / "served" / path
        served.parent.mkdir(parents=True, exist_ok=True)
        served.write_bytes(files[path])
        return served

    monkeypatch.setattr(github, "fetch", fetch)


def pinned_files(license_text: bytes = CC_BY) -> dict[str, bytes]:
    files = {f"mgsm/mgsm_{lang}.tsv": tsv([f"Question in {lang}?\t18"]) for lang in LANGUAGES}
    files["mgsm/mgsm_ja.tsv"] = tsv([f"{JA}\t18", f"{JA}\t18 dollars"])
    files["mgsm/exemplars.py"] = exemplars_py({"en": {"1": EN_1, "2": EN_1}}, [11, 29])
    files["mgsm/LICENSE"] = license_text
    return files


def test_the_command_writes_then_checks_and_names_what_it_leaves_out(tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, pinned_files())
    corpus = tmp_path / "corpus"
    argv = ["import", "mgsm", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main([*argv, "--check"]) == 1
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert sorted(path.name for path in (corpus / "render").iterdir()) == [
        f"mgsm-{lang}.jsonl" for lang in LANGUAGES if lang != "en"
    ]
    assert sorted(path.name for path in (corpus / "parse").iterdir()) == sorted(
        ["mgsm-exemplars.jsonl", *(f"mgsm-{lang}-content.jsonl" for lang in LANGUAGES)]
    )
    assert f"{corpus / 'render' / 'mgsm-ja.jsonl'}: 1 cases" in out
    assert f"{corpus / 'parse' / 'mgsm-ja-content.jsonl'}: 1 cases, 1 distinct messages" in out
    assert f"{corpus / 'parse' / 'mgsm-exemplars.jsonl'}: 1 cases, 1 distinct messages" in out
    assert f"{corpus}: 22 cases in the 22 MGSM sets, 0 left out as repeats, 2 distinct messages" in out
    assert "no case for 1 row(s) (ja row 1): the final answer '18 dollars' is not an integer" in out
    assert (
        "no case for 1 row(s) (exemplar en 2): "
        "the final sentence 'The answer is 11.' does not state the exemplar's final answer 29"
    ) in out
    assert f"{corpus}: the MGSM sets equal a fresh import of {mgsm.SOURCE}" in out


def test_the_command_leaves_out_cases_that_repeat_earlier_ones_and_names_what_they_repeat(
    tmp_path, monkeypatch, capsys
):
    # Rows 1 and 2 ask what row 0 asks, so their render cases repeat row 0's; of their content cases only row 1's does,
    # since row 2's message differs.
    files = pinned_files()
    files["mgsm/mgsm_ja.tsv"] = tsv([f"{JA}\t18", f"{JA}\t18", f"{JA}\t5"])
    serve(tmp_path, monkeypatch, files)
    corpus = tmp_path / "corpus"
    argv = ["import", "mgsm", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    out = capsys.readouterr().out.splitlines()
    repeated = {"mgsm-ja-1": "mgsm-ja-0", "mgsm-ja-2": "mgsm-ja-0", "mgsm-ja-content-1": "mgsm-ja-content-0"}
    for name, first in repeated.items():
        assert f"no case {name}: it repeats {first}" in out
    assert f"{corpus / 'render' / 'mgsm-ja.jsonl'}: 1 cases, 2 left out as repeats" in out
    assert f"{corpus / 'parse' / 'mgsm-ja-content.jsonl'}: 2 cases, 1 left out as repeats, 2 distinct messages" in out
    assert f"{corpus}: 23 cases in the 22 MGSM sets, 3 left out as repeats, 3 distinct messages" in out
    assert [case.name for case in read_cases(corpus / "render" / "mgsm-ja.jsonl")] == ["mgsm-ja-0"]
    assert [case.name for case in read_cases(corpus / "parse" / "mgsm-ja-content.jsonl")] == [
        "mgsm-ja-content-0",
        "mgsm-ja-content-2",
    ]


def test_the_command_writes_the_pinned_license_next_to_the_sets(tmp_path, monkeypatch):
    serve(tmp_path, monkeypatch, pinned_files())
    corpus = tmp_path / "corpus"
    assert main(["import", "mgsm", "--corpus", str(corpus), "--cache", str(tmp_path)]) == 0
    assert (corpus / "licenses" / "mgsm-LICENSE").read_bytes() == CC_BY


def test_check_names_the_license_copy_when_it_is_missing_or_differs(tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, pinned_files())
    corpus = tmp_path / "corpus"
    copy = corpus / "licenses" / "mgsm-LICENSE"
    argv = ["import", "mgsm", "--corpus", str(corpus), "--cache", str(tmp_path)]
    assert main(argv) == 0
    copy.unlink()
    assert main([*argv, "--check"]) == 1
    copy.write_bytes(CC_BY + b"Additional terms apply.\n")
    assert main([*argv, "--check"]) == 1
    copy.write_bytes(CC_BY)
    assert main([*argv, "--check"]) == 0
    assert capsys.readouterr().err.splitlines() == [f"{copy}: missing", f"{copy}: differs from a fresh import"]


def test_the_command_refuses_a_license_that_is_not_cc_by_4_and_writes_nothing(tmp_path, monkeypatch):
    share_alike = b"Creative Commons Attribution-ShareAlike 4.0 International Public License (CC-BY-SA)\n"
    serve(tmp_path, monkeypatch, pinned_files(share_alike))
    with pytest.raises(ValueError, match="not the CC-BY-4.0 license"):
        main(["import", "mgsm", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path)])
    assert not (tmp_path / "corpus").exists()
