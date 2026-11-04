import hashlib
import json

from audio_files import flac, mp3, opus, vorbis, wav

from bellwether.importers import audio_sets, image_sets


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def found(clips: list[bytes]) -> list[audio_sets.Found]:
    return [
        audio_sets.Found(data, {"dataset": "speech", "source": "the tests", "license": "CC0-1.0"}, f"the tests row {n}")
        for n, data in enumerate(clips)
    ]


def test_write_names_the_clips_by_container_and_leaves_repeats_and_other_bytes_out(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    a, b, c, d, e = flac(), opus(), wav(), vorbis(), mp3()
    written, summary = audio_sets.write(
        "speech", found([a, b, a, b"not a clip", c, d, e]), corpus, {"licenses/speech-LICENSE": b"CC0\n"}
    )
    lines = audio_sets.read_set(corpus / "audio" / "speech.jsonl")
    assert [line["path"] for line in lines] == [
        f"audios/speech/speech-00001-{sha(a)[:8]}.flac",
        f"audios/speech/speech-00002-{sha(b)[:8]}.opus",
        f"audios/speech/speech-00003-{sha(c)[:8]}.wav",
        f"audios/speech/speech-00004-{sha(d)[:8]}.ogg",
        f"audios/speech/speech-00005-{sha(e)[:8]}.mp3",
    ]
    assert list(lines[0]) == list(audio_sets.audio.FIELDS)
    assert lines[1]["codec"] == "opus" and lines[1]["sample_rate"] == 48000 and lines[1]["duration_s"] == 2.0
    assert (corpus / lines[2]["path"]).read_bytes() == c
    assert (corpus / "licenses" / "speech-LICENSE").read_bytes() == b"CC0\n"
    assert summary.lines == 5 and summary.bytes == len(a) + len(b) + len(c) + len(d) + len(e)
    assert summary.repeats == [("the tests row 2", f"speech-00001-{sha(a)[:8]}")]
    assert summary.skipped == [("the tests row 3", "not a FLAC, WAV, Ogg, MPEG audio, MP4 or M4A file")]
    assert set(written) == {corpus / line["path"] for line in lines} | {
        corpus / "audio" / "speech.jsonl",
        corpus / "licenses" / "speech-LICENSE",
    }
    audio_sets.report("speech", summary, corpus)
    assert capsys.readouterr().out.splitlines() == [
        f"no line for the tests row 2: it repeats speech-00001-{sha(a)[:8]}",
        "no line for the tests row 3: not a FLAC, WAV, Ogg, MPEG audio, MP4 or M4A file",
        f"{corpus / 'audio' / 'speech.jsonl'}: 5 clips, {summary.bytes} bytes, "
        "1 left out as repeats, 1 rows with no clip",
    ]


def test_check_passes_on_what_write_wrote_and_names_every_kind_of_difference(tmp_path):
    corpus = tmp_path / "corpus"
    a, b, c = flac(), opus(), wav()
    copies = {"licenses/speech-LICENSE": b"CC0\n"}
    audio_sets.write("speech", found([a, b, c]), corpus, copies)
    problems, summary = audio_sets.check("speech", found([a, b, c]), corpus, copies)
    assert problems == [] and summary.lines == 3
    # A pointer in place of a file's content passes the identity check.
    path = corpus / f"audios/speech/speech-00002-{sha(b)[:8]}.opus"
    path.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha(b)}\nsize {len(b)}\n")
    assert audio_sets.check("speech", found([a, b, c]), corpus, copies)[0] == []
    # A tampered line, a file with other bytes, a stray file, a changed licence copy and a shorter import.
    set_file = corpus / "audio" / "speech.jsonl"
    text = set_file.read_text().splitlines()
    tampered = json.loads(text[0])
    tampered["channels"] = 2
    text[0] = json.dumps(tampered, ensure_ascii=False)
    set_file.write_text("".join(line + "\n" for line in text))
    longer = c + b"\x00"
    (corpus / f"audios/speech/speech-00003-{sha(c)[:8]}.wav").write_bytes(longer)
    (corpus / "audios/speech/stray.flac").write_bytes(a)
    (corpus / "licenses" / "speech-LICENSE").write_bytes(b"other\n")
    problems, _ = audio_sets.check("speech", found([a, b]), corpus, copies)
    assert problems == [
        f"{set_file}:1: differs from a fresh import",
        f"{set_file}: 1 line(s) past the 2 a fresh import writes",
        f"{corpus / f'audios/speech/speech-00003-{sha(c)[:8]}.wav'}: no row of the import writes it",
        f"{corpus / 'audios/speech/stray.flac'}: no row of the import writes it",
        f"{corpus / 'licenses' / 'speech-LICENSE'}: differs from a fresh import",
    ]
    problems, _ = audio_sets.check("speech", found([a, b, c]), corpus, copies)
    assert (
        f"{corpus / f'audios/speech/speech-00003-{sha(c)[:8]}.wav'}: holds sha256 {sha(longer)}, "
        f"{len(c) + 1} bytes; a fresh import gives {sha(c)}"
    ) in problems


def test_a_second_write_removes_what_it_no_longer_writes_and_the_found_record_is_the_image_sets(tmp_path):
    corpus = tmp_path / "corpus"
    a, b = flac(), wav()
    audio_sets.write("speech", found([a, b]), corpus)
    audio_sets.write("speech", found([b]), corpus)
    assert sorted(p.name for p in (corpus / "audios" / "speech").iterdir()) == [f"speech-00001-{sha(b)[:8]}.wav"]
    assert audio_sets.Found is image_sets.Found and audio_sets.Summary is image_sets.Summary
