import hashlib
import json
import struct

import pytest
from audio_files import flac, m4a, mp3, ogg_page, opus, vorbis, wav

from bellwether import audio
from bellwether.cli import NOT_IMPLEMENTED, main

LFS_POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize {}\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the catalogue ---


def test_the_catalogue_reads_each_container_from_its_headers():
    assert audio.catalogue(flac()) == audio.Catalogue("flac", "flac", 16000, 1, 16, 3.505)
    assert audio.catalogue(flac(48000, 2, 24, 0)) == audio.Catalogue("flac", "flac", 48000, 2, 24, None)
    assert audio.catalogue(wav()) == audio.Catalogue("wav", "pcm_s16le", 16000, 1, 16, 1.0)
    assert audio.catalogue(wav(16000, 1, 32, tag=3, data_bytes=64000)) == audio.Catalogue(
        "wav", "pcm_f32le", 16000, 1, 32, 1.0
    )
    assert audio.catalogue(wav(8000, 1, 8, tag=7, data_bytes=8000)) == audio.Catalogue(
        "wav", "pcm_mulaw", 8000, 1, None, 1.0
    )
    assert audio.catalogue(opus()) == audio.Catalogue("ogg", "opus", 48000, 1, None, 2.0)
    assert audio.catalogue(vorbis()) == audio.Catalogue("ogg", "vorbis", 44100, 2, None, 1.0)
    assert audio.catalogue(mp3()) == audio.Catalogue("mp3", "mp3", 44100, 2, None, 2.612245)
    assert audio.catalogue(mp3(frames=None, mono=True, id3=True)) == audio.Catalogue(
        "mp3", "mp3", 44100, 1, None, round(417 * 3 * 8 / 128000, 6)
    )
    assert audio.catalogue(m4a()) == audio.Catalogue("m4a", "aac", 44100, 2, None, 3.0)


def test_other_bytes_and_broken_headers_are_refused():
    with pytest.raises(audio.NotAudio, match="not a FLAC, WAV, Ogg, MPEG audio, MP4 or M4A file"):
        audio.catalogue(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    with pytest.raises(audio.NotAudio, match="headers are not whole"):
        audio.catalogue(b"fLaC" + bytes([0x80, 0, 0, 34]) + b"\x00" * 10)
    with pytest.raises(audio.NotAudio, match="not an Opus, Vorbis or FLAC header"):
        audio.catalogue(ogg_page(b"\x80theora" + b"\x00" * 30, 0, 2))
    with pytest.raises(audio.NotAudio, match="no fmt chunk"):
        audio.catalogue(b"RIFF" + struct.pack("<I", 12) + b"WAVE" + b"data" + struct.pack("<I", 0))


# --- the check ---


def line(name: str, dataset: str, data: bytes, ext: str, **override) -> dict:
    what = audio.catalogue(data)
    fields = {
        "name": name,
        "path": f"audios/{dataset}/{name}.{ext}",
        "bytes": len(data),
        "sha256": sha(data),
        "container": what.container,
        "codec": what.codec,
        "sample_rate": what.sample_rate,
        "channels": what.channels,
        "bits": what.bits,
        "duration_s": what.duration_s,
        "notes": f"{dataset} {name}",
        "origin": {"dataset": dataset, "source": "made for the tests", "license": "Apache-2.0"},
    }
    fields.update(override)
    return fields


def write_set(corpus, dataset: str, lines: list[dict]) -> None:
    (corpus / "audio").mkdir(parents=True, exist_ok=True)
    (corpus / "audio" / f"{dataset}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    (corpus / "licenses").mkdir(exist_ok=True)
    (corpus / "licenses" / f"{dataset}-LICENSE").write_text("the licence\n")


def test_a_corpus_of_content_and_pointers_passes_and_the_run_reports_it(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    a, b = flac(), opus()
    (corpus / "audios" / "speech").mkdir(parents=True)
    (corpus / "audios" / "speech" / "speech-00001-aaaaaaaa.flac").write_bytes(a)
    (corpus / "audios" / "speech" / "speech-00002-bbbbbbbb.opus").write_text(LFS_POINTER.format(sha(b), len(b)))
    write_set(
        corpus,
        "speech",
        [line("speech-00001-aaaaaaaa", "speech", a, "flac"), line("speech-00002-bbbbbbbb", "speech", b, "opus")],
    )
    assert audio.problems(corpus) == []
    assert main(["audio", "check", "--corpus", str(corpus)]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{corpus}: 2 clips in 1 sets, {len(a) + len(b)} bytes"]


def test_every_kind_of_disagreement_is_named(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    a, b, c = flac(), wav(), vorbis()
    (corpus / "audios" / "speech").mkdir(parents=True)
    (corpus / "audios" / "speech" / "speech-00001-aaaaaaaa.flac").write_bytes(a)
    longer = b + b"\x00"
    (corpus / "audios" / "speech" / "speech-00002-bbbbbbbb.wav").write_bytes(longer)
    (corpus / "audios" / "speech" / "speech-00004-dddddddd.wav").write_bytes(wav(8000))
    (corpus / "audios" / "speech" / "unnamed.flac").write_bytes(a)
    (corpus / "audios" / "stray").mkdir()
    lines = [
        line("speech-00001-aaaaaaaa", "speech", a, "flac", sample_rate=8000),
        line("speech-00002-bbbbbbbb", "speech", b, "wav"),
        line("speech-00003-cccccccc", "speech", c, "ogg"),
        line("speech-00004-dddddddd", "speech", wav(8000), "wav", path="elsewhere/speech-00004-dddddddd.wav"),
        line("speech-00001-aaaaaaaa", "speech", a, "flac", path="audios/speech/speech-00001-aaaaaaaa.flac"),
    ]
    write_set(corpus, "speech", lines)
    (corpus / "licenses" / "speech-LICENSE").unlink()
    assert main(["audio", "check", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err
    set_file = corpus / "audio" / "speech.jsonl"
    assert f"{set_file}:1: the headers say" in err and "'sample_rate': 16000" in err and "'sample_rate': 8000" in err
    assert f"audios/speech/speech-00002-bbbbbbbb.wav: sha256 {sha(longer)}, the set says {sha(b)}" in err
    assert f"audios/speech/speech-00002-bbbbbbbb.wav: {len(b) + 1} bytes, the set says {len(b)}" in err
    assert "audios/speech/speech-00003-cccccccc.ogg: the set names it, the checkout lacks it" in err
    assert f"{set_file}:4: path elsewhere/speech-00004-dddddddd.wav is not a file directly under audios/speech/" in err
    assert f"{set_file}:5: name speech-00001-aaaaaaaa is already used in {set_file}" in err
    assert f"{set_file}:5: path audios/speech/speech-00001-aaaaaaaa.flac is already used in {set_file}" in err
    assert "audios/speech/speech-00004-dddddddd.wav: no line of speech.jsonl names it" in err
    assert "audios/speech/unnamed.flac: no line of speech.jsonl names it" in err
    assert "audios/stray: no set audio/stray.jsonl names it" in err
    assert "licenses/speech-LICENSE: the licence copy of speech.jsonl is missing" in err
    assert err.rstrip().endswith("bellwether audio check: 11 problem(s)")


def test_a_lines_fields_a_nested_file_and_a_path_outside_the_dataset_are_named_on_pointers_alone(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    a, b, c = flac(), opus(), wav()
    (corpus / "audios" / "speech").mkdir(parents=True)
    pointers = {"speech-00001-aaaaaaaa.flac": a, "speech-00002-bbbbbbbb.opus": b, "speech-00003-cccccccc.wav": c}
    for name, data in pointers.items():
        (corpus / "audios" / "speech" / name).write_text(LFS_POINTER.format(sha(data), len(data)))
    first = line("speech-00001-aaaaaaaa", "speech", a, "flac")
    del first["bits"]  # a field missing from a line, which the pointer check cannot read from the file
    # A field of another type, and a bool, which Python counts as an int and the check does not.
    second = line("speech-00002-bbbbbbbb", "speech", b, "opus", sample_rate="48000", duration_s=True)
    outside = "audios/speech/../speech/speech-00003-cccccccc.wav"  # not directly under the dataset directory
    third = line("speech-00003-cccccccc", "speech", c, "wav", path=outside)
    write_set(corpus, "speech", [first, second, third])
    (corpus / "audios" / "speech" / "nested").mkdir()
    (corpus / "audios" / "speech" / "nested" / "extra.flac").write_bytes(a)  # unnamed, one directory down
    assert main(["audio", "check", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err
    set_file = corpus / "audio" / "speech.jsonl"
    assert f"{set_file}:1: the fields are [" in err and "a line carries [" in err
    assert f"{set_file}:2: sample_rate is '48000', not int or None" in err
    assert f"{set_file}:2: duration_s is True, not int or float or None" in err
    assert f"{set_file}:3: path {outside} is not a file directly under audios/speech/" in err
    assert "audios/speech/nested/extra.flac: no line of speech.jsonl names it" in err
    # The third clip's file: its line's path was not it.
    assert "audios/speech/speech-00003-cccccccc.wav: no line of speech.jsonl names it" in err
    assert err.rstrip().endswith("bellwether audio check: 6 problem(s)")


def test_an_empty_corpus_passes_and_audio_without_a_subcommand_prints_help_and_does_nothing(tmp_path, capsys):
    assert main(["audio", "check", "--corpus", str(tmp_path / "none")]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{tmp_path / 'none'}: 0 clips in 0 sets, 0 bytes"]
    assert main(["audio"]) == NOT_IMPLEMENTED
    assert "check" in capsys.readouterr().out
