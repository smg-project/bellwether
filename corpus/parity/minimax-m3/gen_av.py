#!/usr/bin/env python3
# ruff: noqa: E501  (the recipe lines are kept as the commands they record)
"""Synthetic audio and video corpus (deterministic, with a manifest, every ffmpeg command recorded).

Audio is synthesized in numpy (a 440 Hz tone, a chirp and shaped noise, seeded) and written by
soundfile (WAV/FLAC) or ffmpeg (MP3, Ogg Vorbis, Opus, M4A/AAC). Video comes from ffmpeg's
`testsrc2` (deterministic) with the codecs, rates, durations and container quirks the brief names.

    python -I gen_av.py --out <corpus dir> --ffmpeg <ffmpeg> [--skip-long]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SEED = 20261105


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signal(seconds: float, rate: int, channels: int, kind: str = "tone") -> np.ndarray:
    n = max(int(round(seconds * rate)), 1)
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(SEED + rate + channels)
    if kind == "silence":
        base = np.zeros(n)
    elif kind == "clipping":
        base = np.clip(1.6 * np.sin(2 * np.pi * 440.0 * t), -1.0, 1.0)
    elif kind == "dc":
        base = 0.4 + 0.3 * np.sin(2 * np.pi * 440.0 * t)
    else:
        chirp = np.sin(2 * np.pi * (200.0 + 1500.0 * (t % 5.0) / 5.0) * t)
        base = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * chirp + 0.05 * rng.standard_normal(n)
        base = np.clip(base, -1.0, 1.0)
    if channels == 1:
        return base.astype(np.float32)
    chans = [base * (0.6 + 0.4 * (c % 2)) for c in range(channels)]
    return np.stack(chans, axis=1).astype(np.float32)


class Corpus:
    def __init__(self, out: Path, ffmpeg: str):
        self.out, self.ffmpeg, self.rows = out, ffmpeg, []
        (out / "audios").mkdir(parents=True, exist_ok=True)
        (out / "videos").mkdir(parents=True, exist_ok=True)

    def add(self, name: str, path: Path, modality: str, tags: dict, recipe: str) -> None:
        self.rows.append(
            {
                "name": name,
                "path": str(path.relative_to(self.out)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modality": modality,
                "tags": tags,
                "recipe": recipe,
            }
        )
        print(f"  {name}: {path.stat().st_size} B", file=sys.stderr)

    def fail(self, name: str, modality: str, tags: dict, recipe: str, err: str) -> None:
        self.rows.append(
            {"name": name, "path": None, "modality": modality, "tags": tags, "recipe": recipe, "error": err[:300]}
        )
        print(f"  {name}: FAILED {err[:160]}", file=sys.stderr)

    def ffmpeg_run(
        self, name: str, out_rel: str, args: list[str], modality: str, tags: dict, stdin: bytes | None = None
    ) -> Path | None:
        path = self.out / out_rel
        cmd = [self.ffmpeg, "-y", "-v", "error", *args, str(path)]
        proc = subprocess.run(cmd, input=stdin, capture_output=True)
        recipe = "ffmpeg " + " ".join(cmd[1:]).replace(str(path), "<out>")
        if proc.returncode != 0 or not path.exists() or path.stat().st_size == 0:
            self.fail(name, modality, tags, recipe, proc.stderr.decode(errors="replace"))
            return None
        self.add(name, path, modality, tags, recipe)
        return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    c = Corpus(out, args.ffmpeg)

    # ---------------- audio ----------------
    # WAV sample formats and rates (30 s mono; the format/rate axis)
    for subtype, bits in [("PCM_16", 16), ("PCM_24", 24), ("PCM_32", 32), ("FLOAT", "f32")]:
        p = out / "audios" / f"wav-{bits}-16k-mono-30s.wav"
        sf.write(p, signal(30, 16000, 1), 16000, subtype=subtype)
        c.add(
            p.stem,
            p,
            "audio",
            {"class": "wav", "bits": bits, "rate": 16000, "channels": 1, "seconds": 30},
            f"soundfile.write(subtype={subtype})",
        )
    for rate in [8000, 22050, 44100, 48000, 96000]:
        p = out / "audios" / f"wav-16-{rate}-mono-30s.wav"
        sf.write(p, signal(30, rate, 1), rate, subtype="PCM_16")
        c.add(
            p.stem,
            p,
            "audio",
            {"class": "wav", "bits": 16, "rate": rate, "channels": 1, "seconds": 30},
            "soundfile.write(PCM_16)",
        )
    for ch in [2, 6]:
        p = out / "audios" / f"wav-16-48k-{ch}ch-30s.wav"
        sf.write(p, signal(30, 48000, ch), 48000, subtype="PCM_16")
        c.add(
            p.stem,
            p,
            "audio",
            {"class": "wav", "bits": 16, "rate": 48000, "channels": ch, "seconds": 30},
            "soundfile.write(PCM_16)",
        )
    # durations
    for secs, label in [(0.1, "0.1s"), (1, "1s"), (30, "30s")] + ([] if args.skip_long else [(600, "600s")]):
        p = out / "audios" / f"wav-16-16k-mono-{label}.wav"
        sf.write(p, signal(secs, 16000, 1), 16000, subtype="PCM_16")
        c.add(
            p.stem,
            p,
            "audio",
            {"class": "duration", "rate": 16000, "channels": 1, "seconds": secs},
            "soundfile.write(PCM_16)",
        )
    # content classes
    for kind in ["silence", "clipping", "dc"]:
        p = out / "audios" / f"wav-16-16k-mono-{kind}-30s.wav"
        sf.write(p, signal(30, 16000, 1, kind), 16000, subtype="PCM_16")
        c.add(
            p.stem,
            p,
            "audio",
            {"class": "content", "content": kind, "rate": 16000, "channels": 1, "seconds": 30},
            "soundfile.write(PCM_16)",
        )
    # compressed formats from the 30 s 16-bit 16 kHz mono master and a 44.1 kHz stereo master
    master = out / "audios" / "wav-16-16k-mono-30s.wav"
    stereo = out / "audios" / "wav-16-44100-stereo-30s.wav"
    sf.write(stereo, signal(30, 44100, 2), 44100, subtype="PCM_16")
    c.add(
        stereo.stem,
        stereo,
        "audio",
        {"class": "wav", "bits": 16, "rate": 44100, "channels": 2, "seconds": 30},
        "soundfile.write(PCM_16)",
    )
    c.ffmpeg_run(
        "mp3-cbr-128k-16k-mono-30s",
        "audios/mp3-cbr-128k-16k-mono-30s.mp3",
        ["-i", str(master), "-c:a", "libmp3lame", "-b:a", "128k"],
        "audio",
        {"class": "mp3", "mode": "cbr", "rate": 16000, "channels": 1, "seconds": 30},
    )
    c.ffmpeg_run(
        "mp3-vbr-q4-44k-stereo-30s",
        "audios/mp3-vbr-q4-44k-stereo-30s.mp3",
        ["-i", str(stereo), "-c:a", "libmp3lame", "-q:a", "4"],
        "audio",
        {"class": "mp3", "mode": "vbr", "rate": 44100, "channels": 2, "seconds": 30},
    )
    c.ffmpeg_run(
        "flac-16k-mono-30s",
        "audios/flac-16k-mono-30s.flac",
        ["-i", str(master), "-c:a", "flac"],
        "audio",
        {"class": "flac", "rate": 16000, "channels": 1, "seconds": 30},
    )
    c.ffmpeg_run(
        "ogg-vorbis-44k-stereo-30s",
        "audios/ogg-vorbis-44k-stereo-30s.ogg",
        ["-i", str(stereo), "-c:a", "libvorbis", "-q:a", "5"],
        "audio",
        {"class": "ogg-vorbis", "rate": 44100, "channels": 2, "seconds": 30},
    )
    c.ffmpeg_run(
        "ogg-opus-48k-mono-30s",
        "audios/ogg-opus-48k-mono-30s.opus",
        ["-i", str(master), "-c:a", "libopus", "-b:a", "48k"],
        "audio",
        {"class": "opus", "rate": 48000, "channels": 1, "seconds": 30},
    )
    c.ffmpeg_run(
        "m4a-aac-44k-stereo-30s",
        "audios/m4a-aac-44k-stereo-30s.m4a",
        ["-i", str(stereo), "-c:a", "aac", "-b:a", "128k"],
        "audio",
        {"class": "aac", "rate": 44100, "channels": 2, "seconds": 30},
    )
    c.ffmpeg_run(
        "wav-alaw-8k-mono-30s",
        "audios/wav-alaw-8k-mono-30s.wav",
        ["-i", str(master), "-ar", "8000", "-c:a", "pcm_alaw"],
        "audio",
        {"class": "wav", "bits": "alaw", "rate": 8000, "channels": 1, "seconds": 30},
    )
    if not args.skip_long:
        long = out / "audios" / "wav-16-16k-mono-600s.wav"
        c.ffmpeg_run(
            "mp3-cbr-64k-16k-mono-600s",
            "audios/mp3-cbr-64k-16k-mono-600s.mp3",
            ["-i", str(long), "-c:a", "libmp3lame", "-b:a", "64k"],
            "audio",
            {"class": "mp3", "mode": "cbr", "rate": 16000, "channels": 1, "seconds": 600},
        )

    # ---------------- video ----------------
    def src(seconds: float, fps: float, size: str = "320x240") -> list[str]:
        return ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={seconds}"]

    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    # codecs / containers (5 s, 30 fps, 320x240)
    c.ffmpeg_run(
        "mp4-h264-30fps-5s",
        "videos/mp4-h264-30fps-5s.mp4",
        [*src(5, 30), *x264],
        "video",
        {"class": "codec", "codec": "h264", "container": "mp4", "fps": 30, "seconds": 5},
    )
    c.ffmpeg_run(
        "mp4-h265-30fps-5s",
        "videos/mp4-h265-30fps-5s.mp4",
        [
            *src(5, 30),
            "-c:v",
            "libx265",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-x265-params",
            "log-level=none",
            "-tag:v",
            "hvc1",
        ],
        "video",
        {"class": "codec", "codec": "h265", "container": "mp4", "fps": 30, "seconds": 5},
    )
    c.ffmpeg_run(
        "webm-vp9-30fps-5s",
        "videos/webm-vp9-30fps-5s.webm",
        [
            *src(5, 30),
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "500k",
            "-cpu-used",
            "8",
            "-deadline",
            "realtime",
            "-pix_fmt",
            "yuv420p",
        ],
        "video",
        {"class": "codec", "codec": "vp9", "container": "webm", "fps": 30, "seconds": 5},
    )
    c.ffmpeg_run(
        "mp4-av1-30fps-5s",
        "videos/mp4-av1-30fps-5s.mp4",
        [*src(5, 30), "-c:v", "libsvtav1", "-preset", "12", "-crf", "35", "-pix_fmt", "yuv420p"],
        "video",
        {"class": "codec", "codec": "av1", "container": "mp4", "fps": 30, "seconds": 5},
    )
    c.ffmpeg_run(
        "mkv-h264-30fps-5s",
        "videos/mkv-h264-30fps-5s.mkv",
        [*src(5, 30), *x264[:-2]],
        "video",
        {"class": "codec", "codec": "h264", "container": "mkv", "fps": 30, "seconds": 5},
    )
    c.ffmpeg_run(
        "mov-h264-30fps-5s",
        "videos/mov-h264-30fps-5s.mov",
        [*src(5, 30), *x264[:-2]],
        "video",
        {"class": "codec", "codec": "h264", "container": "mov", "fps": 30, "seconds": 5},
    )
    # frame rates
    for fps in [1, 10, 24, 60]:
        c.ffmpeg_run(
            f"mp4-h264-{fps}fps-5s",
            f"videos/mp4-h264-{fps}fps-5s.mp4",
            [*src(5, fps), *x264],
            "video",
            {"class": "fps", "codec": "h264", "fps": fps, "seconds": 5},
        )
    c.ffmpeg_run(
        "mp4-h264-29.97fps-5s",
        "videos/mp4-h264-29.97fps-5s.mp4",
        [*src(5, "30000/1001"), *x264],
        "video",
        {"class": "fps", "codec": "h264", "fps": 29.97, "seconds": 5},
    )
    # durations
    for secs in [0.5, 1, 10, 60] + ([] if args.skip_long else [300]):
        label = str(secs).replace(".", "p")
        c.ffmpeg_run(
            f"mp4-h264-30fps-{label}s",
            f"videos/mp4-h264-30fps-{label}s.mp4",
            [*src(secs, 30), *x264],
            "video",
            {"class": "duration", "codec": "h264", "fps": 30, "seconds": secs},
        )
    # resolutions (10 s, 30 fps)
    for size in ["640x360", "1280x720", "1920x1080"]:
        c.ffmpeg_run(
            f"mp4-h264-30fps-10s-{size}",
            f"videos/mp4-h264-30fps-10s-{size}.mp4",
            [*src(10, 30, size), *x264],
            "video",
            {"class": "resolution", "codec": "h264", "fps": 30, "seconds": 10, "size": size},
        )
    # odd dimensions (yuv444p so x264 accepts odd sizes)
    c.ffmpeg_run(
        "mp4-h264-odd-321x241",
        "videos/mp4-h264-odd-321x241.mp4",
        [*src(5, 30, "321x241"), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv444p"],
        "video",
        {"class": "odd", "codec": "h264", "fps": 30, "seconds": 5, "size": "321x241"},
    )
    # rotation metadata (display matrix)
    c.ffmpeg_run(
        "mp4-h264-rotate90",
        "videos/mp4-h264-rotate90.mp4",
        [*src(5, 30), *x264, "-display_rotation", "90"],
        "video",
        {"class": "rotation", "codec": "h264", "rotation": 90, "seconds": 5},
    )
    c.ffmpeg_run(
        "mp4-h264-rotate180",
        "videos/mp4-h264-rotate180.mp4",
        [*src(5, 30), *x264, "-display_rotation", "180"],
        "video",
        {"class": "rotation", "codec": "h264", "rotation": 180, "seconds": 5},
    )
    # with and without an audio track; audio-only
    c.ffmpeg_run(
        "mp4-h264-aac-5s",
        "videos/mp4-h264-aac-5s.mp4",
        [*src(5, 30), "-f", "lavfi", "-i", "sine=frequency=440:duration=5", *x264, "-c:a", "aac", "-shortest"],
        "video",
        {"class": "audio-track", "codec": "h264", "audio": "aac", "seconds": 5},
    )
    c.ffmpeg_run(
        "mp4-audio-only-aac-5s",
        "videos/mp4-audio-only-aac-5s.mp4",
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-c:a", "aac"],
        "video",
        {"class": "audio-only", "audio": "aac", "seconds": 5},
    )
    # variable frame rate: two segments (30 fps then 10 fps) concatenated
    a = c.ffmpeg_run("vfr-seg-a", "videos/_vfr-seg-a.mp4", [*src(3, 30), *x264], "video", {"class": "helper"})
    b = c.ffmpeg_run("vfr-seg-b", "videos/_vfr-seg-b.mp4", [*src(3, 10), *x264], "video", {"class": "helper"})
    if a and b:
        listing = out / "videos" / "_vfr.txt"
        listing.write_text(f"file '{a}'\nfile '{b}'\n")
        c.ffmpeg_run(
            "mp4-h264-vfr-30-10",
            "videos/mp4-h264-vfr-30-10.mp4",
            ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-fps_mode", "passthrough"],
            "video",
            {"class": "vfr", "codec": "h264", "seconds": 6, "rates": [30, 10]},
        )
    # gif as video
    c.ffmpeg_run(
        "gif-anim-10fps-3s",
        "videos/gif-anim-10fps-3s.gif",
        [*src(3, 10, "160x120")],
        "video",
        {"class": "gif", "fps": 10, "seconds": 3},
    )

    manifest = out / "manifest-av.json"
    manifest.write_text(json.dumps({"seed": SEED, "generator": "gen_av.py", "rows": c.rows}, indent=1))
    ok = sum(1 for r in c.rows if r.get("path"))
    print(f"AV_DONE ok={ok} err={len(c.rows) - ok} manifest={manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
