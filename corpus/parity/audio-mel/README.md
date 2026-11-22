# The log-mel audio front end

The log-mel spectrogram of the audio front end of the Qwen3 audio models (16 kHz, 128 mel bins, a 400-point FFT, a
hop of 160) for six clips, and a bit-level fingerprint of the same front end for fourteen.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `audio_mel_reference.json` | 6 cases: the PCM samples, the shape of the spectrogram, its values and their sum | `generate_audio_mel_golden.py`; reference `transformers.WhisperFeatureExtractor`, resampler `torchaudio.functional.resample`, transformers 4.57.1, torch 2.13.0 |
| `qwen3_audio_mel_bits.json` | 14 cases: the shape, the token counts and the FNV-1a digest over the little-endian float32 bits of the encoder input | none: the file states its digest and its purpose ("bit-identity gate"), no generator and no library version |

`qwen3_audio_mel_bits.json` has no generator in smg either. It is not a reference's output: it holds what smg's own
front end gave when the file was written, so that a change meant to keep every bit is shown to keep it, where the
first file is compared within a tolerance.
