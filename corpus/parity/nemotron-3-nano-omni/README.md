# Nemotron 3 Nano Omni: image tiles, video sizing and audio features

What the serving engine's preprocessing for the family computes for 4 images, 1 clip and 5 audio clips: the tile rows
and columns or the grid, the token count, and sampled values of the output with its sum and sum of squares.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `nemotron_omni_preprocess.json` | 4 image cases, 1 video case and 5 audio cases | `crates/multimodal/scripts/generate_nemotron_omni_preprocess_golden.py` (the generator's path in smg; it stands here as `generate_nemotron_omni_preprocess_golden.py`); reference "vLLM nano_nemotron_vl dynamic tiler, video sizing and ParakeetExtractor, reproduced with torch"; torch 2.14.0+cu130 |

The file names no commit or tag of the engine.
