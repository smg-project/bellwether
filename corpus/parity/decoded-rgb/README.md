# Decoded RGB: what Pillow decodes an encoded image to

The RGB bytes Pillow gives for 12 small encoded images of different modes, with and without transparency, so that a
consumer's image decode is checked byte for byte before any model's preprocessing runs.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `manifest.json` | 12 cases: the image's name, size, mode and whether it is transparent, a note, and the expected RGB bytes in base64 (and, for a transparent image, the bytes with the alpha channel dropped) | `crates/multimodal/scripts/generate_decoded_rgb_fixtures.py` (the generator's path in smg; it stands here as `generate_decoded_rgb_fixtures.py`); pillow 12.3.0 |

The encoded images the cases name are inputs of smg's tests and stay there.
