# Inkling: audio and vision preprocessing fingerprints

Fingerprints of the audio and vision preprocessing for the family: 4 audio cases (the shape, the token counts and an
FNV-1a digest of the int32 output) and 2 vision cases (the shape, the token counts and an FNV-1a digest of the float32
output), each from a seeded synthetic input.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `inkling_preprocess_fingerprints.json` | 4 audio cases and 2 vision cases | none: the file states no reference, no generator and no library version |

No generator for it stands in smg, and the test that reads it there calls itself a regression check. Until the file
is recorded again from the family's reference with its source stated, it shows that an output did not change, not
that it agrees with the reference.

## Wanted

One vision case whose configuration sets `rescale_image_frac`, recorded from the reference by a generator, not written
by hand. The two vision cases here do not rescale, so no case binds the Lanczos filter of that step.
