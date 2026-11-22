# step3: the image processor's outputs

What the serving engine's processor for the family computes for 14 images: the patch and token counts, the
newline mask, the shapes, and FNV-1a digests of the image and of its patches as float32 and bfloat16.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `step3_vl_preprocess_fingerprints.json` | 14 cases | reference `vllm/transformers_utils/processors/step3_vl.py`, sha256 ce0265261a6bffaf40b98911ef5b9babe30d20443af83a4c8ecf251ce6a861fa; torch 2.13.0+cpu; the file names no generator, `generate_step3_vl_preprocess_fingerprints.py` is the one that writes it |
