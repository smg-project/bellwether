# Gemma 4: the image processor's outputs

What the Hugging Face image processor of the family computes for 40 images and 2 videos: the
target size, the soft token count, the output and position shapes, and FNV-1a digests of the output as float32 and
bfloat16 and of the positions.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `gemma4_preprocess_fingerprints.json` | 40 image cases and 2 video cases under the processor settings the file carries (`processor`) | `generate_gemma4_preprocess_fingerprints.py`; reference `transformers/models/gemma4/image_processing_gemma4.py` (torchvision backend), sha256 5d280d5448b1c219183a27e95b6aa7178b350275772d07881c91c65f13fa815e; transformers 5.17.0, torch 2.13.0+cu130, numpy 2.2.6 |
