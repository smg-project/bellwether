# DeepSeek V4.1 Flash: the image processor's outputs

What the checkpoint's own image processor computes for 12 images: the two grids, the token count, the output
shape and FNV-1a digests of the output as float32 and bfloat16 and of the token types.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `deepseek_v41_preprocess_fingerprints.json` | 12 cases under the vision configuration the file carries (`vision_config`) | `generate_deepseek_v41_preprocess_fingerprints.py`; reference `inference/image_processor.py` of `deepseek-ai/DeepSeek-V4.1-Flash`, sha256 482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272; numpy 2.5.3, pillow 12.3.0, torch 2.14.0 |
