# Qwen2-VL and Qwen3-VL: the image and video processors' outputs

What the Hugging Face processors of the two families compute for fixed inputs: the resized size, the patches of
one image, the sampled frames and the grid of a clip, and the video prompt.

## Files

| file | content | source, as the file states it |
|---|---|---|
| `qwen_smart_resize.json` | the resized size of 26 input sizes under each of three sets of limits (`qwen2_vl`, `qwen3_vl`, `minimax_m3`) | `crates/multimodal/scripts/generate_qwen_smart_resize_golden.py` (the generator's path in smg); transformers 5.17.0; `minimax_m3_image_processor_sha256`, the sha256 of the MiniMax M3 image processor file |
| `qwen_pixel_norm.json` | one 448x448 synthetic image per family: the output shape, the grid and FNV-1a digests of the patches as bytes, float32 and bfloat16 | `generate_qwen_pixel_norm_golden.py`; `Qwen2VLImageProcessor`, transformers 5.17.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, numpy 2.2.6 |
| `qwen_preprocess_fingerprints.json` | 4 image cases and 1 video case: the grid, the shape and the FNV-1a digest of the patch bytes | `generate_qwen_preprocess_fingerprints.py`; pillow 12.3.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, transformers 5.17.0 |
| `qwen3_vl_video.json` | one 8-frame clip through `Qwen3VLVideoProcessor`: the sampled frames, the grid, the output shape and the token count | `generate_qwen3_vl_video_golden.py`; transformers 5.17.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130 |
| `qwen3_vl_video_prompt.json` | 12 cases of the video prompt: the frames picked at 2 fps between 4 and 768 frames, the grid, the timestamps and the placeholder text | transformers 5.17.0; the file names no generator, `generate_qwen3_vl_video_prompt_golden.py` is the one that writes it |

The clip `qwen3_vl_video.json` names (`tests/fixtures/videos/testsrc_128x96_8fps_1s.mp4`) is an input of smg's tests
and stays there.
