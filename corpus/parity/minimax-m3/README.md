# MiniMax M3 size rule: the engine processors' outputs over a synthetic corpus

What the serving engine's own MiniMax M3 processors compute for 63 synthetic images and 24 synthetic clips, recorded
from two vLLM trees so that a consumer can check its image and video preprocessing against the engine it runs:

* `vllm-v0.31.0-*.jsonl`: vLLM v0.31.0 (the release tag), whose `minimax_m3.py` processor is the checkpoint's own
  rule (`smart_resize` with the 672x672 pixel area cap for images, 768 x 28 x 28 per video frame) and whose video
  loader for the family walks the clip at 1 fps from the first frame without keeping the last one. The same file is
  byte-identical at the tags v0.32.0 and v0.33.0rc3.
* `vllm-main-46f36fa-*.jsonl`: vLLM main at 46f36fa294559d080edc61a09e5b57826218a2bf (2026-11-04), after commit
  dbd49af22 ("[New Model] Add MiniMax-M3.1-Flash Model and Kernels"), whose processor sizes by the M3.1 aligned-size
  rule (`compute_aligned_size`: a 168x168 area floor and a 168 px short-side floor, both sides ceil-aligned to 28,
  a long-side cap of 2016 px for images and 672 px for video frames at `detail=auto`, 672 / 3584 and 504 / 2188 at
  `low` / `high`) and whose video loader keeps the clip's last frame as well.

Both are the engine's processors built the way the engine builds them (`MiniMaxVLProcessor.from_pretrained` on the
`MiniMaxAI/MiniMax-M3` checkpoint's configuration files, byte-identical to `MiniMaxAI/MiniMax-M3.1-Flash`'s), fed the
engine's own loader output (images: Pillow, EXIF transpose, transparency composited over white, RGB; videos: OpenCV's
sequential read at the loader's picks, BGR to RGB), under the engine image's library pins (the first line of every
file names them with the processor file's sha256). The checkpoint's own remote-code processor gives the v0.31.0 numbers
for images (the vendored file is its copy).

## Files

| file | content |
|---|---|
| `images.jsonl`, `videos.jsonl` | the classes: one line per input with its file name under the generator's output directory, the generator's recipe, and the file's byte count and sha256 |
| `gen_images.py`, `gen_av.py` | the seeded generators of the inputs (numpy, Pillow and an ffmpeg binary; `python -I gen_images.py --out <dir> --ffmpeg <ffmpeg>`, `python -I gen_av.py --out <dir> --ffmpeg <ffmpeg>`); they write more files than the classes use, the class lists name the ones that count |
| `vllm-<tree>-images.jsonl` | per image: the decoded size, the grid (`t, h, w`), the output shape, the sha256 of the decoded RGB bytes, of the float32 output and of its bfloat16 rounding; `refused` with the processor's own message where it refuses the input (aspect ratios past 200:1, JXL) |
| `vllm-<tree>-videos.jsonl` | per clip: the source rate and frame count, the frame size, the loader's frame picks, the grid, the shape and the two digests; `refused` where neither the loader nor the processor takes the clip (AV1 without a decoder, an audio-only container) |

The inputs are not stored here: the generators are deterministic for a given Pillow, numpy and ffmpeg build, and the
class lists carry every file's sha256, so a regenerated corpus is checked file by file before the numbers are used.

## Reading the numbers

A grid of `[t, h, w]` means `t x h x w` patches of 14 px and `t x h x w / 4` tokens after the 2x2 merge. The two
trees disagree on every input: at the pin a 1024x768 image is a 40x54 grid (540 tokens), at main 56x74 (1036 tokens);
a 320x240 frame is 18x22 at the pin and 18x24 at main; a 5 s clip at 30 fps is five picks at the pin and six at main
(the last frame kept), a 10 s clip ten and eleven. The pin's numbers are what a deployment on vLLM v0.31.0 computes;
main's are what the next engine release carrying dbd49af22 will compute.
