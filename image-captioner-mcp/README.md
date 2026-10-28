# image-captioner-mcp

An MCP server that captions images with a locally hosted model and exposes the
file work around it: reading generator metadata, writing caption files and
making flipped copies, and cleans up and upscales images. It knows nothing about any chat UI, so it works from any
MCP client, including Claude Desktop and Claude Code.

## Tools

| Tool | What it does |
|---|---|
| `list_images(directory)` | Lists png, jpg, jpeg and webp files in a directory. |
| `read_png_metadata(path)` | Returns the prompt an image generator embedded (AUTOMATIC1111/Forge, ComfyUI, NovelAI, InvokeAI, Fooocus), or any other embedded text. |
| `caption_from_png_info(path, output_dir, overwrite)` | Writes the caption from the embedded prompt: everything before `Negative prompt:`, with `( ) [ ] { } < > :`, attention weights and `<lora:...>` tags removed. No model. Reports `found: false` when there is no PNG info. |
| `caption_image(path, style)` | Describes the image with the local Florence-2 model. `style` is `short`, `detailed` or `more_detailed`. |
| `MANUAL_CAPTION(captions, output_dir, overwrite)` | Saves captions the user wrote, `[{path, caption}, ...]`, as `<image stem>.txt`. The text is kept as written, including `(word:1.2)`; only line breaks become spaces. One bad entry is reported and the rest are saved. |
| `BLIP_CAPTIONER(path, style)` | Describes the image with the local BLIP model (default style `short`). Returns the caption without saving it. |
| `write_caption(image_path, text, output_dir, overwrite)` | Writes `Image1.txt` for `Image1.png`. |
| `flip_image(path, output_dir, direction)` | Saves `Image1_flipped.png`. |
| `mirror_caption(text)` | Swaps "left" and "right" so a caption fits the flipped copy. |
| `CLEANUP_AND_ENHANCE(path, output_dir, remove_text, keep_text, regions, scale, overwrite, backend, instructions)` | Removes text laid over the image (watermarks, usernames, labels, Snapchat-style caption bars) and upscales it by `scale` (1, 2 or 4), saving `Image1_clean.png`. Lists every text it removed; `keep_text` spares text containing those words, and `regions` removes extra areas such as a logo, as fractions of the image. `backend` is `local` (default) or `gemini`, which sends the image to Google and also removes the bubbles and boxes around text, guided by a prompt and `instructions`. See [Cleanup and upscaling](#cleanup-and-upscaling). |
| `caption_folder(directory, output_dir, style, overwrite)` | Captions a whole directory, with progress notifications. |

It also exposes a resource, `captions://{folder}` (the caption files in a
folder as JSON), and a prompt, `simplify_caption`.

## Install

```
pip install -e ".[florence]"
```

Install `torch` for your CUDA version first if you want GPU inference; see
<https://pytorch.org/get-started/locally/>.

## Run

```
image-captioner-mcp                                  # stdio
image-captioner-mcp --transport streamable-http      # http://127.0.0.1:8765/mcp
```

## PNG info script

The PNG info captioner ([png_caption.py](src/image_captioner_mcp/png_caption.py))
also runs as a command, without the server or any model:

```
png-info-caption Image1.png
png-info-caption D:\dataset --output-dir D:\dataset\captions --no-overwrite
```

## Download the models

Each model downloads the first time it is used, which can take minutes and
looks like a hang. Download them ahead of time instead:

```
captioner-download-models              # BLIP only, about 1.9 GB
captioner-download-models --cleanup    # Florence-2, LaMa and Swin2SR, for CLEANUP_AND_ENHANCE, about 1.8 GB
captioner-download-models --all        # every model, about 3.7 GB
captioner-download-models --dry-run    # list the files without downloading
```

Only one copy of the weights is fetched (safetensors where available). The
large BLIP repository also holds PyTorch and TensorFlow copies of the same
weights, so a plain `git clone` or `snapshot_download` would fetch about 5.6 GB.

Models are cached in `%USERPROFILE%\.cache\huggingface\hub`. To put the cache
on another drive, set `HF_HOME` before running anything. Nothing needs a
Hugging Face account or token. BLIP is BSD-3-Clause licensed and not gated.

BLIP runs in about 1 GB of VRAM (Florence-2 large uses about 1.8 GB) and loads
in a few seconds once cached.

## Cleanup and upscaling

`CLEANUP_AND_ENHANCE` runs three local models, each loaded on first use:

1. **Find text.** Florence-2's OCR, the model behind `caption_image`, so it is
   not loaded twice. Large images are also read in tiles, so a small corner
   watermark is not lost.
2. **Remove it.** When the text sits on a semi-transparent dark bar running
   across the image, as Snapchat captions do, the bar is first undone: the
   darkening is measured at its edges and reversed, so the picture under it
   comes back instead of being repainted. Then LaMa (about 200 MB) paints out
   the letters, working on crops around them so a large photo does not need
   much VRAM.
3. **Upscale.** Swin2SR real-world x4 (about 50 MB) runs in tiles; the result
   is resized to the requested scale, and capped at `CAPTIONER_MAX_OUTPUT_SIDE`.

The OCR reads all text, including text that belongs in the picture, such as a
sign or a printed shirt. That is why the result lists what was removed: run it
again with `keep_text` to keep some. Text the OCR cannot read, such as a logo
drawn as a picture, can be removed with `regions`.

The original is never changed. PNG text chunks (the generation prompt) are
copied to the clean copy, since removing an overlay does not change what the
prompt describes. The copy has no caption file of its own; save one with
`write_caption`.

### Using Gemini instead

With `backend: "gemini"`, Google's Gemini does the whole job from one prompt: it
removes **every overlay** (text with the bubbles, caption bars, boxes and borders
around it, stickers, emoji, product cards, inset photos), rebuilds those areas,
**restores** the photo (film grain, noise, filters, compression artefacts, blur) and
**sharpens** it; the size asked for is 2K by default (`CAPTIONER_GEMINI_IMAGE_SIZE`; scale 1 keeps the
original size). The prompt deliberately never asks to keep the picture "exactly
the same": that makes Gemini timid and the result soft. No local model decides what goes; the prompt
does.

**This sends the image to Google** and is billed per image, so the server refuses
it unless the operator turned it on: set `CAPTIONER_GEMINI_ENABLED=true` and
`GEMINI_API_KEY`. Without both, the tool returns an error and nothing leaves the
machine. A client should ask the user before choosing it; the captioner-agent web
app does, and enforces it.

**The prompt.** The base prompt is the file named by `CAPTIONER_GEMINI_PROMPT_FILE`
(the web app ships `captioner-agent/config/gemini_cleanup_prompt.md`), or a
built-in copy of it. It is read on every call, so edits apply straight away. To it
the server adds this image's details: the output size, any `keep_text` and
`regions`, and the caller's `instructions` (for example "also remove the emoji
stickers"). Text from the caller is trimmed and stripped of quotes and line breaks
first.

**Shapes.** Gemini only returns certain aspect ratios (1:1, 3:4, 9:16 and so on). A
picture that is not close to one, such as a tall phone screenshot, is placed on a
black background of the nearest supported shape, and the margins are cropped off
Gemini's answer.

**Fallback.** If Gemini declines, errors, runs out of credits, or answers with the
wrong proportions, the local pipeline cleans the same image instead; Google is not
asked twice. The result then has `backend: "local"`, `requested_backend: "gemini"`
and a `fallback_reason`; `image_sent_to_google` says whether a request went out. If
Gemini's image is smaller than asked for, its own pixels are kept, since stretching
them only softens the result; it is only enlarged if it would be smaller than the
original. Set `CAPTIONER_GEMINI_UPSCALE_FALLBACK` to `local` (local upscaler) or
`resize` to reach the full size anyway. With Gemini,
`removed_text` is empty.

In testing, `gemini-2.5-flash-image` declined to remove a watermark and
`gemini-3-pro-image-preview` did the removal and a 2K upscale, so that is the
default. Image models change often: set `CAPTIONER_GEMINI_MODEL` to try another.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `CAPTIONER_ALLOWED_ROOT` | current directory | The only directory tree tools may read or write. |
| `CAPTIONER_BACKEND` | `florence2` | `florence2`, or `stub` (no model, for tests). |
| `CAPTIONER_MODEL_ID` | `microsoft/Florence-2-large` | Use `microsoft/Florence-2-base` for less VRAM. |
| `CAPTIONER_MODEL_REVISION` | unset | Pin the model repository to a commit. |
| `CAPTIONER_DEVICE` | `auto` | `auto`, `cuda` or `cpu`. |
| `CAPTIONER_MIN_FREE_VRAM_MB` | `3000` | Below this much free VRAM, `auto` uses the CPU. |
| `CAPTIONER_MAX_IMAGE_SIDE` | `1024` | Images are downscaled to this before inference. |
| `CAPTIONER_PRELOAD` | `true` | Load the model at startup instead of on first use. |
| `CAPTIONER_BLIP_MODEL_ID` | `Salesforce/blip-image-captioning-large` | Use `Salesforce/blip-image-captioning-base` (about 1 GB) to save disk and VRAM. |
| `CAPTIONER_BLIP_MODEL_REVISION` | unset | Pin the BLIP repository to a commit. |
| `CAPTIONER_PRELOAD_BLIP` | `false` | Load BLIP at startup instead of on first use. |
| `CAPTIONER_LAMA_MODEL_ID` | `fashn-ai/LaMa` | Repository holding the LaMa inpainting model as `big-lama.pt` (TorchScript). |
| `CAPTIONER_LAMA_MODEL_REVISION` | pinned commit | Set to another commit to move off the tested one. |
| `CAPTIONER_UPSCALE_MODEL_ID` | `caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` | Swin2SR upscaler, x4. |
| `CAPTIONER_UPSCALE_MODEL_REVISION` | unset | Pin the upscaler repository to a commit. |
| `CAPTIONER_MAX_OUTPUT_SIDE` | `4096` | Upscaling stops before the longest side passes this. |
| `CAPTIONER_GEMINI_ENABLED` | `false` | Allow `backend: "gemini"`. Also needs `GEMINI_API_KEY`. |
| `GEMINI_API_KEY` | unset | Google AI Studio key. Never logged or included in errors. |
| `CAPTIONER_GEMINI_MODEL` | `gemini-3-pro-image-preview` | The Gemini image model. |
| `CAPTIONER_GEMINI_TIMEOUT_SECONDS` | `600` | How long to wait for one image; 4K takes minutes. |
| `CAPTIONER_GEMINI_IMAGE_SIZE` | `2K` | `1K`, `2K` or `4K`: the size asked for. The prompt, not the size, drives quality. |
| `CAPTIONER_GEMINI_PROMPT_FILE` | unset | Base prompt for Gemini; the built-in one if unset or empty. |
| `CAPTIONER_GEMINI_UPSCALE_FALLBACK` | `none` | When Gemini's image is smaller than asked: `none` keeps it (sharpest), `local` upscales it locally, `resize` stretches it. |
| `CAPTIONER_CRITERIA_FILE` | unset | Markdown file used by the `simplify_caption` prompt. |

## Safety

- **Paths.** Every path is resolved (following `..` and symlinks) and must end
  up inside `CAPTIONER_ALLOWED_ROOT`.
- **Gemini.** Off unless `CAPTIONER_GEMINI_ENABLED` and a key are both set. It is the only
  code that contacts Google, and it does so only for a `backend: "gemini"` call.
- **GPU.** The model loads once, inference runs one image at a time, CUDA is
  used only if enough VRAM is free, and a CUDA out-of-memory error moves the
  model to the CPU instead of failing.
- **Model code.** Florence-2 loads with `trust_remote_code`, which runs Python
  from the model repository. Set `CAPTIONER_MODEL_REVISION` to a commit you
  have reviewed to keep that code fixed. BLIP does not need it: its code ships
  with `transformers` and its weights are plain safetensors.

## Claude Desktop

Add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "image-captioner": {
      "command": "D:\\Projects\\ImageCaptionerAgent\\.venv\\Scripts\\image-captioner-mcp.exe",
      "env": { "CAPTIONER_ALLOWED_ROOT": "D:\\Pictures\\dataset" }
    }
  }
}
```

## Tests

```
pytest
```

The tests use the `stub` backend, so they need neither a GPU nor a model download.
