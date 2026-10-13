# image-captioner-mcp

An MCP server that captions images with a locally hosted model and exposes the
file work around it: reading generator metadata, writing caption files and
making flipped copies. It knows nothing about any chat UI, so it works from any
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
captioner-download-models --all        # BLIP and Florence-2, about 3.4 GB
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
| `CAPTIONER_CRITERIA_FILE` | unset | Markdown file used by the `simplify_caption` prompt. |

## Safety

- **Paths.** Every path is resolved (following `..` and symlinks) and must end
  up inside `CAPTIONER_ALLOWED_ROOT`.
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
