# Image Captioner Agent

A chat app that writes a caption file for every image you give it. Drop in
`Image1.png` and you get `Image1.txt`.

If you would rather write the captions yourself, say so or type them with the
images ("image1 is a red car, image2 is a blue boat"). The automatic steps are
skipped and your words are saved exactly as written, by the `MANUAL_CAPTION`
tool. Otherwise images are captioned in a fixed order, enforced in code:

1. **PNG info first.** A Python script reads the generation prompt embedded in
   the image, keeps everything before `Negative prompt:`, and strips prompt
   syntax such as `( ) : < >` and LoRA tags. No model is involved.
2. **You choose, only when there is no PNG info.** The chat asks whether to
   caption those images with **BLIP** (the `BLIP_CAPTIONER` tool, on your GPU,
   images stay on your machine) or **Anthropic** (the image is sent to Claude).

Afterwards you can choose to enhance the captions with Anthropic (improved or
more creative, built on the existing caption) or to re-caption them with BLIP.
On request, it also makes mirrored copies with matching captions, and clean
copies with watermarks, labels and app captions (Snapchat text and its bar)
removed and the image upscaled, all on your GPU (the `CLEANUP_AND_ENHANCE` tool).

It is built as two applications that talk over the
[Model Context Protocol](https://modelcontextprotocol.io):

```
 Browser                captioner-agent                 image-captioner-mcp
┌────────────┐  HTTP   ┌──────────────────────┐  MCP   ┌──────────────────────────┐
│ Chat UI    │ ──────> │ FastAPI + agent loop │ ─────> │ Tools:                   │
│ uploads    │ <────── │ 1. PNG info script   │ <───── │  caption_from_png_info   │
│ captions   │  SSE    │ 2. Claude vision     │        │  caption_image (Florence)│
└────────────┘         │ then Claude chat     │        │  write_caption           │
                       └──────────────────────┘        │  flip_image, ...         │
                                  │                    └──────────────────────────┘
                                  └──── shared folder: data/sessions ────┘
```

| Folder | What it is |
|---|---|
| [captioner-agent](captioner-agent) | The web chat app and the Claude agent loop. |
| [image-captioner-mcp](image-captioner-mcp) | A standalone MCP server with the image tools and the local model. Also works from Claude Desktop. |

**Privacy:** an image is sent to Anthropic only if you pick Anthropic when the
chat asks how to caption images that have no PNG info (or if you pin
`vision_backend: claude` in [settings.yaml](captioner-agent/config/settings.yaml)).
Choose BLIP and it never leaves your machine. Enhancing with Anthropic sends
only the text of a caption, never the image.

The PNG info script also runs on its own, with no API key:

```powershell
.\.venv\Scripts\png-info-caption.exe D:\dataset --output-dir D:\dataset\captions
```

## Requirements

- Python 3.10 or newer.
- An Anthropic API key.
- About 2 GB of disk for the BLIP model, and an NVIDIA GPU with about 3 GB of
  free VRAM (BLIP itself uses about 1 GB). Without a GPU the model runs on the
  CPU, which is slower.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".\image-captioner-mcp[florence,dev]" -e ".\captioner-agent[dev]"
Copy-Item .env.example .env      # then put your API key in .env
```

Download the BLIP model once, so the first caption does not stall on a 1.9 GB
download:

```powershell
.\.venv\Scripts\captioner-download-models.exe
```

To clean up and upscale images, also fetch the cleanup models (Florence-2, LaMa
and Swin2SR, about 1.8 GB):

```powershell
.\.venv\Scripts\captioner-download-models.exe --cleanup
```

## Run

```powershell
.\run.ps1
```

Open <http://127.0.0.1:8000>.

## Use

1. Attach images with **Images** or **Folder**, drag them in, or paste from the clipboard.
2. Send. The captions appear in the panel on the right as they are written.
3. Answer the question about flipped copies.
4. Edit any caption in the panel, then **Download ZIP**.

To change how Claude writes captions for images without PNG info, edit
[captioner-agent/config/caption_criteria.md](captioner-agent/config/caption_criteria.md).
The next image uses the new rules. PNG info captions are not rewritten.

## Configuration

- [captioner-agent/config/settings.yaml](captioner-agent/config/settings.yaml):
  Claude model and effort, port, upload limit, and how to reach the MCP server.
- Captioning model, device and VRAM limits: the `mcp.env` block in that file.
  The variables are listed in [image-captioner-mcp/README.md](image-captioner-mcp/README.md).

## Tests and checks

```powershell
cd image-captioner-mcp; ..\.venv\Scripts\python.exe -m pytest; ..\.venv\Scripts\python.exe -m mypy; cd ..
cd captioner-agent;     ..\.venv\Scripts\python.exe -m pytest; ..\.venv\Scripts\python.exe -m mypy; cd ..
.\.venv\Scripts\ruff.exe check .
```

Neither test suite needs an API key, a GPU or a model download.

## Docker

`docker compose up --build` runs the two applications as separate containers
sharing one volume, with the MCP server on streamable HTTP. See
[docker-compose.yml](docker-compose.yml).

## Security notes

- The web app binds to `127.0.0.1` and has no login. Do not expose it to a network.
- The MCP server refuses any path outside its allowed root.
- The MCP server is started without your Anthropic credentials in its environment.
- Florence-2 loads with `trust_remote_code`. Pin `CAPTIONER_MODEL_REVISION` to
  fix the code that runs.
