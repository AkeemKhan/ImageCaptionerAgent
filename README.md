# Image Captioner Agent

A chat app that writes a caption file for every image you give it. Drop in
`Image1.png` and you get `Image1.txt`.

Images are captioned in a fixed order, enforced in code:

1. **PNG info first.** A Python script reads the generation prompt embedded in
   the image, keeps everything before `Negative prompt:`, and strips prompt
   syntax such as `( ) : < >` and LoRA tags. No model is involved.
2. **Claude only when there is no PNG info.** Claude looks at the image and
   writes the caption, following your own rules.

On request, it also makes mirrored copies with matching captions.

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

**Privacy:** an image with PNG info never leaves your machine. An image
without it is downscaled and sent to the Anthropic API so Claude can see it.
To keep every image local, set `vision_backend: local` in
[settings.yaml](captioner-agent/config/settings.yaml); the Florence-2 model on
your GPU then does step 2.

The PNG info script also runs on its own, with no API key:

```powershell
.\.venv\Scripts\png-info-caption.exe D:\dataset --output-dir D:\dataset\captions
```

## Requirements

- Python 3.10 or newer.
- An Anthropic API key.
- Only for `vision_backend: local`: an NVIDIA GPU with about 3 GB of free VRAM
  for Florence-2 large. Without one the model runs on the CPU, which is slower.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -e ".\image-captioner-mcp[florence,dev]" -e ".\captioner-agent[dev]"
Copy-Item .env.example .env      # then put your API key in .env
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
