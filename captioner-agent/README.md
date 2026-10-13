# captioner-agent

A local web chat app. You attach images, Claude works out a caption for each
one and saves it as a text file with the same name. The image work is done by
the [image-captioner-mcp](../image-captioner-mcp) server, which this app starts
and talks to over MCP.

## Run

From the repository root, after the setup in the [main README](../README.md):

```
.\run.ps1
```

Then open <http://127.0.0.1:8000>.

## What happens to an image

The order is fixed in code ([pipeline.py](src/captioner_agent/pipeline.py)) and
runs before Claude is asked anything:

1. **PNG info.** The server's `caption_from_png_info` script takes the embedded
   prompt up to `Negative prompt:`, strips prompt syntax, and saves
   `Image1.txt` in the chat's output folder. No model is used.
2. **You choose, only for images with no PNG info.** After step 1 the chat
   stops and asks, with two buttons:
   - **BLIP (local, images stay on this machine):** the `BLIP_CAPTIONER` tool
     runs BLIP on your GPU.
   - **Anthropic (Claude, images are sent to Anthropic):** the image is
     downscaled and sent to Claude, which writes the caption following
     `config/caption_criteria.md`.

   Images that had PNG info are already captioned by then, and nothing is sent
   anywhere until you answer. Your answer applies to every image waiting in
   that batch. You can answer by typing "blip" or "anthropic" too, and you can
   attach more images with your answer. To skip the question, set
   `vision_backend` in `config/settings.yaml` to `blip`, `claude` or `local`.
3. **Optional enhancement.** Claude reports what happened and asks which of
   four things to do: keep the captions, **improve with Anthropic**, **make
   creative with Anthropic**, or **re-caption with BLIP**.
   - The two Anthropic choices call `ENHANCE_CAPTION`, which sends each
     caption's text (not the image) to Claude and saves the rewrite. It builds
     on the existing caption.
   - Re-captioning with BLIP calls `BLIP_CAPTIONER` on each image and saves the
     result over the old caption. BLIP looks at the picture and ignores the old
     caption, so it replaces rather than refines.
4. Claude then asks once whether you also want horizontally flipped copies. If you say yes, each copy is saved as
   `Image1_flipped.png` with a caption in which left and right are swapped.

## Configuration

| File | Purpose |
|---|---|
| `config/caption_criteria.md` | The caption rules. Re-read on every message, so edits apply immediately. |
| `config/settings.yaml` | Model, effort, port, upload limit, data folder and the MCP server connection. |
| `../.env` | `ANTHROPIC_API_KEY`. |

Uploads, captions and chat history are stored under `data/`:

```
data/
├─ agent.db                     chat history
└─ sessions/<chat id>/
   ├─ uploads/                  your images, untouched
   └─ outputs/                  Image1.txt, Image1_flipped.png, ...
```

**Download ZIP** in the captions panel bundles the outputs with the originals,
so every image sits next to its caption. Add `?originals=false` to the download
URL for the outputs alone.

## Layout

| Module | Role |
|---|---|
| `app.py` | FastAPI routes: chats, uploads, streaming messages, captions, ZIP. |
| `pipeline.py` | The fixed captioning order: PNG info, then vision. |
| `agent.py` | The agent loop: model call, tool calls, questions to the user, persistence. |
| `llm.py` | `LLMProvider` interface and the Claude implementation. |
| `toolbox.py` | MCP client that exposes the server's tools to Claude. |
| `prompts.py` | System prompt and the per-message file listing. |
| `files.py` | Upload naming, caption pairing, thumbnails, ZIP. |
| `store.py` | SQLite chat storage. |
| `web/` | The single-page UI (no build step). |

## Tests

```
pytest
```

The tests replace Claude with scripted responses, so they need no API key. One
of them drives the real MCP server in-process with a stub captioning model.
