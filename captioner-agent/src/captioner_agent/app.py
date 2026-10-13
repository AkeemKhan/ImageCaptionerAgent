"""FastAPI application: chat API with streaming, uploads, captions and the static UI."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import CaptionAgent
from .files import InvalidFileError, SessionFiles, make_thumbnail, verify_image
from .llm import AnthropicProvider, LLMProvider
from .settings import Settings
from .store import Session, SessionStore
from .toolbox import McpToolbox, Toolbox

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_FILES_PER_UPLOAD = 500


class MessageRequest(BaseModel):
    text: str = Field(default="", max_length=20_000)
    files: list[str] = Field(default_factory=list, max_length=MAX_FILES_PER_UPLOAD)


class CaptionUpdate(BaseModel):
    text: str = Field(max_length=10_000)


def create_app(
    settings: Settings,
    provider: LLMProvider | None = None,
    toolbox: Toolbox | None = None,
) -> FastAPI:
    """Build the app. ``provider`` and ``toolbox`` can be injected for tests;
    by default the app talks to Claude and starts or connects to the MCP server."""
    store = SessionStore(settings.db_path)
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    # One turn at a time per session, so two requests cannot interleave history.
    turn_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned: McpToolbox | None = None
        active = toolbox
        if active is None:
            owned = McpToolbox(settings)
            await owned.connect()
            active = owned
        app.state.agent = CaptionAgent(provider or AnthropicProvider(settings), active, store, settings)
        try:
            yield
        finally:
            if owned is not None:
                await owned.close()

    app = FastAPI(title="Image Captioner Agent", lifespan=lifespan)

    def get_session(session_id: str) -> Session:
        session = store.get(session_id)
        if session is None:
            raise HTTPException(404, "Chat not found")
        return session

    def session_files(session_id: str) -> SessionFiles:
        get_session(session_id)
        return SessionFiles(settings.sessions_dir, session_id)

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {"model": settings.model, "max_upload_mb": settings.max_upload_mb}

    @app.get("/api/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return store.list()

    @app.post("/api/sessions", status_code=201)
    def create_session() -> dict[str, Any]:
        session = store.create()
        SessionFiles(settings.sessions_dir, session.id).ensure()
        return {"id": session.id, "title": session.title}

    @app.get("/api/sessions/{session_id}")
    def read_session(session_id: str) -> dict[str, Any]:
        session = get_session(session_id)
        return {
            "id": session.id,
            "title": session.title,
            "transcript": session.transcript,
            "pending_choice": session.pending["choice"] if session.pending else None,
        }

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> Response:
        files = session_files(session_id)
        if turn_locks[session_id].locked():
            raise HTTPException(409, "This chat is busy")
        store.delete(session_id)
        files.remove()
        turn_locks.pop(session_id, None)
        return Response(status_code=204)

    @app.post("/api/sessions/{session_id}/files")
    async def upload_files(session_id: str, files: list[UploadFile]) -> dict[str, Any]:
        """Store uploaded images; returns the saved names and any rejected files."""
        target = session_files(session_id)
        target.ensure()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        saved: list[str] = []
        rejected: list[dict[str, str]] = []
        for upload in files[:MAX_FILES_PER_UPLOAD]:
            original = upload.filename or "image"
            path: Path | None = None
            try:
                path = target.new_upload_path(original)
                await _save_upload(upload, path, max_bytes)
                await asyncio.to_thread(verify_image, path)
            except InvalidFileError as exc:
                if path is not None:
                    path.unlink(missing_ok=True)
                rejected.append({"name": original, "reason": str(exc)})
                continue
            saved.append(path.name)
        return {"saved": saved, "rejected": rejected}

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, request: MessageRequest) -> StreamingResponse:
        """Run one agent turn and stream its events as server-sent events."""
        session = get_session(session_id)
        files = SessionFiles(settings.sessions_dir, session_id)
        try:
            uploaded = [files.resolve("uploads", name).name for name in request.files]
        except (InvalidFileError, FileNotFoundError) as exc:
            raise HTTPException(400, "Unknown uploaded file") from exc
        if not request.text.strip() and not uploaded:
            raise HTTPException(400, "Message is empty")
        lock = turn_locks[session_id]
        if lock.locked():
            raise HTTPException(409, "This chat is still working on the previous message")
        agent: CaptionAgent = app.state.agent

        async def events() -> AsyncIterator[str]:
            async with lock:
                try:
                    async for event in agent.run(session, request.text, uploaded):
                        yield _sse(event)
                except Exception:
                    logger.exception("Agent turn failed")
                    yield _sse({"type": "error", "message": "Something went wrong. See the server log."})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}/outputs")
    def list_outputs(session_id: str) -> list[dict[str, Any]]:
        return session_files(session_id).list_items()

    @app.put("/api/sessions/{session_id}/captions/{image_name}", status_code=204)
    def update_caption(session_id: str, image_name: str, update: CaptionUpdate) -> Response:
        if not update.text.strip():
            raise HTTPException(400, "Caption is empty")
        try:
            session_files(session_id).write_caption(image_name, update.text)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Image not found") from exc
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/files/{kind}/{name}")
    async def read_file(
        session_id: str, kind: Literal["uploads", "outputs"], name: str, thumb: bool = False
    ) -> Response:
        try:
            path = session_files(session_id).resolve(kind, name)
        except (InvalidFileError, FileNotFoundError) as exc:
            raise HTTPException(404, "File not found") from exc
        if not thumb:
            return FileResponse(path)
        try:
            data = await asyncio.to_thread(make_thumbnail, path)
        except Exception as exc:
            raise HTTPException(415, "Not an image") from exc
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/sessions/{session_id}/download")
    async def download(session_id: str, originals: bool = True) -> Response:
        data = await asyncio.to_thread(session_files(session_id).build_zip, originals)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="captions-{session_id[:8]}.zip"'},
        )

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


async def _save_upload(upload: UploadFile, path: Path, max_bytes: int) -> None:
    """Stream an upload to disk, raising if it exceeds ``max_bytes`` (the caller cleans up)."""
    written = 0
    with path.open("wb") as out:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise InvalidFileError(f"file is larger than {max_bytes // (1024 * 1024)} MB")
            out.write(chunk)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"
