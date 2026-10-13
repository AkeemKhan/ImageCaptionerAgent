"""The agent loop: the pipeline captions uploads, then Claude reports and handles requests."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import jsonschema

from .files import FileKind, InvalidFileError, SessionFiles
from .llm import LLMError, LLMProvider, TurnResult
from .pipeline import CaptionOutcome, CaptionPipeline
from .prompts import (
    ASK_USER_TOOL,
    DESCRIBE_TOOL,
    ENHANCE_MODES,
    ENHANCE_TOOL,
    build_system_prompt,
    build_user_message,
    load_criteria,
)
from .settings import Settings
from .store import Session, SessionStore, make_title
from .toolbox import Toolbox, ToolResult

logger = logging.getLogger(__name__)

MAX_MODEL_CALLS_PER_TURN = 60
MAX_MALFORMED_RETRIES = 2
RESULT_PREVIEW_CHARS = 400
# Server tools that caption with the local model. Claude gets describe_image
# instead, which follows the configured vision backend.
HIDDEN_SERVER_TOOLS = frozenset({"caption_image", "caption_folder"})

DESCRIBE_DEFINITION: dict[str, Any] = {
    "name": DESCRIBE_TOOL,
    "description": (
        "Have the vision model look at one image from this chat and write its caption, "
        "following the caption criteria. Returns the caption text and does not save it. "
        "Use it only for an image that caption_from_png_info reported as found: false."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Image path inside upload_dir or output_dir, as listed by the app.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "eager_input_streaming": True,
}

ENHANCE_DEFINITION: dict[str, Any] = {
    "name": ENHANCE_TOOL,
    "description": (
        "Enhance the existing caption of one image with Claude and save the result over "
        "the old caption file. Use it only after the image has been captioned, and only "
        "when the user wants enhancement. Only the caption text is sent, not the image, "
        "so the rewrite keeps the facts of the original caption. Returns the original "
        "and the enhanced caption."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the captioned image inside upload_dir or output_dir.",
            },
            "mode": {
                "type": "string",
                "enum": list(ENHANCE_MODES),
                "description": (
                    "improved: clear, natural sentences with the same content. "
                    "creative: a more vivid and expressive description."
                ),
            },
            "guidance": {
                "type": "string",
                "description": "Optional extra wishes from the user, for example a tone or length.",
            },
        },
        "required": ["path", "mode"],
        "additionalProperties": False,
    },
    "eager_input_streaming": True,
}

# Tools the agent runs itself instead of sending to the MCP server.
LOCAL_TOOLS = frozenset({DESCRIBE_TOOL, ENHANCE_TOOL})

ASK_USER_DEFINITION: dict[str, Any] = {
    "name": ASK_USER_TOOL,
    "description": (
        "Ask the user a multiple-choice question and wait for their answer. The app "
        "shows the options as buttons; the user can also type a different reply. "
        "The result is the user's answer as text. Ask one question at a time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to show."},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
                "description": "Short button labels, one per choice.",
            },
        },
        "required": ["question", "options"],
        "additionalProperties": False,
    },
    "eager_input_streaming": True,
}

Event = dict[str, Any]


class CaptionAgent:
    """Runs one user turn at a time for a session and persists the outcome.

    A turn is committed to the store only when it completes or pauses on a
    question. If it fails part-way, the session keeps its previous state, so
    the conversation sent to Claude never ends on an unanswered tool call.
    """

    def __init__(
        self, provider: LLMProvider, toolbox: Toolbox, store: SessionStore, settings: Settings
    ) -> None:
        self._provider = provider
        self._toolbox = toolbox
        self._store = store
        self._settings = settings
        self._pipeline = CaptionPipeline(provider, toolbox, settings)

    async def run(self, session: Session, text: str, uploaded: list[str]) -> AsyncIterator[Event]:
        """Handle a user message, yielding UI events as the turn progresses.

        Uploaded images are captioned by the fixed pipeline (PNG info, then
        vision) before Claude is asked anything; Claude receives the results.
        """
        files = SessionFiles(self._settings.sessions_dir, session.id)
        messages = copy.deepcopy(session.messages)
        transcript: list[Event] = [{"type": "user", "text": text, "files": uploaded}]

        outcomes: list[CaptionOutcome] = []
        async for event in self._pipeline.run(files, uploaded, outcomes):
            if event["type"] == "tool_end":
                transcript.append({**event, "type": "tool"})
            yield event
        messages.append(self._user_turn(session, files, text, [o.to_dict() for o in outcomes]))
        pending: dict[str, Any] | None = None

        tools = [
            *(t for t in self._toolbox.definitions() if t["name"] not in HIDDEN_SERVER_TOOLS),
            DESCRIBE_DEFINITION,
            ENHANCE_DEFINITION,
            ASK_USER_DEFINITION,
        ]
        schemas = {tool["name"]: tool["input_schema"] for tool in tools}
        system = build_system_prompt(load_criteria(self._settings.criteria_file))

        try:
            for _ in range(MAX_MODEL_CALLS_PER_TURN):
                result: TurnResult | None = None
                reply = ""
                async for item in self._model_call(system, messages, tools):
                    if isinstance(item, str):
                        reply += item
                        yield {"type": "text", "text": item}
                    else:
                        result = item
                if result is None:
                    raise LLMError("The model returned no response.")
                messages.append({"role": "assistant", "content": result.content})
                if reply:
                    transcript.append({"type": "assistant", "text": reply})
                    yield {"type": "text_end"}

                if result.stop_reason == "refusal":
                    note = result.refusal_explanation or "Claude declined this request."
                    transcript.append({"type": "notice", "text": note})
                    yield {"type": "notice", "text": note}
                    break
                if result.stop_reason == "pause_turn":
                    continue
                tool_uses = [b for b in result.content if b.get("type") == "tool_use"]
                if not tool_uses:
                    break
                if result.stop_reason == "max_tokens":
                    # The last tool input may be cut off; running it would act on partial data.
                    raise LLMError("The response was cut off mid-action. Try a smaller batch.")

                results: list[dict[str, Any]] = []
                async for event in self._run_tools(files, tool_uses, schemas, results):
                    if event["type"] == "tool_end":
                        transcript.append({**event, "type": "tool"})
                    yield event

                question = self._pending_question(tool_uses, results)
                if question is not None:
                    pending = question
                    choice = {"type": "choice", **question["choice"]}
                    transcript.append(choice)
                    yield choice
                    break
                messages.append({"role": "user", "content": results})
            else:
                note = "Stopped after reaching the step limit for one turn. Say 'continue' to go on."
                transcript.append({"type": "notice", "text": note})
                yield {"type": "notice", "text": note}
        except LLMError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        session.messages = messages
        session.transcript.extend(transcript)
        session.pending = pending
        if not session.title:
            session.title = make_title(text, len(uploaded))
        self._store.save(session)
        yield {"type": "done", "title": session.title}

    def _user_turn(
        self, session: Session, files: SessionFiles, text: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the user message; if a question was pending, the text answers it."""
        if session.pending is None:
            return {"role": "user", "content": build_user_message(files, text, results)}
        answer = text.strip() or "(no answer)"
        content: list[dict[str, Any]] = [
            *session.pending["results"],
            {"type": "tool_result", "tool_use_id": session.pending["tool_use_id"], "content": answer},
        ]
        if results:
            content.append({"type": "text", "text": build_user_message(files, "", results)})
        return {"role": "user", "content": content}

    async def _model_call(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | TurnResult]:
        """One model response, re-issued if a streamed tool input was unparseable JSON."""
        for attempt in range(MAX_MALFORMED_RETRIES + 1):
            try:
                async for item in self._provider.stream_turn(system=system, messages=messages, tools=tools):
                    yield item
                return
            except ValueError:
                # Raised by the SDK's stream parser, before the tool call exists,
                # so there is nothing to answer; ask for the turn again.
                logger.warning("Malformed tool input from the model (attempt %d)", attempt + 1)
        raise LLMError("The model kept producing malformed tool calls. Please try again.")

    async def _local_tool(self, files: SessionFiles, block: dict[str, Any]) -> ToolResult:
        """Run describe_image or ENHANCE_CAPTION for a path inside this session's folders."""
        tool_input = block["input"]
        parts = tool_input["path"].replace("\\", "/").strip("/").split("/")
        if len(parts) < 2 or parts[-2] not in ("uploads", "outputs"):
            return ToolResult("path must be an image inside upload_dir or output_dir", is_error=True)
        kind: FileKind = "uploads" if parts[-2] == "uploads" else "outputs"
        name = parts[-1]
        try:
            if block["name"] == DESCRIBE_TOOL:
                return ToolResult(await self._pipeline.describe(files, kind, name))
            enhanced = await self._pipeline.enhance(
                files, kind, name, tool_input.get("mode", "improved"), tool_input.get("guidance", "")
            )
            return ToolResult(json.dumps(enhanced, ensure_ascii=False))
        except FileNotFoundError:
            return ToolResult(f"image not found: {name}", is_error=True)
        except (LLMError, InvalidFileError) as exc:
            return ToolResult(str(exc), is_error=True)

    async def _run_tools(
        self,
        files: SessionFiles,
        tool_uses: list[dict[str, Any]],
        schemas: dict[str, dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> AsyncIterator[Event]:
        """Run every tool call concurrently, appending tool_result blocks to ``results``."""
        calls = [b for b in tool_uses if b["name"] != ASK_USER_TOOL]
        for block in calls:
            yield {"type": "tool_start", "id": block["id"], "name": block["name"], "input": block["input"]}

        async def run(block: dict[str, Any]) -> tuple[dict[str, Any], ToolResult]:
            error = _validate(block, schemas)
            if error is not None:
                return block, ToolResult(error, is_error=True)
            if block["name"] in LOCAL_TOOLS:
                return block, await self._local_tool(files, block)
            return block, await self._toolbox.call(block["name"], block["input"])

        outcomes: dict[str, ToolResult] = {}
        for finished in asyncio.as_completed([run(block) for block in calls]):
            block, outcome = await finished
            outcomes[block["id"]] = outcome
            yield {
                "type": "tool_end",
                "id": block["id"],
                "name": block["name"],
                "input": block["input"],
                "ok": not outcome.is_error,
                "summary": _preview(outcome.text),
            }
        # Results go back in the order Claude made the calls.
        for block in calls:
            outcome = outcomes[block["id"]]
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": outcome.text,
                    "is_error": outcome.is_error,
                }
            )

    def _pending_question(
        self, tool_uses: list[dict[str, Any]], results: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """If Claude asked the user something, return the state needed to resume later.

        Invalid or surplus questions are answered with an error result straight away.
        """
        question: dict[str, Any] | None = None
        for block in tool_uses:
            if block["name"] != ASK_USER_TOOL:
                continue
            error = _validate(block, {ASK_USER_TOOL: ASK_USER_DEFINITION["input_schema"]})
            if error is None and question is not None:
                error = "Only one question can be asked at a time."
            if error is not None:
                results.append(
                    {"type": "tool_result", "tool_use_id": block["id"], "content": error, "is_error": True}
                )
                continue
            question = block
        if question is None:
            return None
        return {
            "tool_use_id": question["id"],
            "results": results,
            "choice": {
                "id": question["id"],
                "question": question["input"]["question"],
                "options": question["input"]["options"],
            },
        }


def _validate(block: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> str | None:
    """Check a tool input against its schema; return an error message or ``None``.

    With eager input streaming the API does not validate inputs, so a truncated
    or mistyped input can arrive here looking like a normal object.
    """
    schema = schemas.get(block["name"])
    if schema is None:
        return f"Unknown tool: {block['name']}"
    try:
        jsonschema.validate(block["input"], schema)
    except jsonschema.ValidationError as exc:
        return json.dumps({"INVALID_JSON": json.dumps(block["input"]), "reason": exc.message})
    return None


def _preview(text: str) -> str:
    return text if len(text) <= RESULT_PREVIEW_CHARS else text[:RESULT_PREVIEW_CHARS] + "…"
