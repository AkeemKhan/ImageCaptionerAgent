"""The language model behind the agent, behind a small interface."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from .settings import Settings

# Lets the API pick a fallback model by refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Room for the model's reasoning plus a caption of a few sentences.
CAPTION_MAX_TOKENS = 4000
# One word is the answer; the rest is room for the model's brief reasoning.
ROUTER_MAX_TOKENS = 1500


class LLMError(Exception):
    """A model request failed; the message is safe to show to the user."""


@dataclass(frozen=True)
class TurnResult:
    """One completed model response."""

    content: list[dict[str, Any]]
    stop_reason: str | None
    refusal_explanation: str | None = None


class LLMProvider(Protocol):
    def stream_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | TurnResult]:
        """Yield text as it is generated, then exactly one ``TurnResult``."""
        ...

    async def caption_image(self, *, image: bytes, media_type: str, instructions: str) -> str:
        """Look at one image and return its caption, following ``instructions``."""
        ...

    async def enhance_caption(self, *, caption: str, instructions: str) -> str:
        """Rewrite an existing caption, following ``instructions``. Text only; no image is sent."""
        ...

    async def wants_manual_captioning(self, *, text: str, instructions: str) -> bool:
        """Whether ``text`` asks to caption images by hand, or supplies the captions itself."""
        ...


class AnthropicProvider:
    """Claude through the Anthropic API, authenticated with ``ANTHROPIC_API_KEY``."""

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._settings = settings
        self._client = client or anthropic.AsyncAnthropic()

    def _fallback_options(self) -> dict[str, Any]:
        if not self._settings.refusal_fallbacks:
            return {}
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    async def stream_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[str | TurnResult]:
        with self._api_errors():
            async with self._client.beta.messages.stream(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                system=system,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                output_config={"effort": self._settings.effort},
                # Cache the growing prefix (tools, system prompt, earlier turns).
                cache_control={"type": "ephemeral"},
                **self._fallback_options(),
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield event.text
                message = await stream.get_final_message()

        details = message.stop_details
        yield TurnResult(
            content=[block.to_dict() for block in message.content],
            stop_reason=message.stop_reason,
            refusal_explanation=details.explanation if details is not None else None,
        )

    async def caption_image(self, *, image: bytes, media_type: str, instructions: str) -> str:
        messages: list[Any] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(image).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": "Write the caption for this image."},
                ],
            }
        ]
        return await self._caption_reply(instructions, messages)

    async def enhance_caption(self, *, caption: str, instructions: str) -> str:
        return await self._caption_reply(instructions, [{"role": "user", "content": caption}])

    async def wants_manual_captioning(self, *, text: str, instructions: str) -> bool:
        """Classify a message with a small, cheap model. Any doubt counts as "no".

        A wrong "no" only means the automatic pipeline runs, which asks before
        sending anything anywhere; a wrong "yes" would skip captioning the user
        wanted. So errors, cut-off replies and anything unclear return False.
        """
        with self._api_errors():
            message = await self._client.messages.create(
                model=self._settings.router_model,
                max_tokens=ROUTER_MAX_TOKENS,
                system=instructions,
                messages=[{"role": "user", "content": text}],
                output_config={"effort": "low"},
            )
        if message.stop_reason != "end_turn":
            return False
        reply = "".join(block.text for block in message.content if block.type == "text")
        return reply.strip().lower().startswith("manual")

    async def _caption_reply(self, instructions: str, messages: list[Any]) -> str:
        """One short request whose whole reply is a caption."""
        with self._api_errors():
            message = await self._client.beta.messages.create(
                model=self._settings.model,
                max_tokens=CAPTION_MAX_TOKENS,
                system=instructions,
                messages=messages,
                # One caption per request does not need deep reasoning.
                output_config={"effort": "low"},
                **self._fallback_options(),
            )
        if message.stop_reason == "refusal":
            raise LLMError("Claude declined this caption request.")
        if message.stop_reason == "max_tokens":
            raise LLMError("The caption was cut off before it finished.")
        return "".join(block.text for block in message.content if block.type == "text").strip()

    @contextmanager
    def _api_errors(self) -> Iterator[None]:
        """Translate SDK failures into ``LLMError`` messages a user can act on."""
        try:
            yield
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in your .env file."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("This API key is not allowed to use the configured model.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model not found: {self._settings.model}. Check settings.yaml.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the Anthropic API. Wait a moment and try again.") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise LLMError("The Anthropic API is having trouble. Try again shortly.") from exc
            raise LLMError(f"The Anthropic API rejected the request: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API. Check your connection.") from exc
        except TypeError as exc:
            # The SDK raises a bare TypeError when it finds no credentials at all.
            if "authentication method" not in str(exc):
                raise
            raise LLMError(
                "No Anthropic API key found. Add ANTHROPIC_API_KEY to the .env file and restart the app."
            ) from exc
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Could not call the Anthropic API: {exc}") from exc
