from __future__ import annotations

import pytest

from captioner_agent.llm import AnthropicProvider, LLMError
from captioner_agent.settings import Settings

pytestmark = pytest.mark.anyio


async def test_missing_api_key_is_reported_clearly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    provider = AnthropicProvider(settings)

    with pytest.raises(LLMError, match="No Anthropic API key found"):
        async for _ in provider.stream_turn(
            system="s", messages=[{"role": "user", "content": "hi"}], tools=[]
        ):
            pass
