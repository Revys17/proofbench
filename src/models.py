import asyncio
import logging

import anthropic
from anthropic.types import Message, MessageParam, ToolParam

log = logging.getLogger(__name__)


class LLMClient:
    """Async Anthropic API wrapper with retry logic."""

    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

    @classmethod
    def create(cls) -> "LLMClient":
        return cls(anthropic.AsyncAnthropic())

    async def send(
        self,
        *,
        model: str,
        system: str,
        messages: list[MessageParam],
        tools: list[ToolParam],
        max_tokens: int = 16384,
    ) -> Message:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await self._client.messages.create(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
            except anthropic.RateLimitError as exc:
                last_error = exc
                wait = 2 ** attempt
                log.warning("Rate limited (attempt %d/3), retrying in %ds", attempt + 1, wait)
                await asyncio.sleep(wait)
            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = exc
                    wait = 2 ** attempt
                    log.warning("Server error %d (attempt %d/3), retrying in %ds", exc.status_code, attempt + 1, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_error  # type: ignore[misc]
