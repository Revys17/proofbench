from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .costs import CostTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    tool_calls: tuple[ToolCall, ...]
    text: str
    stop_reason: str  # "end_turn" or "tool_use"
    usage: Usage = Usage()


# Provider-agnostic tool definition (stored in Anthropic format)
ToolDef = dict[str, Any]


class LLMClient:
    """Provider-agnostic LLM client interface."""

    def __init__(self, max_concurrent: int = 20, cost_tracker: CostTracker | None = None) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cost_tracker = cost_tracker

    async def send(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[ToolDef],
        max_tokens: int = 16384,
    ) -> LLMResponse:
        async with self._semaphore:
            response = await self._send(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            )
        if self._cost_tracker and (response.usage.input_tokens or response.usage.output_tokens):
            self._cost_tracker.record(model, response.usage.input_tokens, response.usage.output_tokens)
        return response

    async def _send(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[ToolDef],
        max_tokens: int = 16384,
    ) -> LLMResponse:
        raise NotImplementedError

    def format_assistant(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Format assistant response as message(s) to append to history."""
        raise NotImplementedError

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        """Format tool results as message(s) to append to history."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Async Anthropic API client."""

    def __init__(self, max_concurrent: int = 20, cost_tracker: CostTracker | None = None) -> None:
        super().__init__(max_concurrent, cost_tracker)
        import anthropic
        self._client = anthropic.AsyncAnthropic()
        self._anthropic = anthropic

    async def _send(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[ToolDef],
        max_tokens: int = 16384,
    ) -> LLMResponse:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.messages.create(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
                return self._normalize(resp)
            except self._anthropic.RateLimitError as exc:
                last_error = exc
                wait = 2 ** attempt
                log.warning("Anthropic rate limited (attempt %d/3), retrying in %ds", attempt + 1, wait)
                await asyncio.sleep(wait)
            except self._anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = exc
                    wait = 2 ** attempt
                    log.warning("Anthropic server error %d (attempt %d/3), retrying in %ds",
                                exc.status_code, attempt + 1, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_error  # type: ignore[misc]

    def _normalize(self, resp: Any) -> LLMResponse:
        tool_calls = tuple(
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in resp.content if b.type == "tool_use"
        )
        text_parts = [b.text for b in resp.content if b.type == "text"]
        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        return LLMResponse(
            tool_calls=tool_calls,
            text="\n".join(text_parts),
            stop_reason=resp.stop_reason,
            usage=usage,
        )

    def format_assistant(self, response: LLMResponse) -> list[dict[str, Any]]:
        # Rebuild Anthropic content blocks from our normalized response
        content: list[dict[str, Any]] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        return [{"role": "assistant", "content": content}]

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # Anthropic: all tool results in one user message
        return [{"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            }
            for r in results
        ]}]


class OpenAIClient(LLMClient):
    """Async OpenAI API client."""

    def __init__(self, max_concurrent: int = 20, cost_tracker: CostTracker | None = None) -> None:
        super().__init__(max_concurrent, cost_tracker)
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()

    async def _send(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[ToolDef],
        max_tokens: int = 16384,
    ) -> LLMResponse:
        oai_tools = [self._convert_tool(t) for t in tools]
        oai_messages = [{"role": "system", "content": system}, *messages]

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                import openai as oai_module
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=oai_messages,
                    tools=oai_tools,
                    max_tokens=max_tokens,
                )
                return self._normalize(resp)
            except Exception as exc:
                # openai.RateLimitError or server errors
                status = getattr(exc, "status_code", 0)
                if status == 429 or status >= 500:
                    last_error = exc
                    wait = 2 ** attempt
                    log.warning("OpenAI error %s (attempt %d/3), retrying in %ds",
                                status, attempt + 1, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _convert_tool(tool: ToolDef) -> dict[str, Any]:
        """Convert Anthropic-format tool def to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        }

    def _normalize(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = ()
        if msg.tool_calls:
            tool_calls = tuple(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                )
                for tc in msg.tool_calls
            )

        usage = Usage()
        if resp.usage:
            usage = Usage(
                input_tokens=resp.usage.prompt_tokens or 0,
                output_tokens=resp.usage.completion_tokens or 0,
            )

        stop = "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
        return LLMResponse(
            tool_calls=tool_calls,
            text=msg.content or "",
            stop_reason=stop,
            usage=usage,
        )

    def format_assistant(self, response: LLMResponse) -> list[dict[str, Any]]:
        msg: dict[str, Any] = {"role": "assistant", "content": response.text or None}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in response.tool_calls
            ]
        return [msg]

    def format_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # OpenAI: each tool result is a separate message
        return [
            {"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
            for r in results
        ]


def create_client(
    provider: str,
    max_concurrent: int = 20,
    cost_tracker: CostTracker | None = None,
) -> LLMClient:
    """Create an LLM client for the given provider."""
    if provider == "anthropic":
        return AnthropicClient(max_concurrent, cost_tracker)
    if provider == "openai":
        return OpenAIClient(max_concurrent, cost_tracker)
    raise ValueError(f"Unknown provider: {provider}")
