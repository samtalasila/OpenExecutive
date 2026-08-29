"""Codex App Server backend for Anthropic-shaped Open Executive calls.

Codex's App Server is a stateful JSON-RPC protocol rather than an
OpenAI-compatible HTTP endpoint.  This adapter starts an isolated, ephemeral
App Server thread for each OE completion, translates its streamed text into
the Anthropic stream shape, and exposes OE client tools through Codex's
experimental ``item/tool/call`` callback.

The callback intentionally *defers* tool execution to OE's existing tool-use
loop. It returns a synthetic acknowledgement to the App Server, records the
validated call, and the completed Anthropic-shaped message carries those calls as normal
``tool_use`` blocks.  The existing Executive, research, memory, and workflow
dispatchers then execute the authoritative handlers and send their normal
``tool_result`` turn.  This preserves OE's approval, audit, and authority
boundaries rather than giving the Codex subprocess direct access to them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any, Protocol

from openexecutive.codex.runtime import create_official_codex_app_server
from openexecutive.providers.feature_gate import FeatureSpec, apply_feature_gates

logger = logging.getLogger(__name__)

CODEX_MODEL_PREFIX = "codex/"

# Codex App Server has no Anthropic prompt-cache annotations or native
# Anthropic server-side web-search equivalent.  OE's client tools remain
# supported through the dynamic-tool bridge below.
CODEX_FEATURE_SPEC = FeatureSpec(
    supports_cache_control=False,
    supports_thinking=False,
    supports_web_search=False,
    supports_tool_use=True,
)


class _CodexClient(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def initialize(self) -> Any: ...

    def account_read(self, params: Any = None) -> Any: ...

    def thread_start(self, params: dict[str, Any]) -> Any: ...

    def turn_start(
        self, thread_id: str, input_items: list[dict[str, Any]], params: dict[str, Any] | None = None
    ) -> Any: ...

    def next_turn_notification(self, turn_id: str) -> Any: ...

    def unregister_turn_notifications(self, turn_id: str) -> None: ...


CodexClientFactory = Callable[[Callable[[str, dict[str, Any] | None], dict[str, Any]], str], _CodexClient]


def is_codex_model(model: str) -> bool:
    """Whether *model* is an Open Executive Codex model slug."""
    return model.startswith(CODEX_MODEL_PREFIX) and len(model) > len(CODEX_MODEL_PREFIX)


def codex_model_slug(model: str) -> str:
    """Remove OE's unambiguous ``codex/`` routing prefix."""
    if not is_codex_model(model):
        raise ValueError(f"Not a Codex model slug: {model!r}")
    return model[len(CODEX_MODEL_PREFIX) :]


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            chunks.append(block["text"])
        elif kind == "tool_use":
            chunks.append(
                "<tool_use name="
                f"{json.dumps(str(block.get('name', '')))} id={json.dumps(str(block.get('id', '')))}>"
                f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                "</tool_use>"
            )
        elif kind == "tool_result":
            chunks.append(
                "<tool_result tool_use_id="
                f"{json.dumps(str(block.get('tool_use_id', '')))}>"
                f"{_text_from_content(block.get('content'))}"
                "</tool_result>"
            )
    return "\n\n".join(chunks)


def _system_text(system: Any) -> str:
    return _text_from_content(system)


def _conversation_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        rendered.append(f"<{role}>\n{_text_from_content(message.get('content'))}\n</{role}>")
    return "\n\n".join(rendered)


def _image_inputs(messages: Any) -> list[dict[str, str]]:
    """Translate Anthropic base64 image blocks to App Server data URLs."""
    if not isinstance(messages, list):
        return []
    inputs: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            source = block.get("source")
            if not isinstance(source, dict):
                continue
            if source.get("type") == "base64":
                media_type = source.get("media_type")
                data = source.get("data")
                if isinstance(media_type, str) and isinstance(data, str):
                    inputs.append({"type": "image", "url": f"data:{media_type};base64,{data}"})
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                inputs.append({"type": "image", "url": source["url"]})
    return inputs


def _dynamic_tools(tools: Any) -> tuple[list[dict[str, Any]], set[str]]:
    """Translate only OE client tools to App Server dynamic-tool specs."""
    if not isinstance(tools, list):
        return [], set()
    specs: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            # Anthropic server tools (e.g. web_search) do not have an input
            # schema and cannot be granted access through this bridge.
            continue
        specs.append(
            {
                "type": "function",
                "name": name,
                "description": str(tool.get("description", "")),
                "inputSchema": schema,
            }
        )
        names.add(name)
    return specs, names


def _account_is_chatgpt(account_response: Any) -> bool:
    account = getattr(account_response, "account", None)
    if account is None:
        return False
    root = getattr(account, "root", account)
    mode = getattr(root, "type", None)
    value = getattr(mode, "value", mode)
    return value == "chatgpt"


def _usage_from_notification(payload: Any) -> SimpleNamespace:
    breakdown = getattr(getattr(payload, "token_usage", None), "last", None)
    if breakdown is None:
        return SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            cost=None,
        )
    return SimpleNamespace(
        input_tokens=int(getattr(breakdown, "input_tokens", 0) or 0),
        output_tokens=int(getattr(breakdown, "output_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(breakdown, "cache_write_input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(breakdown, "cached_input_tokens", 0) or 0),
        # ChatGPT subscription usage has no per-request USD cost.
        cost=None,
    )


class _DeferredToolBridge:
    """Validate and retain dynamic tool calls for OE's existing dispatchers."""

    def __init__(self, allowed_names: set[str]) -> None:
        self._allowed_names = allowed_names
        self._calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if method != "item/tool/call" or not isinstance(params, dict):
            # Never approve Codex's local shell/file operations.  OE tools are
            # the only capability exposed to the model.
            return {"decision": "decline"}

        name = params.get("tool")
        call_id = params.get("callId")
        arguments = params.get("arguments")
        if (
            not isinstance(name, str)
            or name not in self._allowed_names
            or not isinstance(call_id, str)
            or not isinstance(arguments, dict)
        ):
            return {
                "success": False,
                "contentItems": [
                    {"type": "inputText", "text": "This tool is not available to Open Executive."}
                ],
            }

        with self._lock:
            self._calls.append({"id": call_id, "name": name, "input": arguments})
        # The caller below intentionally discards the model's continuation
        # after a deferred call and returns an Anthropic tool_use message.
        # This is the only safe way to preserve OE's authority-gated handlers
        # without granting the subprocess direct access to them.
        return {
            # Acknowledge rather than fail the call: failure can make Codex
            # retry a side-effecting request before OE gets its normal
            # authority-gated turn to run it. The following model text is
            # discarded; OE returns the captured call as Anthropic tool_use.
            "success": True,
            "contentItems": [
                {
                    "type": "inputText",
                    "text": "Open Executive accepted this tool call and will continue the conversation.",
                }
            ],
        }

    @property
    def may_receive_tools(self) -> bool:
        return bool(self._allowed_names)

    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._calls)


class CodexProvider:
    """LLMProvider implementation backed by an authenticated Codex App Server."""

    def __init__(self, client_factory: CodexClientFactory = create_official_codex_app_server) -> None:
        self._client_factory = client_factory

    def messages_create(self, **kwargs: Any) -> Awaitable[Any]:
        return self._messages_create(kwargs)

    async def _messages_create(self, kwargs: dict[str, Any]) -> Any:
        async with self.messages_stream(**kwargs) as stream:
            async for _ in stream:
                pass
            return await stream.get_final_message()

    def messages_stream(self, **kwargs: Any) -> AbstractAsyncContextManager[Any]:
        request_timeout = kwargs.pop("timeout", None)
        model = kwargs.pop("model", "")
        if not isinstance(model, str) or not is_codex_model(model):
            raise ValueError("CodexProvider requires a codex/<model> slug")
        return _CodexStream(
            client_factory=self._client_factory,
            model=codex_model_slug(model),
            request=apply_feature_gates(CODEX_FEATURE_SPEC, kwargs),
            timeout_s=request_timeout,
        )


class _CodexStream:
    """Translate one App Server turn to an Anthropic-compatible async stream."""

    def __init__(
        self,
        *,
        client_factory: CodexClientFactory,
        model: str,
        request: dict[str, Any],
        timeout_s: float | None,
    ) -> None:
        self._client_factory = client_factory
        self._model = model
        self._request = request
        self._timeout_s = timeout_s
        self._workspace: str | None = None
        self._client: _CodexClient | None = None
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self._bridge: _DeferredToolBridge | None = None
        self._text: list[str] = []
        self._usage = _usage_from_notification(None)
        self._final: SimpleNamespace | None = None

    async def __aenter__(self) -> _CodexStream:
        self._workspace = tempfile.mkdtemp(prefix="openexecutive-codex-")
        dynamic_tools, tool_names = _dynamic_tools(self._request.get("tools"))
        self._bridge = _DeferredToolBridge(tool_names)
        self._client = self._client_factory(self._bridge.handle, self._workspace)
        try:
            await asyncio.to_thread(self._client.start)
            await asyncio.to_thread(self._client.initialize)
            account = await asyncio.to_thread(self._client.account_read)
            if not _account_is_chatgpt(account):
                raise RuntimeError(
                    "Codex model routing requires a connected ChatGPT subscription. "
                    "Connect it in Settings first."
                )

            developer_instructions = _system_text(self._request.get("system"))
            tool_choice = self._request.get("tool_choice")
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
                required_name = tool_choice.get("name")
                if isinstance(required_name, str) and required_name in tool_names:
                    developer_instructions = (
                        f"{developer_instructions}\n\n"
                        f"For this turn, call the `{required_name}` tool before answering."
                    )
            thread_params: dict[str, Any] = {
                "model": self._model,
                "ephemeral": True,
                "cwd": self._workspace,
                # Deny local network/file-write capabilities.  The callback
                # denies every local approval request as a second layer.
                "approvalPolicy": "untrusted",
                "sandbox": "read-only",
            }
            if developer_instructions:
                thread_params["developerInstructions"] = developer_instructions
            if dynamic_tools:
                # ``dynamicTools`` is experimental in App Server v0.147.0;
                # the runtime is initialized with experimental_api=True.
                thread_params["dynamicTools"] = dynamic_tools

            started_thread = await asyncio.to_thread(self._client.thread_start, thread_params)
            self._thread_id = str(started_thread.thread.id)
            input_items: list[dict[str, Any]] = [
                {"type": "text", "text": _conversation_text(self._request.get("messages"))}
            ]
            input_items.extend(_image_inputs(self._request.get("messages")))
            started_turn = await asyncio.to_thread(
                self._client.turn_start,
                self._thread_id,
                input_items,
                {
                    "model": self._model,
                    "cwd": self._workspace,
                    "approvalPolicy": "untrusted",
                    "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
                },
            )
            self._turn_id = str(started_turn.turn.id)
        except BaseException:
            await self._close()
            raise
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._close()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter_events()

    async def _next_notification(self) -> Any:
        if self._client is None or self._turn_id is None:
            raise RuntimeError("Codex stream was not started")
        waiter = asyncio.to_thread(self._client.next_turn_notification, self._turn_id)
        if self._timeout_s is None:
            return await waiter
        return await asyncio.wait_for(waiter, timeout=float(self._timeout_s))

    async def _iter_events(self) -> AsyncIterator[Any]:
        # A Codex model can emit explanatory text before requesting a dynamic
        # tool. OE's external tool loop must discard that pre-tool draft, so
        # buffer text for tool-capable turns until completion tells us whether
        # it was a final answer or a deferred tool request.
        defer_text = bool(self._bridge and self._bridge.may_receive_tools)
        text_started = False
        try:
            while True:
                notification = await self._next_notification()
                method = getattr(notification, "method", "")
                payload = getattr(notification, "payload", None)
                if method == "item/agentMessage/delta":
                    delta = getattr(payload, "delta", "")
                    if not isinstance(delta, str) or not delta:
                        continue
                    self._text.append(delta)
                    if not defer_text:
                        if not text_started:
                            text_started = True
                            yield SimpleNamespace(
                                type="content_block_start",
                                index=0,
                                content_block=SimpleNamespace(type="text", text=""),
                            )
                        yield SimpleNamespace(
                            type="content_block_delta",
                            index=0,
                            delta=SimpleNamespace(type="text_delta", text=delta),
                        )
                    continue
                if method == "thread/tokenUsage/updated":
                    self._usage = _usage_from_notification(payload)
                    continue
                if method != "turn/completed":
                    continue

                turn = getattr(payload, "turn", None)
                status = getattr(getattr(turn, "status", None), "value", getattr(turn, "status", None))
                if status == "failed":
                    error = getattr(turn, "error", None)
                    raise RuntimeError(getattr(error, "message", None) or "Codex turn failed")
                has_deferred_tool = bool(self._bridge and self._bridge.calls())
                self._final = self._build_final_message()
                if defer_text and not has_deferred_tool and self._text:
                    text_started = True
                    yield SimpleNamespace(
                        type="content_block_start",
                        index=0,
                        content_block=SimpleNamespace(type="text", text=""),
                    )
                    yield SimpleNamespace(
                        type="content_block_delta",
                        index=0,
                        delta=SimpleNamespace(type="text_delta", text="".join(self._text)),
                    )
                if text_started:
                    yield SimpleNamespace(type="content_block_stop", index=0)
                yield SimpleNamespace(
                    type="message_delta",
                    delta=SimpleNamespace(stop_reason=self._final.stop_reason, stop_sequence=None),
                    usage=self._usage,
                )
                yield SimpleNamespace(type="message_stop")
                return
        except TimeoutError as exc:
            raise TimeoutError("Codex turn timed out") from exc

    def _build_final_message(self) -> SimpleNamespace:
        tool_calls = self._bridge.calls() if self._bridge is not None else []
        content: list[SimpleNamespace] = []
        if not tool_calls and self._text:
            content.append(SimpleNamespace(type="text", text="".join(self._text)))
        content.extend(
            SimpleNamespace(type="tool_use", id=call["id"], name=call["name"], input=call["input"])
            for call in tool_calls
        )
        return SimpleNamespace(
            id=self._turn_id or "",
            type="message",
            role="assistant",
            model=f"{CODEX_MODEL_PREFIX}{self._model}",
            content=content,
            stop_reason="tool_use" if tool_calls else "end_turn",
            stop_sequence=None,
            usage=self._usage,
        )

    async def get_final_message(self) -> Any:
        if self._final is None:
            async for _ in self._iter_events():
                pass
        if self._final is None:
            self._final = self._build_final_message()
        return self._final

    async def _close(self) -> None:
        client, self._client = self._client, None
        turn_id, self._turn_id = self._turn_id, None
        if client is not None:
            if turn_id:
                try:
                    await asyncio.to_thread(client.unregister_turn_notifications, turn_id)
                except Exception:
                    logger.debug("Could not unregister Codex turn", exc_info=True)
            try:
                await asyncio.to_thread(client.close)
            except Exception:
                logger.debug("Could not close Codex App Server", exc_info=True)
        workspace, self._workspace = self._workspace, None
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
