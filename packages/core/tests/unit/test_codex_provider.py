"""Codex App Server provider tests without a real subscription or subprocess."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.providers.codex_provider import (
    CODEX_MODEL_PREFIX,
    CodexProvider,
    _DeferredToolBridge,
)


def _chatgpt_account() -> Any:
    return SimpleNamespace(account=SimpleNamespace(root=SimpleNamespace(type="chatgpt")))


class _FakeCodexClient:
    def __init__(self, approval_handler: Any, workspace: str, *, call_tool: bool = False) -> None:
        self._approval_handler = approval_handler
        self.workspace = workspace
        self.call_tool = call_tool
        self.started = False
        self.closed = False
        self.thread_params: dict[str, Any] | None = None
        self.turn_input: list[dict[str, Any]] | None = None
        self.turn_params: dict[str, Any] | None = None
        self.tool_response: dict[str, Any] | None = None
        self._notifications: list[Any] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def initialize(self) -> Any:
        return SimpleNamespace()

    def account_read(self, _params: Any = None) -> Any:
        return _chatgpt_account()

    def thread_start(self, params: dict[str, Any]) -> Any:
        self.thread_params = params
        return SimpleNamespace(thread=SimpleNamespace(id="thread-1"))

    def turn_start(
        self,
        _thread_id: str,
        input_items: list[dict[str, Any]],
        params: dict[str, Any] | None = None,
    ) -> Any:
        self.turn_input = input_items
        self.turn_params = params
        if self.call_tool:
            self.tool_response = self._approval_handler(
                "item/tool/call",
                {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "callId": "call-1",
                    "tool": "consult_specialist",
                    "arguments": {"specialist": "cfo", "query": "cash"},
                },
            )
        self._notifications = [
            SimpleNamespace(
                method="item/agentMessage/delta",
                payload=SimpleNamespace(delta="draft that must not leak"),
            ),
            SimpleNamespace(
                method="thread/tokenUsage/updated",
                payload=SimpleNamespace(
                    token_usage=SimpleNamespace(
                        last=SimpleNamespace(
                            input_tokens=12,
                            output_tokens=7,
                            cached_input_tokens=0,
                            cache_write_input_tokens=0,
                        )
                    )
                ),
            ),
            SimpleNamespace(
                method="turn/completed",
                payload=SimpleNamespace(turn=SimpleNamespace(status="completed", error=None)),
            ),
        ]
        return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

    def next_turn_notification(self, _turn_id: str) -> Any:
        return self._notifications.pop(0)

    def unregister_turn_notifications(self, _turn_id: str) -> None:
        return None


def _provider(*, call_tool: bool = False) -> CodexProvider:
    def factory(handler: Any, workspace: str) -> _FakeCodexClient:
        return _FakeCodexClient(handler, workspace, call_tool=call_tool)

    return CodexProvider(factory)


async def _collect(provider: CodexProvider, **kwargs: Any) -> tuple[list[Any], Any]:
    async with provider.messages_stream(**kwargs) as stream:
        events = [event async for event in stream]
        return events, await stream.get_final_message()


def test_codex_stream_translates_text_usage_and_cleans_ephemeral_workspace() -> None:
    provider = _provider()
    events, final = asyncio.run(
        _collect(
            provider,
            model="codex/gpt-5.4",
            max_tokens=20,
            system=[{"type": "text", "text": "You are OE", "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    deltas = [event.delta.text for event in events if event.type == "content_block_delta"]
    assert deltas == ["draft that must not leak"]
    assert final.stop_reason == "end_turn"
    assert final.content[0].text == "draft that must not leak"
    assert final.usage.input_tokens == 12
    assert final.usage.output_tokens == 7


def test_codex_dynamic_tool_is_returned_to_existing_oe_tool_loop() -> None:
    holder: dict[str, _FakeCodexClient] = {}

    def factory(handler: Any, workspace: str) -> _FakeCodexClient:
        client = _FakeCodexClient(handler, workspace, call_tool=True)
        holder["client"] = client
        return client

    events, final = asyncio.run(
        _collect(
            CodexProvider(factory),
            model=f"{CODEX_MODEL_PREFIX}gpt-5.4",
            max_tokens=20,
            system=[{"type": "text", "text": "You are OE"}],
            tools=[
                {
                    "name": "consult_specialist",
                    "description": "Ask a specialist",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            messages=[{"role": "user", "content": "How is cash?"}],
        )
    )

    client = holder["client"]
    assert client.thread_params is not None
    assert client.thread_params["dynamicTools"] == [
        {
            "type": "function",
            "name": "consult_specialist",
            "description": "Ask a specialist",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    assert client.tool_response is not None
    assert client.tool_response["success"] is True

    # No pre-tool draft reaches the user; Executive receives the normal
    # Anthropic-shape function call and applies its existing authority checks.
    assert not [event for event in events if event.type == "content_block_delta"]
    assert final.stop_reason == "tool_use"
    assert [(block.name, block.input) for block in final.content] == [
        ("consult_specialist", {"specialist": "cfo", "query": "cash"})
    ]


def test_codex_request_uses_restricted_ephemeral_runtime() -> None:
    holder: dict[str, _FakeCodexClient] = {}

    def factory(handler: Any, workspace: str) -> _FakeCodexClient:
        client = _FakeCodexClient(handler, workspace)
        holder["client"] = client
        return client

    provider = CodexProvider(factory)
    asyncio.run(
        _collect(
            provider,
            model="codex/gpt-5.4",
            max_tokens=20,
            system=[{"type": "text", "text": "System"}],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect this image"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                        },
                    ],
                }
            ],
        )
    )

    client = holder["client"]
    assert client.started and client.closed
    assert client.thread_params == {
        "model": "gpt-5.4",
        "ephemeral": True,
        "cwd": client.workspace,
        "approvalPolicy": "untrusted",
        "sandbox": "read-only",
        "developerInstructions": "System",
    }
    assert client.turn_input == [
        {"type": "text", "text": "<user>\ninspect this image\n</user>"},
        {"type": "image", "url": "data:image/png;base64,aGVsbG8="},
    ]
    assert client.turn_params == {
        "model": "gpt-5.4",
        "cwd": client.workspace,
        "approvalPolicy": "untrusted",
        "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
    }
    assert not Path(client.workspace).exists()


def test_codex_bridge_declines_local_and_undeclared_capabilities() -> None:
    bridge = _DeferredToolBridge({"consult_specialist"})

    assert bridge.handle("item/commandExecution/requestApproval", {}) == {"decision": "decline"}
    denied = bridge.handle(
        "item/tool/call",
        {"callId": "call-1", "tool": "shell", "arguments": {}},
    )
    assert denied["success"] is False
    assert bridge.calls() == []


def test_codex_model_requires_prefixed_slug() -> None:
    provider = _provider()
    with pytest.raises(ValueError, match="codex/<model>"):
        provider.messages_stream(model="gpt-5.4", messages=[])
