"""Authenticated Codex model discovery tests."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from openexecutive.codex import catalog


class _CatalogClient:
    def __init__(self, account: Any) -> None:
        self.account = account
        self.closed = False

    def start(self) -> None:
        return None

    def initialize(self) -> None:
        return None

    def account_read(self) -> Any:
        return self.account

    def model_list(self) -> Any:
        return SimpleNamespace(
            data=[
                SimpleNamespace(model="gpt-5.4"),
                SimpleNamespace(model="gpt-5.4"),
                SimpleNamespace(model="gpt-5.4-mini"),
            ]
        )

    def close(self) -> None:
        self.closed = True


def test_discovery_returns_only_connected_chatgpt_models(monkeypatch: Any) -> None:
    client = _CatalogClient(
        SimpleNamespace(account=SimpleNamespace(root=SimpleNamespace(type="chatgpt")))
    )
    monkeypatch.setattr(catalog, "create_official_codex_app_server", lambda *_args: client)

    assert asyncio.run(catalog.discover_models()) == ["codex/gpt-5.4", "codex/gpt-5.4-mini"]
    assert client.closed is True


def test_discovery_hides_catalog_without_chatgpt_account(monkeypatch: Any) -> None:
    client = _CatalogClient(SimpleNamespace(account=None))
    monkeypatch.setattr(catalog, "create_official_codex_app_server", lambda *_args: client)

    assert asyncio.run(catalog.discover_models()) == []
    assert client.closed is True
