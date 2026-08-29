"""Authenticated model discovery for ChatGPT-backed Codex routing."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from typing import Any

from openexecutive.codex.runtime import create_official_codex_app_server
from openexecutive.providers.codex_provider import CODEX_MODEL_PREFIX


def _deny_local_capabilities(_method: str, _params: dict[str, Any] | None) -> dict[str, Any]:
    """Catalog discovery must never approve an App Server local operation."""
    return {"decision": "decline"}


def _account_is_chatgpt(account_response: Any) -> bool:
    account = getattr(account_response, "account", None)
    if account is None:
        return False
    root = getattr(account, "root", account)
    mode = getattr(root, "type", None)
    return getattr(mode, "value", mode) == "chatgpt"


async def discover_models() -> list[str]:
    """Return models the connected ChatGPT subscription currently exposes.

    This intentionally makes a short-lived App Server process instead of
    sharing a client with an in-flight generation. App Server blocks its reader
    thread while a dynamic tool callback is answered; sharing it with model
    discovery (or a nested specialist) can deadlock that callback.
    """
    workspace = tempfile.mkdtemp(prefix="openexecutive-codex-catalog-")
    client: Any = None
    try:
        client = create_official_codex_app_server(_deny_local_capabilities, workspace)
        await asyncio.to_thread(client.start)
        await asyncio.to_thread(client.initialize)
        account = await asyncio.to_thread(client.account_read)
        if not _account_is_chatgpt(account):
            return []
        response = await asyncio.to_thread(client.model_list)
        models: list[str] = []
        for item in getattr(response, "data", []):
            model = getattr(item, "model", None)
            if isinstance(model, str) and model and f"{CODEX_MODEL_PREFIX}{model}" not in models:
                models.append(f"{CODEX_MODEL_PREFIX}{model}")
        return models
    finally:
        if client is not None:
            await asyncio.to_thread(client.close)
        shutil.rmtree(workspace, ignore_errors=True)
