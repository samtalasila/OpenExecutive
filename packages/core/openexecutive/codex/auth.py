"""Managed ChatGPT login through OpenAI's official Codex App Server.

This module does not implement OAuth or process credentials. The official SDK
owns device authorization, credential persistence, and refresh; this manager
only owns the process-local device-login handle required for status/cancel.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

from openexecutive.codex.models import (
    CodexAuthConflict,
    CodexAuthStatus,
    CodexAuthUnavailable,
    CodexDeviceLogin,
    CodexNoActiveLogin,
)
from openexecutive.codex.runtime import create_official_codex_client

logger = logging.getLogger(__name__)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if isinstance(value, Enum) or isinstance(enum_value, str):
        return str(enum_value)
    return str(value)


class CodexAuthManager:
    """One-process owner of the official Codex App Server auth session."""

    def __init__(
        self, client_factory: Callable[[], Any] = create_official_codex_client
    ) -> None:
        self._client_factory = client_factory
        self._client: Any = None
        self._active_login: Any = None
        self._login_task: asyncio.Task[None] | None = None
        self._last_error: str | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            client = self._client_factory()
            await client.__aenter__()
        except Exception as exc:
            logger.exception("Codex App Server initialization failed")
            raise CodexAuthUnavailable(
                "Codex is unavailable. Check the server logs and runtime installation."
            ) from exc
        self._client = client
        self._last_error = None
        return client

    async def _discard_client(self) -> None:
        """Drop a failed App Server so the next request can restart it."""
        client = self._client
        self._client = None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

    @staticmethod
    def _connected_status(account_response: Any) -> CodexAuthStatus | None:
        account = getattr(account_response, "account", None)
        if account is None:
            return None
        root = getattr(account, "root", account)
        return CodexAuthStatus(
            state="connected",
            auth_mode=_enum_value(getattr(root, "type", None)),
            email=getattr(root, "email", None),
            plan_type=_enum_value(getattr(root, "plan_type", None)),
        )

    async def status(self) -> CodexAuthStatus:
        async with self._lock:
            if self._active_login is not None:
                return CodexAuthStatus(
                    state="pending",
                    login_id=self._active_login.login_id,
                    verification_url=self._active_login.verification_url,
                    user_code=self._active_login.user_code,
                )
            try:
                account_response = await (await self._get_client()).account()
            except CodexAuthUnavailable as exc:
                return CodexAuthStatus(state="unavailable", error=str(exc))
            except Exception:
                logger.exception("Codex account status check failed")
                await self._discard_client()
                return CodexAuthStatus(
                    state="error",
                    error="Could not read Codex account status. Check the server logs.",
                )

            connected = self._connected_status(account_response)
            if connected is not None:
                self._last_error = None
                return connected
            if self._last_error:
                return CodexAuthStatus(state="error", error=self._last_error)
            return CodexAuthStatus(state="disconnected")

    async def start_device_login(self) -> CodexDeviceLogin:
        async with self._lock:
            if self._active_login is not None:
                raise CodexAuthConflict("A Codex login is already in progress.")

            client = await self._get_client()
            try:
                connected = self._connected_status(await client.account())
                if connected is not None:
                    raise CodexAuthConflict("Codex is already authenticated.")
                handle = await client.login_chatgpt_device_code()
            except CodexAuthConflict:
                raise
            except Exception as exc:
                logger.exception("Could not start Codex device-code login")
                await self._discard_client()
                raise CodexAuthUnavailable(
                    "Could not start ChatGPT sign-in. Check the server logs."
                ) from exc

            self._active_login = handle
            self._last_error = None
            self._login_task = asyncio.create_task(
                self._watch_login(handle), name="codex-device-login"
            )
            logger.info("Codex device-code login started")
            return CodexDeviceLogin(
                login_id=handle.login_id,
                verification_url=handle.verification_url,
                user_code=handle.user_code,
            )

    async def _watch_login(self, handle: Any) -> None:
        watcher_failed = False
        try:
            completed = await handle.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Codex device-code login watcher failed")
            success = False
            watcher_failed = True
        else:
            success = bool(getattr(completed, "success", False))
            if not success:
                logger.warning(
                    "Codex device-code login failed: %s",
                    getattr(completed, "error", None) or "unknown error",
                )

        async with self._lock:
            if self._active_login is not handle:
                return
            self._active_login = None
            self._login_task = None
            self._last_error = (
                None if success else "ChatGPT sign-in did not complete. Please try again."
            )
            if watcher_failed:
                await self._discard_client()
        if success:
            logger.info("Codex device-code login completed")

    async def cancel_device_login(self) -> str:
        async with self._lock:
            handle = self._active_login
            if handle is None:
                raise CodexNoActiveLogin("No Codex login is in progress.")
            try:
                response = await handle.cancel()
            except Exception as exc:
                task = self._login_task
                self._active_login = None
                self._login_task = None
                self._last_error = None
                await self._discard_client()
                if task is not None and not task.done():
                    task.cancel()
                logger.exception("Could not cancel Codex device-code login")
                raise CodexAuthUnavailable(
                    "Could not cancel ChatGPT sign-in. Check the server logs."
                ) from exc
            task = self._login_task
            self._active_login = None
            self._login_task = None
            self._last_error = None
            if task is not None and not task.done():
                task.cancel()
            status = _enum_value(getattr(response, "status", None)) or "canceled"
            logger.info("Codex device-code login cancellation result: %s", status)
            return status

    async def close(self) -> None:
        async with self._lock:
            client = self._client
            task = self._login_task
            self._client = None
            self._active_login = None
            self._login_task = None
            self._last_error = None
        if client is not None:
            try:
                await client.close()
            except Exception:
                logger.exception("Codex App Server shutdown failed")
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


_manager = CodexAuthManager()


def get_codex_auth_manager() -> CodexAuthManager:
    return _manager


async def close_codex_auth_manager() -> None:
    await _manager.close()
