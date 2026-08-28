"""Public models and expected errors for the managed Codex login flow."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

CodexAuthState = Literal[
    "unavailable", "disconnected", "pending", "connected", "error"
]


class CodexAuthStatus(BaseModel):
    state: CodexAuthState
    auth_mode: str | None = None
    email: str | None = None
    plan_type: str | None = None
    login_id: str | None = None
    verification_url: str | None = None
    user_code: str | None = None
    error: str | None = None


class CodexDeviceLogin(BaseModel):
    login_id: str
    verification_url: str
    user_code: str


class CodexAuthError(RuntimeError):
    """Base class for safe, expected connection-flow failures."""


class CodexAuthConflict(CodexAuthError):
    """A login is already active or Codex is already authenticated."""


class CodexAuthUnavailable(CodexAuthError):
    """The official Codex runtime could not be initialized."""


class CodexNoActiveLogin(CodexAuthError):
    """Cancellation was requested without an active device-code flow."""
