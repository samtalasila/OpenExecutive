"""Isolated process configuration for OpenAI's official Codex App Server."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# Codex receives only runtime variables needed to start and make TLS requests.
# In particular, no deployment credentials are inherited from Open Executive.
_SAFE_CODEX_ENVIRONMENT = (
    "PATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


def codex_child_environment(codex_home: str) -> dict[str, str]:
    """Return a minimal environment for the App Server subprocess."""
    environment = {"CODEX_HOME": codex_home, "HOME": codex_home}
    for name in _SAFE_CODEX_ENVIRONMENT:
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _official_codex_config(*, experimental_api: bool, cwd: str | None = None) -> Any:
    """Build the shared, isolated App Server configuration.

    The auth manager uses only stable account methods. Provider turns opt into
    the App Server's experimental dynamic-tool protocol, but still inherit the
    same private credential root and minimal child environment.
    """
    # Imported lazily so API import and test collection do not require the
    # platform-specific Codex binary.
    from openai_codex import CodexConfig

    from openexecutive.config import get_settings

    settings = get_settings()
    codex_home = settings.codex_home_path or (
        settings.company_profile_path.parent / ".codex"
    )
    codex_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    codex_home.chmod(0o700)
    return CodexConfig(
        client_name="open_executive",
        client_title="Open Executive",
        client_version="0.1.0",
        env=codex_child_environment(str(codex_home)),
        cwd=cwd,
        experimental_api=experimental_api,
    )


def create_official_codex_client() -> Any:
    """Create the lazy App Server client used only for account authentication."""
    from openai_codex import AsyncCodex

    return AsyncCodex(_official_codex_config(experimental_api=False))


def create_official_codex_app_server(
    approval_handler: Callable[[str, dict[str, Any] | None], dict[str, Any]],
    workspace: str,
) -> Any:
    """Create one restricted App Server for an ephemeral provider turn.

    This deliberately returns the low-level sync client: its reader thread is
    the only SDK surface that can answer App Server ``item/tool/call`` requests.
    Provider code offloads its blocking methods so FastAPI's event loop remains
    responsive.
    """
    from openai_codex.client import CodexClient

    return CodexClient(
        _official_codex_config(experimental_api=True, cwd=workspace),
        approval_handler=approval_handler,
    )
