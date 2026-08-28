"""Isolated process configuration for OpenAI's official Codex App Server."""
from __future__ import annotations

import os
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


def create_official_codex_client() -> Any:
    """Create the lazy App Server client with an isolated credential home."""
    # Imported lazily so API import and test collection do not require the
    # platform-specific Codex binary.
    from openai_codex import AsyncCodex, CodexConfig

    from openexecutive.config import get_settings

    settings = get_settings()
    codex_home = settings.codex_home_path or (
        settings.company_profile_path.parent / ".codex"
    )
    codex_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    codex_home.chmod(0o700)

    return AsyncCodex(
        CodexConfig(
            client_name="open_executive",
            client_title="Open Executive",
            client_version="0.1.0",
            env=codex_child_environment(str(codex_home)),
            experimental_api=False,
        )
    )
