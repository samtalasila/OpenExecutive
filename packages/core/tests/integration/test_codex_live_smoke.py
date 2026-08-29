"""Opt-in authenticated smoke test for the official Codex App Server.

Run only on a machine where the principal has already connected ChatGPT:

    CODEX_LIVE_SMOKE_MODEL=gpt-5.4 uv run pytest tests/integration/test_codex_live_smoke.py -q

The test intentionally uses no OE tools and no company data.
"""
from __future__ import annotations

import os

import pytest

from openexecutive.providers.codex_provider import CODEX_MODEL_PREFIX, CodexProvider


@pytest.mark.asyncio
async def test_connected_chatgpt_subscription_completes_a_codex_turn() -> None:
    model = os.getenv("CODEX_LIVE_SMOKE_MODEL")
    if not model:
        pytest.skip("Set CODEX_LIVE_SMOKE_MODEL after connecting ChatGPT to run this smoke test")
    if not model.startswith(CODEX_MODEL_PREFIX):
        model = f"{CODEX_MODEL_PREFIX}{model}"

    message = await CodexProvider().messages_create(
        model=model,
        max_tokens=32,
        system="Reply concisely and do not use tools.",
        messages=[{"role": "user", "content": "Reply exactly: Codex live smoke OK"}],
        timeout=90.0,
    )

    text = "".join(block.text for block in message.content if block.type == "text")
    assert text.strip()
