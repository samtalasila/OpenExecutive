"""Per-turn actor context for tools that need server-derived authority."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ActorContext:
    """Identity and capabilities resolved before an Executive turn begins."""

    person_id: int | None = None
    can_manage_roster: bool = False


current_actor: ContextVar[ActorContext | None] = ContextVar(
    "current_actor", default=None
)


def get_current_actor() -> ActorContext:
    """Return the active actor or a no-capability context outside a turn."""
    return current_actor.get() or ActorContext()
