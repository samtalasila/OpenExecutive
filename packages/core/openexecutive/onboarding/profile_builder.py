from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from openexecutive.memory.company_profile import CompanyProfile
from openexecutive.onboarding.wizard import (
    WizardState,
    build_people_from_answers,
    build_profile_from_answers,
)

logger = logging.getLogger(__name__)


class PrincipalBindingError(RuntimeError):
    """The mandatory principal identity could not be safely persisted."""


def build_and_save_profile(
    state: WizardState,
    profile_path: Path | str | None = None,
    *,
    principal_email: str | None = None,
) -> CompanyProfile:
    if profile_path is None:
        from openexecutive.config import get_settings

        profile_path = get_settings().company_profile_path

    # The principal is the authorization root for the installation. Persist it
    # before replacing the profile file so a reported onboarding completion can
    # never leave an unadministrable instance behind.
    _save_wizard_people(state.answers, principal_email=principal_email)

    profile_path = Path(profile_path)
    raw = build_profile_from_answers(state.answers)
    profile = CompanyProfile.model_validate(raw)
    profile.save_to_yaml(profile_path)
    return profile


def _normalized_email(principal_email: str | None) -> str:
    return (principal_email or "").strip().lower()


def _ensure_principal(
    principal_record: dict[str, Any], *, principal_email: str | None
) -> None:
    """Create or verify the principal record; never fail open."""
    from openexecutive.people import store as people_store
    from openexecutive.people.models import AuthorityScope

    email = _normalized_email(principal_email)
    if not email:
        raise PrincipalBindingError("A verified principal email is required.")

    existing = people_store.find_principal_person()
    if existing is not None:
        existing_email = _normalized_email(existing.email)
        if existing_email != email:
            raise PrincipalBindingError(
                "The configured principal has a different email. "
                "Run 'openexecutive principal bind-email <email>' locally to repair it."
            )
        if existing.id is None:
            raise PrincipalBindingError("The configured principal has no persistent ID.")
        if AuthorityScope.WILDCARD not in existing.authority_scope:
            people_store.set_authority_scope(
                existing.id,
                [*existing.authority_scope, AuthorityScope.WILDCARD],
            )
        return

    record = dict(principal_record)
    raw_scopes: list[str] = record.pop("authority_scope", [])
    record["email"] = email
    try:
        principal_id = people_store.upsert_person(**record)
    except ValueError as exc:
        # Another bootstrap can win the SQLite uniqueness race. It is safe to
        # continue only when it established the same verified identity.
        existing = people_store.find_principal_person()
        if (
            existing is not None
            and existing.id is not None
            and _normalized_email(existing.email) == email
        ):
            principal_id = existing.id
        else:
            raise PrincipalBindingError("Could not establish the principal identity.") from exc
    if principal_id is None:
        raise PrincipalBindingError("Could not establish the principal identity.")

    scopes: list[AuthorityScope] = []
    for token in raw_scopes:
        with contextlib.suppress(ValueError):
            scopes.append(AuthorityScope(token))
    if AuthorityScope.WILDCARD not in scopes:
        scopes.append(AuthorityScope.WILDCARD)
    people_store.set_authority_scope(principal_id, scopes)


def _save_non_principal(record: dict[str, Any]) -> None:
    """Persist optional roster data without blocking a completed onboarding."""
    from openexecutive.people import store as people_store
    from openexecutive.people.models import AuthorityScope

    raw_scopes: list[str] = record.pop("authority_scope", [])
    person_id = people_store.upsert_person(**record)
    scopes: list[AuthorityScope] = []
    for token in raw_scopes:
        with contextlib.suppress(ValueError):
            scopes.append(AuthorityScope(token))
    if scopes:
        people_store.set_authority_scope(person_id, scopes)


def _save_wizard_people(
    answers: dict[str, Any], *, principal_email: str | None = None
) -> None:
    """Persist the principal strictly and optional roster rows best-effort."""
    from openexecutive.people import registry as people_registry
    from openexecutive.people import store as people_store

    records = build_people_from_answers(answers)
    if not records:
        return
    people_store.initialize_db()

    principal_record = next(
        (record for record in records if record.get("is_principal")), None
    )
    if principal_record is not None:
        _ensure_principal(principal_record, principal_email=principal_email)

    for source_record in records:
        if source_record.get("is_principal"):
            continue
        try:
            _save_non_principal(dict(source_record))
        except Exception:
            logger.warning(
                "onboarding skipped optional roster record",
                exc_info=True,
            )
    people_registry.invalidate()


def load_or_create_profile(path: Path | str | None = None) -> CompanyProfile:
    if path is None:
        from openexecutive.config import get_settings

        path = get_settings().company_profile_path

    return CompanyProfile.load_from_yaml(Path(path))
