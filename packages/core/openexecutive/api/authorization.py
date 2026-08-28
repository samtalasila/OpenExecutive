"""Route dependencies for permissions derived from the trusted UI identity."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from openexecutive.people.store import find_person_by_email, find_principal_person


def require_principal(
    x_caller_email: Annotated[
        str | None, Header(alias="X-Caller-Email")
    ] = None,
) -> None:
    """Require a principal identified by the trusted UI proxy.

    The shared-secret middleware is the outer boundary for API traffic. The UI
    proxy strips all client-provided ``x-caller-*`` headers and stamps this
    value from the verified NextAuth session before it reaches FastAPI.
    """
    if not x_caller_email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Principal access required.",
        )
    caller = find_person_by_email(x_caller_email)
    principal = find_principal_person()
    # The database permits only one active principal. Only that record may
    # administer instance-wide resources such as the Codex App Server session.
    if caller is not None and principal is not None and caller.id == principal.id:
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Principal access required.",
    )


PrincipalOnly = Annotated[None, Depends(require_principal)]
