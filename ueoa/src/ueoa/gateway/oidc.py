# pyright: basic
"""Inbound sign-in via OpenID Connect (Microsoft Entra ID in production, any OIDC provider locally).

Validates Bearer JWTs against the provider's JWKS and maps app roles / groups to UEOA roles.
Disabled (no-op) until UEOA_OIDC_ISSUER and UEOA_OIDC_AUDIENCE are set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

ROLE_MAP_DEFAULT = "UEOA.Admin=admin,UEOA.Reviewer=reviewer,UEOA.Viewer=viewer,UEOA.Approver=approver"


class OidcError(Exception):
    """Token rejected (invalid, expired, wrong audience/issuer, or no UEOA role)."""


@dataclass(frozen=True)
class OidcSettings:
    issuer: str
    audience: str
    roles_claim: str
    role_map: dict[str, str]
    groups_map: dict[str, str]
    allow_demo_role: bool
    clock_skew_s: int

    @classmethod
    def from_env(cls) -> OidcSettings:
        def pairs(raw: str) -> dict[str, str]:
            out: dict[str, str] = {}
            for item in raw.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    out[key.strip()] = value.strip()
            return out

        return cls(
            issuer=os.environ.get("UEOA_OIDC_ISSUER", "").strip().rstrip("/"),
            audience=os.environ.get("UEOA_OIDC_AUDIENCE", "").strip(),
            roles_claim=os.environ.get("UEOA_OIDC_ROLES_CLAIM", "roles").strip() or "roles",
            role_map=pairs(os.environ.get("UEOA_OIDC_ROLE_MAP", ROLE_MAP_DEFAULT)),
            groups_map=pairs(os.environ.get("UEOA_OIDC_GROUPS_MAP", "")),
            allow_demo_role=os.environ.get("UEOA_ALLOW_DEMO_ROLE", "true").strip().lower() != "false",
            clock_skew_s=int(os.environ.get("UEOA_OIDC_CLOCK_SKEW_S", "60")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.issuer and self.audience)


@dataclass(frozen=True)
class Principal:
    subject: str
    name: str
    roles: tuple[str, ...]
    source: str  # "oidc" or "demo"


class OidcVerifier:
    """Verifies tokens for one issuer. Keys are fetched from the discovery document and cached."""

    def __init__(self, settings: OidcSettings, jwks_client: Any | None = None):
        self.settings = settings
        self._jwks = jwks_client

    def _jwks_client(self) -> Any:
        if self._jwks is None:
            from jwt import PyJWKClient

            discovery = httpx.get(f"{self.settings.issuer}/.well-known/openid-configuration", timeout=10)
            discovery.raise_for_status()
            self._jwks = PyJWKClient(discovery.json()["jwks_uri"], cache_keys=True)
        return self._jwks

    def principal(self, token: str) -> Principal:
        import jwt

        try:
            key = self._jwks_client().get_signing_key_from_jwt(token).key
            claims: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.settings.audience,
                issuer=self.settings.issuer,
                leeway=self.settings.clock_skew_s,
            )
        except jwt.PyJWTError as exc:
            raise OidcError(f"invalid token: {exc.__class__.__name__}") from exc
        except httpx.HTTPError as exc:
            raise OidcError("identity provider unreachable") from exc

        roles = self._roles(claims)
        if not roles:
            raise OidcError("signed in, but no UEOA role assigned")
        name = claims.get("preferred_username") or claims.get("email") or claims.get("name") or claims.get("sub")
        subject = claims.get("oid") or claims.get("sub") or ""
        return Principal(subject=str(subject), name=str(name), roles=roles, source="oidc")

    def _roles(self, claims: dict[str, Any]) -> tuple[str, ...]:
        mapped: set[str] = set()
        for raw in _dig(claims, self.settings.roles_claim) or []:
            if raw in self.settings.role_map:
                mapped.add(self.settings.role_map[raw])
        if not mapped and self.settings.groups_map:
            for group_id in claims.get("groups") or []:
                if group_id in self.settings.groups_map:
                    mapped.add(self.settings.groups_map[group_id])
        return tuple(sorted(mapped))


def _dig(claims: dict[str, Any], dotted: str) -> Any:
    value: Any = claims
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


@lru_cache(maxsize=1)
def verifier() -> OidcVerifier | None:
    """Process-wide verifier, or None when OIDC is not configured."""
    settings = OidcSettings.from_env()
    return OidcVerifier(settings) if settings.enabled else None


def bearer_token(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None