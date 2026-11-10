# pyright: basic
"""OIDC token validation and role mapping (offline: local RSA key, fake JWKS client)."""

from __future__ import annotations

import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ueoa.gateway.oidc import OidcError, OidcSettings, OidcVerifier, bearer_token

ISSUER = "[https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/v2.0"](https://login.microsoftonline.com/11111111-2222-3333-4444-555555555555/v2.0")
AUDIENCE = "api://ueoa-api-poc"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PEM = _KEY.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
)


class _FakeSigningKey:
    key = _KEY.public_key()


class _FakeJwks:
    def get_signing_key_from_jwt(self, token: str) -> _FakeSigningKey:
        return _FakeSigningKey()


def settings(**overrides: Any) -> OidcSettings:
    base = OidcSettings(
        issuer=ISSUER,
        audience=AUDIENCE,
        roles_claim="roles",
        role_map={"UEOA.Admin": "admin", "UEOA.Reviewer": "reviewer"},
        groups_map={},
        allow_demo_role=True,
        clock_skew_s=60,
    )
    return OidcSettings(**{**base.__dict__, **overrides})


def token(**claims: Any) -> str:
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 600,
        "sub": "sub-1",
        "oid": "oid-1",
        "preferred_username": "rohan@example.com",
        **claims,
    }
    return jwt.encode(payload, _PEM, algorithm="RS256", headers={"kid": "test"})


def test_valid_token_maps_app_role() -> None:
    principal = OidcVerifier(settings(), _FakeJwks()).principal(token(roles=["UEOA.Reviewer"]))
    assert principal.roles == ("reviewer",)
    assert principal.name == "rohan@example.com"
    assert principal.subject == "oid-1"
    assert principal.source == "oidc"


def test_wrong_audience_rejected() -> None:
    with pytest.raises(OidcError):
        OidcVerifier(settings(), _FakeJwks()).principal(token(aud="api://other", roles=["UEOA.Admin"]))


def test_expired_token_rejected() -> None:
    with pytest.raises(OidcError):
        OidcVerifier(settings(), _FakeJwks()).principal(token(exp=int(time.time()) - 3600, roles=["UEOA.Admin"]))


def test_no_mapped_role_rejected() -> None:
    with pytest.raises(OidcError, match="no UEOA role"):
        OidcVerifier(settings(), _FakeJwks()).principal(token(roles=["Something.Else"]))


def test_groups_fallback() -> None:
    verifier = OidcVerifier(settings(groups_map={"grp-1": "admin"}), _FakeJwks())
    assert verifier.principal(token(groups=["grp-1"])).roles == ("admin",)


def test_disabled_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UEOA_OIDC_ISSUER", raising=False)
    monkeypatch.delenv("UEOA_OIDC_AUDIENCE", raising=False)
    assert OidcSettings.from_env().enabled is False


def test_bearer_token_parsing() -> None:
    assert bearer_token("Bearer abc") == "abc"
    assert bearer_token("bearer abc") == "abc"
    assert bearer_token("Basic abc") is None
    assert bearer_token(None) is None