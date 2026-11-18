"""Outbound URL guard (SSRF): http/https only, no embedded credentials, host on an explicit allow-list.

Allow-list entries are exact host names or wildcard patterns such as `*.directline.botframework.com`.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from urllib.parse import urlsplit

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class UrlNotAllowedError(ValueError):
    pass


def url_host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def is_loopback(url: str) -> bool:
    return url_host(url) in LOOPBACK_HOSTS


def check_credentialed_url(url: str, allowed_hosts: Iterable[str]) -> str:
    """Stricter guard for URLs that receive platform credentials: loopback or an exact (non-wildcard) allowed host."""
    check_url(url, allowed_hosts)
    host = url_host(url)
    exact = {pattern.lower() for pattern in allowed_hosts if not any(c in pattern for c in "*?[")}
    if host not in LOOPBACK_HOSTS and host not in exact:
        raise UrlNotAllowedError(f"host '{host}' needs an exact (non-wildcard) allow-list entry for Entra auth")
    return url


def check_url(url: str, allowed_hosts: Iterable[str]) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise UrlNotAllowedError(f"scheme '{parts.scheme}' not allowed")
    if parts.username or parts.password:
        raise UrlNotAllowedError("credentials in URL not allowed")
    host = (parts.hostname or "").lower()
    if not host or not any(fnmatch.fnmatchcase(host, pattern) for pattern in allowed_hosts if pattern != "*"):
        raise UrlNotAllowedError(f"host '{host}' is not on the registration allow-list")
    return url
