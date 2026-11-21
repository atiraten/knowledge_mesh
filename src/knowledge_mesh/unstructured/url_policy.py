"""URL normalization, crawl scope and SSRF protection.

Canonicalization policy (FR-03.6):
  * scheme and host are lower-cased; a trailing dot on the host is removed;
  * default ports (80 for http, 443 for https) are dropped; userinfo is rejected;
  * the fragment is removed; an empty path becomes ``/``; dot segments are resolved;
  * tracking/session query parameters (``crawler.strip_query_params``, ``*`` wildcards) are
    removed and remaining parameters are sorted, so parameter order variants collapse;
  * percent-encoding and path case are preserved (servers may treat them as distinct).

Safety policy (FR-03.10): only http/https; hosts must resolve exclusively to public
(global) addresses — loopback, private, link-local (incl. 169.254.169.254 metadata),
multicast, reserved and unspecified addresses are blocked, as are well-known metadata
hostnames. Every redirect target is re-validated. DNS answers are checked right before each
request; residual DNS-rebinding risk between check and connect is documented in the README.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import posixpath
import socket
from typing import Awaitable, Callable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from ..common.errors import BlockedURL, InvalidArgument

Resolver = Callable[[str, int], Awaitable[list[str]]]

_NON_CONTENT_SCHEMES = {"mailto", "javascript", "tel", "data", "ftp", "file", "sms", "about", "blob"}
_BINARY_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tif", ".tiff",
    ".css", ".js", ".mjs", ".map", ".json", ".xml", ".rss", ".atom",
    ".zip", ".gz", ".tgz", ".tar", ".rar", ".7z", ".exe", ".dmg", ".msi", ".bin", ".iso",
    ".mp3", ".mp4", ".m4a", ".wav", ".avi", ".mov", ".mkv", ".webm", ".woff", ".woff2", ".ttf", ".eot",
    ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".doc",
}


def _strip_param(name: str, patterns: list[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, p.lower()) for p in patterns)


def normalize_url(url: str, base: str | None = None, strip_params: list[str] | None = None) -> str | None:
    """Return the canonical form of ``url`` (resolved against ``base``) or None if not crawlable."""
    url = (url or "").strip()
    if not url:
        return None
    scheme_part = url.split(":", 1)[0].lower() if ":" in url else ""
    if scheme_part in _NON_CONTENT_SCHEMES:
        return None
    if base:
        from urllib.parse import urljoin

        url = urljoin(base, url)
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    host = parts.hostname.lower().rstrip(".")
    try:
        port = parts.port
    except ValueError:
        return None
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    path = parts.path or "/"
    trailing = path.endswith("/")
    path = posixpath.normpath(path) if path not in {"", "/"} else "/"
    if path == ".":
        path = "/"
    if trailing and not path.endswith("/"):
        path += "/"
    path = quote(path, safe="/%:@!$&'()*+,;=-._~")
    query_items = parse_qsl(parts.query, keep_blank_values=True)
    if strip_params:
        query_items = [(k, v) for k, v in query_items if not _strip_param(k, strip_params)]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def looks_binary(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return posixpath.splitext(path)[1] in _BINARY_EXTENSIONS


def has_infinite_pattern(url: str, max_repeats: int = 2) -> bool:
    segments = [s for s in urlsplit(url).path.split("/") if s]
    if len(segments) > 20:
        return True
    counts: dict[str, int] = {}
    for s in segments:
        counts[s] = counts.get(s, 0) + 1
        if counts[s] > max_repeats:
            return True
    return False


class ScopePolicy:
    """Exact starting hostname by default; other hosts/subdomains only through the allowlist."""

    def __init__(self, start_url: str, allowed_hosts: list[str]):
        host = urlsplit(start_url).hostname
        if not host:
            raise InvalidArgument("start URL has no hostname")
        self.start_host = host.lower().rstrip(".")
        self.allowed = [h.lower().rstrip(".") for h in allowed_hosts]

    def in_scope(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower().rstrip(".")
        if host == self.start_host:
            return True
        for pattern in self.allowed:
            if pattern.startswith("*."):
                if host.endswith(pattern[1:]) and host != pattern[2:]:
                    return True
            elif host == pattern:
                return True
        return False


def is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return bool(addr.is_global) and not (
        addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
        or addr.is_reserved or addr.is_unspecified
    )


async def system_resolver(host: str, port: int) -> list[str]:
    import asyncio

    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


class SafetyGuard:
    def __init__(self, blocked_hosts: list[str], allow_private: bool = False, resolver: Resolver | None = None):
        self.blocked_hosts = {h.lower() for h in blocked_hosts}
        self.allow_private = allow_private
        self.resolver = resolver or system_resolver

    async def check(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            raise BlockedURL("only http and https URLs are allowed", {"url": url})
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            raise BlockedURL("URL has no host", {"url": url})
        if host in self.blocked_hosts or host.endswith(".internal") or host.endswith(".localhost"):
            raise BlockedURL("host is blocked", {"url": url, "reason": "blocked_host"})
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            literal = ipaddress.ip_address(host.strip("[]"))
            addresses = [str(literal)]
        except ValueError:
            try:
                addresses = await self.resolver(host, port)
            except OSError as exc:
                raise BlockedURL("host could not be resolved", {"url": url, "reason": "dns_failure"}) from exc
        if not addresses:
            raise BlockedURL("host could not be resolved", {"url": url, "reason": "dns_failure"})
        if not self.allow_private:
            bad = [a for a in addresses if not is_public_ip(a)]
            if bad:
                raise BlockedURL(
                    "destination resolves to a private, loopback, link-local or reserved address",
                    {"url": url, "reason": "private_network"},
                )
