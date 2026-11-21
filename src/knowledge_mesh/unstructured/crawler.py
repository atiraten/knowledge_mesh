"""Breadth-first website crawler with scope, robots.txt, rate-limit and SSRF controls."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from ..common.errors import BlockedURL, InvalidArgument
from ..observability import get_metrics, span
from .url_policy import (
    Resolver, SafetyGuard, ScopePolicy, has_infinite_pattern, looks_binary, normalize_url,
)

log = logging.getLogger(__name__)

ACCEPTED_TYPES = {
    "text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_LOC = re.compile(rb"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_MAX_ROBOTS_BYTES = 512 * 1024
_MAX_SITEMAP_BYTES = 5 * 1024 * 1024


@dataclass
class PageOutcome:
    url: str
    status: str  # indexed | skipped | failed
    reason: str | None = None
    depth: int = 1  # 1 = start page
    document_id: str | None = None
    final_url: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


# (final_url, body, content_type, depth) -> {"status", "reason", "document_id", ...}
PageHandler = Callable[[str, bytes, str, int], dict[str, Any]]


class _Skip(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _RateLimiter:
    def __init__(self) -> None:
        self._next: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, host: str, interval: float) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            at = max(now, self._next.get(host, 0.0))
            self._next[host] = at + interval
        if at > now:
            await asyncio.sleep(at - now)


class WebCrawler:
    def __init__(self, settings, transport: httpx.AsyncBaseTransport | None = None,
                 resolver: Resolver | None = None):
        self.cfg = settings
        self.transport = transport
        self.guard = SafetyGuard(settings.blocked_hosts, settings.allow_private_networks, resolver)

    def normalize(self, url: str, base: str | None = None) -> str | None:
        return normalize_url(url, base, self.cfg.strip_query_params)

    async def validate_start(self, url: str) -> str:
        normalized = self.normalize(url)
        if not normalized:
            raise InvalidArgument("a valid http(s) URL is required", {"url": url})
        try:
            await self.guard.check(normalized)
        except BlockedURL:
            get_metrics().blocked_urls.add(1, {"reason": "start_url"})
            raise
        return normalized

    async def crawl(self, start_url: str, handler: PageHandler, on_item: Callable[[PageOutcome], None], *,
                    max_depth: int, max_pages: int) -> dict[str, Any]:
        start = await self.validate_start(start_url)
        scope = ScopePolicy(start, self.cfg.allowed_hosts)
        metrics = get_metrics()
        counts: dict[str, int] = defaultdict(int)
        skipped_recorded = 0
        limiter = _RateLimiter()
        robots: dict[str, RobotFileParser | None] = {}
        sem = asyncio.Semaphore(self.cfg.concurrency_per_host)
        base_interval = 1.0 / self.cfg.requests_per_second_per_host

        def record(outcome: PageOutcome) -> None:
            nonlocal skipped_recorded
            counts[outcome.status] += 1
            metrics.crawler_pages.add(1, {"status": outcome.status})
            if outcome.status == "skipped" and outcome.reason in {"out_of_scope", "already_seen"}:
                if skipped_recorded >= self.cfg.max_manifest_skipped:
                    counts["skipped_unrecorded"] += 1
                    return
                skipped_recorded += 1
            on_item(outcome)

        headers = {"User-Agent": self.cfg.user_agent, "Accept": ", ".join(sorted(ACCEPTED_TYPES)) + ";q=0.9,*/*;q=0.1"}
        async with httpx.AsyncClient(transport=self.transport, timeout=self.cfg.request_timeout_seconds,
                                     follow_redirects=False, headers=headers) as client:

            def interval_for(host: str) -> float:
                rp = robots.get(host)
                delay = rp.crawl_delay(self.cfg.user_agent) if rp is not None else None
                return max(base_interval, float(delay or 0))

            async def fetch(url: str, max_bytes: int) -> tuple[str, int, str, bytes]:
                current = url
                for _ in range(self.cfg.max_redirects + 1):
                    try:
                        await self.guard.check(current)
                    except BlockedURL as exc:
                        metrics.blocked_urls.add(1, {"reason": exc.details.get("reason", "blocked")})
                        raise
                    host = urlsplit(current).netloc
                    await limiter.wait(host, interval_for(host))
                    async with client.stream("GET", current) as resp:
                        if resp.status_code in (301, 302, 303, 307, 308):
                            location = resp.headers.get("location")
                            nxt = self.normalize(location, base=current) if location else None
                            if not nxt:
                                raise _Skip("redirect_invalid")
                            if not scope.in_scope(nxt):
                                metrics.blocked_urls.add(1, {"reason": "redirect_out_of_scope"})
                                raise _Skip("redirect_out_of_scope")
                            current = nxt
                            continue
                        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                        if resp.status_code >= 400:
                            return current, resp.status_code, ctype, b""
                        declared = resp.headers.get("content-length")
                        if declared and declared.isdigit() and int(declared) > max_bytes:
                            raise _Skip("response_too_large")
                        buf = bytearray()
                        async for chunk in resp.aiter_bytes():
                            buf.extend(chunk)
                            if len(buf) > max_bytes:
                                raise _Skip("response_too_large")
                        return current, resp.status_code, ctype, bytes(buf)
                raise _Skip("too_many_redirects")

            async def load_robots(url: str) -> RobotFileParser | None:
                parts = urlsplit(url)
                host = parts.netloc
                if host in robots:
                    return robots[host]
                if not self.cfg.respect_robots_txt:
                    robots[host] = None
                    return None
                rp = RobotFileParser()
                try:
                    _, status, _, body = await fetch(f"{parts.scheme}://{host}/robots.txt", _MAX_ROBOTS_BYTES)
                    if status < 400:
                        rp.parse(body.decode("utf-8", errors="ignore").splitlines())
                    elif status < 500:
                        rp.allow_all = True  # RFC 9309: unavailable robots.txt -> no restrictions
                    else:
                        rp.disallow_all = True  # RFC 9309: unreachable -> assume complete disallow
                except (_Skip, BlockedURL, httpx.HTTPError):
                    rp.disallow_all = True
                robots[host] = rp
                return rp

            async def sitemap_urls(url: str, rp: RobotFileParser | None) -> list[str]:
                parts = urlsplit(url)
                candidates = list((rp.site_maps() or []) if rp else []) or [f"{parts.scheme}://{parts.netloc}/sitemap.xml"]
                found: list[str] = []
                for sm in candidates[:5]:
                    sm_url = self.normalize(sm)
                    if not sm_url or not scope.in_scope(sm_url):
                        continue
                    try:
                        _, status, _, body = await fetch(sm_url, _MAX_SITEMAP_BYTES)
                    except (_Skip, BlockedURL, httpx.HTTPError):
                        continue
                    if status >= 400:
                        continue
                    locs = [m.decode("utf-8", "ignore") for m in _LOC.findall(body)]
                    if b"<sitemapindex" in body[:2000].lower():
                        for nested in locs[:5]:
                            nested_url = self.normalize(nested)
                            if not nested_url or not scope.in_scope(nested_url):
                                continue
                            try:
                                _, st, _, nb = await fetch(nested_url, _MAX_SITEMAP_BYTES)
                            except (_Skip, BlockedURL, httpx.HTTPError):
                                continue
                            if st < 400:
                                found += [m.decode("utf-8", "ignore") for m in _LOC.findall(nb)]
                    else:
                        found += locs
                    if len(found) > max_pages * 4:
                        break
                return found

            async def process(url: str, depth: int) -> tuple[PageOutcome, list[str]]:
                async with sem:
                    try:
                        final_url, status, ctype, body = await fetch(url, self.cfg.max_response_bytes)
                    except BlockedURL as exc:
                        return PageOutcome(url, "skipped", f"blocked:{exc.details.get('reason', 'blocked')}", depth), []
                    except _Skip as exc:
                        return PageOutcome(url, "skipped", exc.reason, depth), []
                    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                        return PageOutcome(url, "failed", f"fetch_error:{type(exc).__name__}", depth), []
                if status >= 400:
                    return PageOutcome(url, "failed", f"http_{status}", depth, final_url=final_url), []
                if not ctype:
                    ctype = "text/html" if not looks_binary(final_url) else "application/octet-stream"
                if ctype not in ACCEPTED_TYPES:
                    return PageOutcome(url, "skipped", f"unsupported_content_type:{ctype}", depth,
                                       final_url=final_url), []
                links: list[str] = []
                noindex = nofollow = False
                if ctype in {"text/html", "application/xhtml+xml"}:
                    links, noindex, nofollow = _html_links(body)
                if nofollow:
                    links = []
                if noindex:
                    return PageOutcome(url, "skipped", "meta_noindex", depth, final_url=final_url), links
                try:
                    result = await asyncio.to_thread(handler, final_url, body, ctype, depth)
                except Exception as exc:  # one bad page never stops the crawl
                    log.warning("page handling failed", extra={"url": final_url, "error": type(exc).__name__})
                    return PageOutcome(url, "failed", f"processing_error:{type(exc).__name__}", depth,
                                       final_url=final_url), links
                return PageOutcome(url, result["status"], result.get("reason"), depth, result.get("document_id"),
                                   final_url, {k: v for k, v in result.items()
                                               if k not in {"status", "reason", "document_id"}}), links

            # Depth is 1-based: level 1 is the start page. max_depth == 1 is a user-requested fetch of that
            # one page, not a crawl: no robots.txt, sitemap or link discovery (documented FR-03.5 exception).
            single_page = max_depth == 1
            with span("crawler.crawl", "CHAIN", {"crawl.start_url": start, "crawl.max_depth": max_depth,
                                                  "crawl.max_pages": max_pages,
                                                  "crawl.single_page": single_page}) as sp:
                rp = None if single_page else await load_robots(start)
                seen: set[str] = {start}
                query_variants: dict[str, int] = defaultdict(int)
                sitemap_seed: list[str] = []
                if self.cfg.use_sitemap and max_depth >= 2:
                    for loc in await sitemap_urls(start, rp):
                        n = self.normalize(loc)
                        if n and n not in seen and scope.in_scope(n) and not looks_binary(n):
                            seen.add(n)
                            sitemap_seed.append(n)
                frontier = [start]
                depth = 1
                fetched = 0
                while frontier and depth <= max_depth and fetched < max_pages:
                    batch: list[str] = []
                    for url in frontier:
                        if fetched + len(batch) >= max_pages:
                            break
                        if not single_page:
                            robots_rules = await load_robots(url)
                            if robots_rules is not None and not robots_rules.can_fetch(self.cfg.user_agent, url):
                                record(PageOutcome(url, "skipped", "robots_disallowed", depth))
                                continue
                        batch.append(url)
                    fetched += len(batch)
                    results = await asyncio.gather(*(process(u, depth) for u in batch))
                    next_frontier: list[str] = []
                    if depth == 1:
                        next_frontier.extend(sitemap_seed)
                    for outcome, links in results:
                        record(outcome)
                        if depth >= max_depth:
                            continue
                        base = outcome.final_url or outcome.url
                        for link in links:
                            n = self.normalize(link, base=base)
                            if not n or (self.cfg.deduplicate_urls and n in seen):
                                continue
                            seen.add(n)
                            if not scope.in_scope(n):
                                metrics.blocked_urls.add(1, {"reason": "out_of_scope"})
                                record(PageOutcome(n, "skipped", "out_of_scope", depth + 1))
                                continue
                            if looks_binary(n):
                                counts["ignored_non_content"] += 1
                                continue
                            if len(n) > self.cfg.max_url_length or has_infinite_pattern(n):
                                record(PageOutcome(n, "skipped", "infinite_url_pattern", depth + 1))
                                continue
                            if "?" in n:
                                key = n.split("?", 1)[0]
                                query_variants[key] += 1
                                if query_variants[key] > self.cfg.max_query_variants_per_path:
                                    record(PageOutcome(n, "skipped", "query_variant_limit", depth + 1))
                                    continue
                            next_frontier.append(n)
                    frontier = next_frontier
                    depth += 1
                sp.set_attribute("crawl.pages_fetched", fetched)
                for k, v in counts.items():
                    sp.set_attribute(f"crawl.count.{k}", v)
        summary = {"start_url": start, "pages_fetched": fetched, "max_depth_reached": max(1, depth - 1),
                   "counts": dict(counts)}
        if single_page:
            summary["robots_txt"] = "not_checked_single_page"
        return summary


def _html_links(body: bytes) -> tuple[list[str], bool, bool]:
    from bs4 import BeautifulSoup, SoupStrainer

    soup = BeautifulSoup(body, "html.parser", parse_only=SoupStrainer(["a", "meta", "link"]))
    noindex = nofollow = False
    for meta in soup.find_all("meta"):
        if str(meta.get("name", "")).lower() == "robots":
            content = str(meta.get("content", "")).lower()
            noindex = noindex or "noindex" in content or "none" in content
            nofollow = nofollow or "nofollow" in content or "none" in content
    links = []
    for a in soup.find_all("a", href=True):
        rel = " ".join(a.get("rel") or []).lower()
        if "nofollow" in rel:
            continue
        links.append(a["href"])
    return links, noindex, nofollow
