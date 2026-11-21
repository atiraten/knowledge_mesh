import asyncio

import httpx
import pytest

from knowledge_mesh.common.errors import BlockedURL
from knowledge_mesh.unstructured.url_policy import SafetyGuard, ScopePolicy, is_public_ip, normalize_url

PUBLIC = {"site.test": ["93.184.216.34"], "cdn.site.test": ["93.184.216.40"], "other.test": ["93.184.216.50"],
          "evil.test": ["10.0.0.7"], "rebind.site.test": ["127.0.0.1"]}


async def resolver(host, port):
    if host not in PUBLIC:
        raise OSError("NXDOMAIN")
    return PUBLIC[host]


SECTION0 = ("<html><body><h1>Section 0</h1><p>Section 0 describes customer accounts.</p>"
            + "".join(f'<a href="/s0/a{j}?utm_source=x">a{j}</a>' for j in range(6)) + "</body></html>")


def site_handler(request: httpx.Request) -> httpx.Response:
    """index -> 10 sections -> 6 articles each (71 pages within depth 3), plus traps."""
    host, path = request.url.host, request.url.path
    if host != "site.test":
        return httpx.Response(200, html="<p>other host</p>")
    if path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
    if path == "/":
        links = "".join(f'<a href="/s{i}/">s{i}</a>' for i in range(10))
        links += ('<a href="mailto:x@site.test">m</a><a href="javascript:void(0)">j</a>'
                  '<a href="https://other.test/page">off-scope</a><a href="/private/secret">p</a>'
                  '<a href="/s0/#frag">dup-fragment</a><a href="/logo.png">img</a>'
                  '<a href="/redirect-out">r</a><a href="/copy">copy</a>')
        return httpx.Response(200, html=f"<html><title>Home</title><body><h1>Home</h1><p>Customers and orders "
                                        f"overview.</p>{links}</body></html>")
    if path == "/redirect-out":
        return httpx.Response(302, headers={"location": "https://other.test/landing"})
    if path == "/copy":  # byte-identical to /s0/ under a different URL
        return httpx.Response(200, html=SECTION0)
    if path.startswith("/s") and path.count("/") == 2 and path.endswith("/"):
        i = int(path[2:-1])
        links = "".join(f'<a href="/s{i}/a{j}?utm_source=x">a{j}</a>' for j in range(6))
        if i == 0:
            return httpx.Response(200, html=SECTION0)
        return httpx.Response(200, html=f"<html><body><h1>Section {i}</h1><p>Section {i} explains payments "
                                        f"and invoices number {i}.</p>{links}</body></html>")
    if path.startswith("/s"):
        return httpx.Response(200, html=f"<html><body><h2>Article {path}</h2><p>Article at {path} about "
                                        f"suppliers and shipments.</p><a href='/'>home</a></body></html>")
    return httpx.Response(404, text="missing")


@pytest.fixture
def crawl_mesh(mesh_factory, config_factory):
    def make(overrides=None):
        return mesh_factory(config_factory(overrides), http_transport=httpx.MockTransport(site_handler),
                            resolver=resolver)
    return make


def test_url_normalization():
    n = lambda u, b=None: normalize_url(u, b, ["utm_*", "sessionid"])  # noqa: E731
    assert n("HTTP://Site.Test:80/a/./b/../c?b=2&a=1&utm_source=x#frag") == "http://site.test/a/c?a=1&b=2"
    assert n("https://site.test") == "https://site.test/"
    assert n("mailto:a@b.c") is None and n("javascript:alert(1)") is None
    assert n("/x y", "https://site.test/dir/") == "https://site.test/x%20y"
    assert n("https://user:pw@site.test/") is None
    assert n("https://site.test/a%2Fb") == "https://site.test/a%2Fb"


def test_scope_policy_exact_host_and_allowlist():
    scope = ScopePolicy("https://site.test/", [])
    assert scope.in_scope("https://site.test/x")
    assert not scope.in_scope("https://cdn.site.test/x") and not scope.in_scope("https://other.test/")
    assert ScopePolicy("https://site.test/", ["*.site.test"]).in_scope("https://cdn.site.test/x")


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data",
                                 "http://[::1]/", "http://10.1.2.3/", "http://evil.test/", "http://metadata.google.internal/",
                                 "ftp://site.test/", "http://nonexistent.test/"])
def test_blocked_destinations(url):
    with pytest.raises(BlockedURL):
        asyncio.run(SafetyGuard(["metadata.google.internal", "localhost"], resolver=resolver).check(url))


def test_public_ip_classification():
    assert is_public_ip("93.184.216.34")
    assert not any(is_public_ip(ip) for ip in ["127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1",
                                               "169.254.169.254", "::1", "fc00::1", "0.0.0.0", "::ffff:127.0.0.1"])


def test_crawl_rejects_private_start_url_before_queuing(crawl_mesh):
    mesh = crawl_mesh()
    with pytest.raises(BlockedURL):
        mesh.crawl_website("http://evil.test/")
    assert mesh.job_store.list() == []


def test_crawl_50_plus_pages_with_limits_and_safety(crawl_mesh):
    mesh = crawl_mesh()
    job = mesh.crawl_website("https://site.test/", max_depth=3, max_pages=60, wait_seconds=120)
    assert job["status"] in {"succeeded", "partial"}
    items = mesh.job_items(job["job_id"], limit=500)["items"]
    fetched = [i for i in items if i["reason"] not in {"out_of_scope", "robots_disallowed"}]
    indexed = [i for i in items if i["status"] == "indexed"]
    assert len(fetched) <= 60 and len(indexed) >= 50
    assert job["result"]["pages_fetched"] == 60
    assert max(i["detail"].get("depth", 0) for i in items if i["status"] == "indexed") <= 3
    by_reason = {i["reason"] for i in items}
    assert {"out_of_scope", "robots_disallowed", "redirect_out_of_scope", "duplicate_content"} <= by_reason
    off = [i for i in items if i["reason"] == "out_of_scope"]
    assert off[0]["ref"] == "https://other.test/page"
    urls = [i["ref"] for i in items]
    assert not any("utm_source" in u or "#" in u or u.startswith("mailto") for u in urls)
    assert len(urls) == len(set(urls))
    assert "manifest_ref" in job and len(str(job)) < 5000  # bounded summary, no page content


def test_depth_and_page_limits(crawl_mesh):
    mesh = crawl_mesh()
    job = mesh.crawl_website("https://site.test/", max_depth=2, max_pages=100, wait_seconds=60)
    items = mesh.job_items(job["job_id"], limit=500)["items"]
    assert all(i["detail"].get("depth", 0) <= 2 for i in items if i["status"] == "indexed")
    assert not any("/a" in i["ref"] for i in items if i["status"] == "indexed")
    assert any(i["reason"] == "robots_disallowed" for i in items)  # depth >= 2 still obeys robots.txt
    job2 = mesh.crawl_website("https://site.test/", max_depth=4, max_pages=5, wait_seconds=60)
    assert job2["result"]["pages_fetched"] == 5


def test_depth_one_fetches_only_the_given_page(mesh_factory, config_factory):
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/private/page":  # disallowed by robots.txt, linking onwards
            return httpx.Response(200, html="<html><body><h1>Private</h1><p>Refund policy for orders.</p>"
                                            "<a href='/s1/'>s1</a></body></html>")
        return site_handler(request)

    mesh = mesh_factory(config_factory(), http_transport=httpx.MockTransport(handler), resolver=resolver)
    job = mesh.crawl_website("https://site.test/private/page", max_depth=1, max_pages=100, wait_seconds=60)
    items = mesh.job_items(job["job_id"], limit=500)["items"]
    assert requested == ["/private/page"]  # no robots.txt, no sitemap, no linked pages
    assert [(i["ref"], i["status"]) for i in items] == [("https://site.test/private/page", "indexed")]
    assert items[0]["detail"]["depth"] == 1
    assert job["result"]["pages_fetched"] == 1
    assert job["result"]["robots_txt"] == "not_checked_single_page"
    requested.clear()
    job = mesh.crawl_website("https://site.test/", max_depth=1, max_pages=100, wait_seconds=60)
    assert requested == ["/"] and job["result"]["max_depth_reached"] == 1
    assert [i["status"] for i in mesh.job_items(job["job_id"], limit=500)["items"]] == ["indexed"]


def test_recrawl_unchanged_produces_no_duplicate_chunks(crawl_mesh):
    mesh = crawl_mesh()
    mesh.crawl_website("https://site.test/", max_depth=3, max_pages=40, wait_seconds=120)
    before = mesh.doc_index.stats()["embedded_nodes"]
    job = mesh.crawl_website("https://site.test/", max_depth=3, max_pages=40, wait_seconds=120)
    after = mesh.doc_index.stats()["embedded_nodes"]
    assert before == after
    items = mesh.job_items(job["job_id"], limit=500)["items"]
    assert not [i for i in items if i["status"] == "indexed"]
    assert any(i["reason"] == "unchanged" for i in items)
