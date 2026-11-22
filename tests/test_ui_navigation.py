"""Main menu: "Document Ingestion" replaces "Documents" and "Crawl"; its tabs Documents and URLs keep the
former pages' content, each with its own address. The rendering itself is checked in a browser."""

import re

from knowledge_mesh.api.app import STATIC_DIR


def test_document_ingestion_replaces_documents_and_crawl():
    html = (STATIC_DIR / "index.html").read_text()
    nav = re.search(r'<nav id="nav">(.*?)</nav>', html, re.S).group(1)
    assert re.findall(r">([^<]+)</a>", nav) == ["Search", "Document Ingestion", "Catalog", "Definitions", "Jobs"]
    assert 'href="#ingestion"' in nav and "#documents" not in html and "#crawl" not in html
    js = (STATIC_DIR / "app.js").read_text()
    assert "ingestion: renderIngestion" in js
    assert "documents: renderDocuments," not in js and "crawl: renderCrawl" not in js  # old addresses dropped
    assert 'documents: { label: "Documents", render: renderDocuments }' in js
    assert 'urls: { label: "URLs", render: renderCrawl }' in js
    assert 'localStorage.getItem("meshIngestionTab")' in js  # last tab used
    assert '"Crawl a website"' not in js and 'el("h1", {}, "Documents")' not in js  # former titles dropped
    for still in ('"/api/documents", fd, true', '"/api/ingest-directory"', '"/api/crawl"', "Start crawl", "Ingested"):
        assert still in js, still  # the former pages' functionality is unchanged


def test_table_page_has_metadata_and_profile_tabs():
    """#dataset/<id> opens the Dataset Metadata tab, #dataset/<id>/profile the Profile tab; in the columns grid the
    business term and its sensitive flag are on one line, its review status and buttons on the line below."""
    js = (STATIC_DIR / "app.js").read_text()
    assert 'const DATASET_TABS = { metadata: "Dataset Metadata", profile: "Profile" };' in js
    assert '`#dataset/${d.dataset_id}/${key}`' in js and 'if (tab === "profile")' in js
    assert '"Empty or blank", "Data format", "Content type",' in js  # the profile grid
    assert '"Column profiles"' not in js  # moved from a collapsed panel to the Profile tab
    assert 'el("div", { class: "term-review" }, statusTag(c.business_term.status),' in js
    css = (STATIC_DIR / "style.css").read_text()
    assert ".term-review { display: flex;" in css


def test_catalog_pages_use_the_full_window_width():
    """Other pages stay centered at 1200px; on catalog pages the pane right of the facet grows with the window."""
    css = (STATIC_DIR / "style.css").read_text()
    assert "main { max-width: 1200px;" in css and "main:has(> .catalog-layout) { max-width: none; }" in css
    assert "grid-template-columns: minmax(220px, 280px) minmax(0, 1fr)" in css


def test_ui_files_are_revalidated_on_every_load(mesh_factory):
    """Browsers must re-check the page and its scripts, or an old app.js keeps running after an update."""
    from fastapi.testclient import TestClient

    from knowledge_mesh.api.app import create_app

    client = TestClient(create_app(mesh_factory()))
    for path in ("/", "/static/app.js", "/static/style.css"):
        res = client.get(path)
        assert res.status_code == 200 and res.headers["cache-control"] == "no-cache", path
    for path in ("/static/app.js", "/static/style.css"):  # unchanged files: a cheap "not modified" answer
        etag = client.get(path).headers["etag"]
        assert client.get(path, headers={"If-None-Match": etag}).status_code == 304, path
    assert "cache-control" not in client.get("/api/health").headers  # API responses are unchanged
