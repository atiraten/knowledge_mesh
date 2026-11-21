import io

import pytest

from knowledge_mesh.common.errors import UnsupportedMediaType
from knowledge_mesh.unstructured.chunking import HierarchicalChunker
from knowledge_mesh.unstructured.loaders import detect_type, extract

ALLOWED = [".pdf", ".docx", ".txt", ".md", ".html", ".htm"]


def make_pdf(pages: list[str]) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 720
        for line in text.split("\n"):
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def make_docx() -> bytes:
    import docx

    d = docx.Document()
    d.add_heading("Customer Policy", level=1)
    d.add_paragraph("Each customer receives a unique identifier.")
    d.add_heading("Orders", level=2)
    d.add_paragraph("A customer places orders.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Field", "Meaning"
    t.cell(1, 0).text, t.cell(1, 1).text = "customer_id", "Customer Identifier"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_pdf_pages_are_preserved():
    doc = extract(make_pdf(["Customer identification rules.", "Order lifecycle and payments."]), "application/pdf")
    assert doc.page_count == 2
    assert [s.page for s in doc.sections] == [1, 2]
    assert "Order lifecycle" in doc.sections[1].text


def test_docx_headings_and_tables():
    doc = extract(make_docx(), detect_type("p.docx", make_docx(), None, ALLOWED)[1])
    headings = [s.heading for s in doc.sections]
    assert "Customer Policy" in headings and "Orders" in headings
    table = next(s for s in doc.sections if s.heading and s.heading.endswith("(table)"))
    assert "customer_id | Customer Identifier" in table.text
    assert doc.title == "Customer Policy"


def test_html_sections_links_and_boilerplate_removed():
    html = b"""<html><head><title>Help</title><meta name="robots" content="noarchive"></head><body>
    <nav><a href="/a">A</a></nav><h1>Intro</h1><p>Welcome text.</p><script>alert(1)</script>
    <h2>Billing</h2><p>Invoices are issued monthly.</p><table><tr><td>x</td><td>y</td></tr></table></body></html>"""
    doc = extract(html, "text/html")
    assert doc.title == "Help"
    assert "/a" in doc.links
    assert [s.heading for s in doc.sections] == ["Intro", "Billing"]
    assert "alert" not in doc.text and "x | y" in doc.text


def test_markdown_sections():
    doc = extract(b"# Title\nintro\n## Part\nbody", "text/markdown")
    assert [(s.heading, s.text) for s in doc.sections] == [("Title", "intro"), ("Part", "body")]


@pytest.mark.parametrize("name,data", [("x.exe", b"MZ..."), ("x.pdf", b"not a pdf"), ("x.docx", b"plain"),
                                       ("x.txt", b"\x00\x01binary")])
def test_unsupported_or_mismatched_types_rejected(name, data):
    with pytest.raises(UnsupportedMediaType):
        detect_type(name, data, None, ALLOWED)


def test_hierarchy_levels_parent_links_and_stable_ids():
    text = " ".join(f"Sentence {i} explains how customers place orders and pay invoices." for i in range(500))
    doc = extract(text.encode(), "text/plain")
    chunker = HierarchicalChunker([2048, 1024, 256], 20)
    a = chunker.chunk("doc-x", "hash1", doc, {"source_uri": "u", "source_type": "upload"})
    b = chunker.chunk("doc-x", "hash1", doc, {"source_uri": "u", "source_type": "upload"})
    assert {r["level"] for r in a.records} == {0, 1, 2}
    assert [n.node_id for n in a.all_nodes] == [n.node_id for n in b.all_nodes]
    ids = {n.node_id for n in a.all_nodes}
    for leaf in a.leaf_nodes:
        assert leaf.node_id.startswith("chunk-")
        assert leaf.parent_node and leaf.parent_node.node_id in ids
        assert leaf.ref_doc_id == "doc-x"
    c = chunker.chunk("doc-x", "hash2", doc, {"source_uri": "u", "source_type": "upload"})
    assert {n.node_id for n in c.all_nodes}.isdisjoint(ids)
