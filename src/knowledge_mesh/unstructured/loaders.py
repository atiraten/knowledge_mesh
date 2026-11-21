"""Text extraction for PDF, DOCX, TXT/Markdown and HTML.

Extraction preserves page numbers (PDF), section headings (DOCX, Markdown, HTML) and table
boundaries (DOCX/HTML tables become their own sections) so chunks can cite a page or section.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common.errors import ExtractionFailed, UnsupportedMediaType

MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}
EXT_BY_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
}


@dataclass
class Section:
    text: str
    page: int | None = None
    heading: str | None = None


@dataclass
class ExtractedDocument:
    mime_type: str
    title: str | None
    sections: list[Section]
    page_count: int | None = None
    links: list[str] = field(default_factory=list)
    meta_robots: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)

    def to_json(self) -> str:
        data = asdict(self)
        data.pop("links", None)  # links are only needed while crawling
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "ExtractedDocument":
        data = json.loads(raw)
        data["sections"] = [Section(**s) for s in data.get("sections", [])]
        return cls(**data)


def detect_type(filename: str, data: bytes, declared_mime: str | None, allowed_exts: list[str]) -> tuple[str, str]:
    """Return (extension, mime). Rejects unsupported types and extension/content mismatches."""
    ext = Path(filename).suffix.lower()
    mime_hint = (declared_mime or "").split(";")[0].strip().lower()
    if ext not in MIME_BY_EXT and mime_hint in EXT_BY_MIME:
        ext = EXT_BY_MIME[mime_hint]
    if ext not in MIME_BY_EXT or ext not in allowed_exts:
        raise UnsupportedMediaType(
            f"unsupported file type: {ext or mime_hint or 'unknown'}",
            {"filename": filename, "allowed_extensions": allowed_exts},
        )
    head = data[:8]
    if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
        raise UnsupportedMediaType("file content is not a PDF", {"filename": filename})
    if ext == ".docx" and not head.startswith(b"PK\x03\x04"):
        raise UnsupportedMediaType("file content is not a DOCX (zip) document", {"filename": filename})
    if ext in {".txt", ".md", ".html", ".htm"} and b"\x00" in data[:4096]:
        raise UnsupportedMediaType("text file contains binary content", {"filename": filename})
    return ext, MIME_BY_EXT[ext]


def extract(data: bytes, mime_type: str, filename: str = "") -> ExtractedDocument:
    try:
        if mime_type == "application/pdf":
            doc = _extract_pdf(data)
        elif mime_type.endswith("wordprocessingml.document"):
            doc = _extract_docx(data)
        elif mime_type in {"text/html", "application/xhtml+xml"}:
            doc = extract_html(data)
        elif mime_type in {"text/plain", "text/markdown"}:
            doc = _extract_text(data, mime_type)
        else:
            raise UnsupportedMediaType(f"unsupported mime type: {mime_type}")
    except (UnsupportedMediaType, ExtractionFailed):
        raise
    except Exception as exc:  # parser-specific errors
        raise ExtractionFailed(f"could not parse {filename or mime_type}: {type(exc).__name__}: {exc}"[:400]) from exc
    doc.sections = [s for s in (Section(_clean(s.text), s.page, s.heading) for s in doc.sections) if s.text]
    if not doc.sections:
        raise ExtractionFailed(
            "no extractable text (scanned PDFs need OCR, which is out of scope for this POC)",
            {"filename": filename},
        )
    if not doc.title:
        doc.title = Path(filename).stem if filename else None
    return doc


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _extract_pdf(data: bytes) -> ExtractedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ExtractionFailed("encrypted PDF cannot be read") from exc
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(text=text, page=i, heading=None))
    title = None
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None
    except Exception:
        title = None
    return ExtractedDocument("application/pdf", title, sections, page_count=len(reader.pages))


def _extract_docx(data: bytes) -> ExtractedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(data))
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    title = (document.core_properties.title or "").strip() or None

    def flush() -> None:
        if buf:
            sections.append(Section("\n".join(buf), None, heading))
            buf.clear()

    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            style = (para.style.name if para.style is not None else "") or ""
            if not text:
                continue
            if style.startswith("Heading") or style == "Title":
                flush()
                heading = text
                if title is None and style in {"Title", "Heading 1"}:
                    title = text
            else:
                buf.append(text)
        elif tag == "tbl":
            flush()
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                sections.append(Section("\n".join(rows), None, f"{heading} (table)" if heading else "Table"))
    flush()
    return ExtractedDocument(MIME_BY_EXT[".docx"], title, sections)


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _extract_text(data: bytes, mime_type: str) -> ExtractedDocument:
    text = _decode(data)
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []
    title: str | None = None
    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            if buf and "".join(buf).strip():
                sections.append(Section("\n".join(buf), None, heading))
            buf = []
            heading = m.group(2).strip()
            title = title or heading
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append(Section("\n".join(buf), None, heading))
    if title is None:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        title = first[:120] or None
    return ExtractedDocument(mime_type, title, sections)


_HEADING_MARK = "\u0000H\u0000"
_DROP_TAGS = ["script", "style", "noscript", "template", "svg", "iframe", "object", "embed", "form",
              "nav", "footer", "header"]


def extract_html(data: bytes) -> ExtractedDocument:
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(data, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    meta_robots: list[str] = []
    for meta in soup.find_all("meta", attrs={"name": re.compile("^robots$", re.I)}):
        meta_robots += [t.strip().lower() for t in str(meta.get("content", "")).split(",") if t.strip()]
    title = soup.title.get_text(" ", strip=True) if soup.title else None
    body = soup.body or soup
    for tag in body.find_all(_DROP_TAGS):
        tag.decompose()
    if not title:
        h1 = body.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else None
    for tr in body.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        tr.replace_with(NavigableString("\n" + " | ".join(cells) + "\n"))
    for h in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        h.replace_with(NavigableString(f"\n{_HEADING_MARK}{h.get_text(' ', strip=True)}\n"))
    for br in body.find_all(["br", "p", "div", "li", "section", "article"]):
        br.insert_after(NavigableString("\n"))
    raw = body.get_text("")
    # The text before the first heading has no heading; every other part starts with its heading line.
    parts = raw.split(_HEADING_MARK)
    sections: list[Section] = []
    if parts and parts[0].strip():
        sections.append(Section(parts[0], None, None))
    for part in parts[1:]:
        heading_line, _, rest = part.partition("\n")
        heading = heading_line.strip() or None
        if rest.strip():
            sections.append(Section(rest, None, heading))
    return ExtractedDocument("text/html", title, sections, links=links, meta_robots=meta_robots)
