"""``generate_html``: an HTML fragment from typed content blocks (spec 3.18). No LLM is involved.

Blocks: ``heading`` (text, level 2-4), ``paragraph`` (text), ``list`` (items, ordered), ``table``
(columns + rows, optional caption) and ``citations`` (items). Every value is content: Jinja2
autoescaping escapes it, so markup inside a value is shown as text, never interpreted. Only a fixed
set of tags is produced (div, h2-h4, p, ul, ol, li, table, caption, thead, tbody, tr, th, td, span).
A table (e.g. an execute_sql result) is a grid: ``<table class="km-grid">`` with a header row, which the
web page draws with cell borders. The header and cells of the columns named in ``sensitive_columns``
(data that can identify a person) get ``class="km-sensitive"``, shown in red, and a note under the
table names them. Tables are capped at ``html_output.max_table_rows`` rows and the fragment at ``html_output.max_chars``;
``truncated`` reports any cut. The web page sanitizes the fragment again before inserting it.
"""

from __future__ import annotations

from typing import Any, Literal

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, Field, model_validator

ALLOWED_TAGS = ("div", "h2", "h3", "h4", "p", "ul", "ol", "li", "table", "caption", "thead", "tbody", "tr", "th",
                "td", "span")


class HtmlBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["heading", "paragraph", "list", "table", "citations"]
    text: str | None = Field(default=None, max_length=20000, description="heading / paragraph text.")
    level: int = Field(default=2, ge=2, le=4, description="heading level (2-4).")
    items: list[str] = Field(default_factory=list, max_length=500, description="list / citations items.")
    ordered: bool = Field(default=False, description="numbered list.")
    caption: str | None = Field(default=None, max_length=500, description="table caption.")
    columns: list[str] = Field(default_factory=list, max_length=200, description="table column headers.")
    rows: list[list[Any]] = Field(default_factory=list, description="table rows (values are shown as text).")
    result_ref: str | None = Field(default=None, max_length=16,
                                   description="Agent only: evidence ID of an execute_sql result (e.g. R1) to "
                                               "render as this table instead of columns/rows.")
    sensitive_columns: list[str] = Field(default_factory=list, max_length=200,
                                         description="table: names (from columns) of the columns holding sensitive "
                                                     "data, i.e. execute_sql columns with sensitive=true; their "
                                                     "header and cells are marked in red. Filled in automatically "
                                                     "from result_ref in the agent.")

    @model_validator(mode="after")
    def _content(self) -> "HtmlBlock":
        if self.type in ("heading", "paragraph") and not (self.text or "").strip():
            raise ValueError(f"a {self.type} block needs text")
        if self.type in ("list", "citations") and not self.items:
            raise ValueError(f"a {self.type} block needs items")
        if self.type == "table" and not (self.columns or self.result_ref):
            raise ValueError("a table block needs columns and rows (or result_ref)")
        return self


class HtmlRequest(BaseModel):
    """Input of ``generate_html``."""

    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, max_length=300, description="Optional title (rendered as a heading).")
    blocks: list[HtmlBlock] = Field(min_length=1, max_length=50, description="Content blocks, in display order.")


_TEMPLATE = """<div class="km-html">
{%- if title %}<h2>{{ title }}</h2>{% endif -%}
{%- for b in blocks -%}
{%- if b.type == "heading" %}<h{{ b.level }}>{{ b.text }}</h{{ b.level }}>
{%- elif b.type == "paragraph" %}<p>{{ b.text }}</p>
{%- elif b.type == "list" %}<{{ "ol" if b.ordered else "ul" }}>{% for i in b["items"] %}<li>{{ i }}</li>{% endfor %}</{{ "ol" if b.ordered else "ul" }}>
{%- elif b.type == "citations" %}<ul class="km-citations">{% for i in b["items"] %}<li>{{ i }}</li>{% endfor %}</ul>
{%- elif b.type == "table" %}<table class="km-grid">{% if b.caption %}<caption>{{ b.caption }}</caption>{% endif %}<thead><tr>{% for c in b.columns %}<th{% if loop.index0 in b.marked %} class="km-sensitive"{% endif %}>{{ c }}</th>{% endfor %}</tr></thead><tbody>{% for r in b.rows %}<tr>{% for v in r %}<td{% if loop.index0 in b.marked %} class="km-sensitive"{% endif %}>{{ v }}</td>{% endfor %}</tr>{% endfor %}</tbody></table>
{%- if b.marked %}<p class="km-note">Sensitive data (can identify a person) in red: {{ b.marked_names | join(", ") }}.</p>{% endif %}
{%- if b.omitted %}<p class="km-note">{{ b.omitted }} more row(s) not shown.</p>{% endif %}
{%- endif -%}
{%- endfor -%}
</div>"""
_ENV = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
_TPL = _ENV.from_string(_TEMPLATE)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v)) if abs(v) < 1e15 else str(v)
    return str(v)


def render_html(request: HtmlRequest | dict[str, Any], max_chars: int = 200000,
                max_table_rows: int = 500) -> dict[str, Any]:
    """Render the blocks; values are escaped. ``result_ref`` blocks must be resolved by the caller."""
    req = request if isinstance(request, HtmlRequest) else HtmlRequest.model_validate(request)
    truncated = False
    blocks: list[dict[str, Any]] = []
    for b in req.blocks:
        if b.type == "table" and b.result_ref and not b.columns:
            raise ValueError(f"table result_ref {b.result_ref} was not resolved")
        item = b.model_dump()
        if b.type == "table":
            unknown = sorted(set(b.sensitive_columns) - set(b.columns))
            if unknown:
                raise ValueError(f"sensitive_columns not in the table's columns: {', '.join(unknown)[:200]}")
            item["marked"] = [i for i, c in enumerate(b.columns) if c in set(b.sensitive_columns)]
            item["marked_names"] = list(dict.fromkeys(b.columns[i] for i in item["marked"]))
            width = len(b.columns)
            rows = [[_cell(v) for v in (list(r) + [None] * width)[:width]] for r in b.rows]
            item["omitted"] = max(0, len(rows) - max_table_rows)
            item["rows"] = rows[:max_table_rows]
            truncated = truncated or bool(item["omitted"])
        blocks.append(item)
    html = _TPL.render(title=req.title, blocks=blocks)
    while len(html) > max_chars and blocks:
        truncated = True
        last = blocks[-1]
        if last["type"] == "table" and len(last["rows"]) > 1:
            cut = max(1, len(last["rows"]) // 2)
            last["omitted"] += len(last["rows"]) - cut
            last["rows"] = last["rows"][:cut]
        else:
            blocks.pop()
        html = _TPL.render(title=req.title, blocks=blocks)
    return {"status": "ok", "html": html, "blocks_rendered": len(blocks), "truncated": truncated,
            "format": "html_fragment", "allowed_tags": list(ALLOWED_TAGS)}
