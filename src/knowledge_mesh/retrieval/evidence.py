"""Evidence blocks and citation checks shared by answer mode and the search agent's synthesis.

Each evidence item is rendered as a labeled block (``[D1] document passage ...``) that the model
must cite; citations in the model output are checked against the IDs actually supplied.
"""

from __future__ import annotations

import re
from typing import Any

# Any bracketed ID-like token; those not in the supplied evidence are dropped as unresolvable.
CITE = re.compile(r"\[([A-Z]{1,2}\d{1,3})\]")


def document_block(ident: str, d: dict[str, Any], printed: dict[str, str] | None = None) -> tuple[str, dict[str, Any]]:
    """A retrieved passage with its expanded parents. ``printed`` maps parent node IDs already shown in
    an earlier block, so shared parent text is printed once and later blocks point back to it."""
    loc = ", ".join(x for x in (d.get("title"), f"page {d['page']}" if d.get("page") else None,
                                d.get("section")) if x)
    body = f"Matched chunk:\n{d['text']}"
    for p in d.get("parents") or []:
        if not p.get("expanded"):
            continue
        label = f"Parent context (level {p['level']}, {p['node_id']})"
        if printed is not None and p["node_id"] in printed:
            body += f"\n{label}: same as in [{printed[p['node_id']]}]"
        else:
            if printed is not None:
                printed[p["node_id"]] = ident
            body += f"\n{label}:\n{p['text']}"
    entry = {"type": "document_passage", "ref": d["citation"]["chunk_ref"], "document_id": d["document_id"],
             "chunk_id": d["chunk_id"], "source_uri": d["source_uri"], "page": d.get("page"),
             "section": d.get("section"), "label": d.get("title") or d.get("source_uri")}
    return f"[{ident}] document passage ({loc or d['source_uri']}):\n{body}", entry


def glossary_block(ident: str, g: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    desc = f" — {g['description']}" if g.get("description") else ""
    where = f"{g['dataset']}.{g['column']}" if g.get("column") else g.get("dataset")
    text = (f"[{ident}] {g['type'].replace('_', ' ')} '{g['label']}' (catalog {g['catalog_ref']}, "
            f"physical {where}){desc}")
    entry = {"type": g["type"], "ref": g["source_ref"], "catalog_ref": g["catalog_ref"],
             "graph_ref": g["graph_ref"], "label": g["label"]}
    return text, entry


def graph_block(ident: str, k: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tag = " (inferred)" if k.get("inferred") else ""
    entry = {"type": k["type"], "ref": k.get("source_ref"), "graph_ref": k["graph_ref"],
             "label": k.get("label"), "inferred": bool(k.get("inferred"))}
    return f"[{ident}] {k['type'].replace('_', ' ')}{tag}: {k['text']}", entry


def check_citations(text: str, catalog: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Return (text without unknown citations, cited known IDs, dropped unknown IDs)."""
    cited, invalid = [], []
    for ident in dict.fromkeys(CITE.findall(text)):
        (cited if ident in catalog else invalid).append(ident)
    for ident in invalid:
        text = text.replace(f"[{ident}]", "")
    text = re.sub(r"[ \t]{2,}", " ", text).replace(" .", ".")
    return text.strip(), cited, invalid
