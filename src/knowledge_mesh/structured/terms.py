"""Business term (per column) and business entity (per table/dataset) generation (FR-12).

The prompt is ``prompts/term_generation.txt``; placeholders ``{industry_name}``,
``{language_name}``, ``{table}``, ``{columns}``, ``{class_words_section}`` and
``{industry_abbreviation}`` are filled from ``term_generation`` settings. Only catalog
metadata (table and column names) is sent — never connector secrets or row values.

Output lines ``<table>,<column>,<term>,<sensitive>,<description>,<comments>`` are validated against
the cataloged columns before anything is published: unknown columns are dropped, missing
columns are retried once and then reported as failures, and terms must be 1-10 words.
``<sensitive>`` (Y/N) says whether the term can identify a physical person, alone or combined with
other information (PII / PHI / GDPR); it is stored as ``columns.term_sensitive`` and drives the
red marking of SQL results. A line without it (older prompt) leaves the flag unknown (NULL).

Entity label (open decision in the spec, resolved here): the prompt asks the model to start
each line with the *decoded* table name; when that differs from the physical name it becomes
the entity label (method ``llm_decoded_table``). Otherwise the label is derived from the
physical name: split on ``_``/camelCase, expand configured abbreviations, singularize, and
title-case (method ``heuristic``).
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from ..common.errors import NotConfigured, NotFound
from ..common.ids import new_id, sha256_text, slugify, utcnow
from ..jobs import JobContext
from ..observability import get_metrics, span
from ..storage.db import Database
from .catalog import CatalogService

log = logging.getLogger(__name__)
_PLACEHOLDERS = ("industry_name", "language_name", "table", "columns", "class_words_section", "industry_abbreviation")


@dataclass
class TermRecord:
    column: str
    term: str
    description: str
    explanation: str
    decoded_table: str
    sensitive: bool | None = None  # None: the model gave no Y/N flag


def sanitize_identifier(name: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", "", str(name)).replace(",", "_")[:128]


def load_abbreviations(settings) -> dict[str, str]:
    abbr = {k.lower(): v for k, v in (settings.abbreviations or {}).items()}
    path: Path | None = settings.abbreviations_path
    if path and path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
            abbr.update({str(k).lower(): str(v) for k, v in data.items()})
        else:
            for row in csv.reader(text.splitlines()):
                if len(row) == 1 and "=" in row[0]:
                    row = row[0].split("=", 1)
                if len(row) >= 2 and row[0].strip() and not row[0].startswith("#"):
                    abbr[row[0].strip().lower()] = row[1].strip()
    return abbr


def render_prompt(template: str, table: str, columns: list[str], settings, abbreviations: dict[str, str]) -> str:
    class_words = settings.class_words or []
    values = {
        "industry_name": settings.industry_name,
        "language_name": settings.language_name,
        "table": sanitize_identifier(table),
        "columns": ", ".join(sanitize_identifier(c) for c in columns),
        "class_words_section": ("   Allowed class words: " + ", ".join(class_words)) if class_words else "",
        "industry_abbreviation": "\n".join(f"{k}={v}" for k, v in sorted(abbreviations.items())) or "(none provided)",
    }
    out = template
    for key in _PLACEHOLDERS:
        out = out.replace("{" + key + "}", values[key])
    return out


_HEADER = re.compile(r"^<?\s*table[ _]name\s*>?\s*,", re.I)
_FLAGS = {"y": True, "yes": True, "true": True, "n": False, "no": False, "false": False}


def parse_term_output(text: str, table: str, columns: list[str]) -> tuple[dict[str, TermRecord], list[str]]:
    by_lower = {c.lower(): c for c in columns}
    records: dict[str, TermRecord] = {}
    issues: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("`").strip()
        if not line or line.startswith("```") or line.startswith("#") or _HEADER.match(line):
            continue
        line = re.sub(r"^[-*]\s+", "", line).replace("**", "")
        try:
            fields = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            issues.append(f"unparseable line: {line[:80]}")
            continue
        if len(fields) < 3:
            issues.append(f"line has fewer than 3 fields: {line[:80]}")
            continue
        tbl, col = fields[0].strip().strip("'\""), fields[1].strip().strip("'\"`")
        term = re.sub(r"[^\w ]+", " ", fields[2], flags=re.UNICODE)
        term = re.sub(r"\s+", " ", term).strip()
        rest = fields[3:]
        sensitive = _FLAGS.get(rest[0].strip().strip("'\"`").lower()) if rest else None
        if sensitive is not None:
            rest = rest[1:]
        description, explanation = _split_description(rest, spaced=all(f.startswith(" ") for f in fields[1:]))
        actual = by_lower.get(col.lower())
        if actual is None:
            issues.append(f"model returned unknown column {col[:60]!r}; dropped")
            continue
        if not term or not 1 <= len(term.split()) <= 10:
            issues.append(f"invalid business term for {actual}")
            continue
        if actual.lower() in {k.lower() for k in records}:
            continue
        records[actual] = TermRecord(actual, term, description[:2000], explanation[:2000], tbl[:120], sensitive)
    labels = Counter(r.term.lower() for r in records.values())
    for label, n in labels.items():
        if n > 1:
            issues.append(f"{n} columns of {table} share the business term {label!r}")
    return records, issues


def _split_description(rest: list[str], spaced: bool) -> tuple[str, str]:
    """Separate description from comments when free text contains unquoted commas.

    The prompt forbids spaces around separator commas, so a field starting with a space
    continues the previous field — unless it holds ``segment=resolution`` pairs, which always
    start the comments. If the model spaced every comma, split at the first such field.
    """
    if not rest:
        return "", ""
    if spaced:
        split_at = next((i for i, f in enumerate(rest) if "=" in f and i > 0), None)
        if split_at is None:
            return rest[0].strip(), ",".join(rest[1:]).strip()
        return ",".join(rest[:split_at]).strip(), ",".join(rest[split_at:]).strip()
    merged: list[str] = []
    for f in rest:
        if merged and f.startswith(" ") and "=" not in f:
            merged[-1] += "," + f
        else:
            merged.append(f)
    return merged[0].strip(), ",".join(merged[1:]).strip()


def heuristic_entity_label(table: str, abbreviations: dict[str, str]) -> str:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", table)
    parts = [p for p in re.split(r"[_\W]+", words) if p]
    parts = [p for p in parts if p.lower() not in {"tbl", "tb", "t", "dim", "fact", "stg", "vw", "v"}] or parts
    expanded = [abbreviations.get(p.lower(), p) for p in parts]
    if expanded:
        last = expanded[-1]
        low = last.lower()
        if low.endswith("ies") and len(low) > 4:
            last = last[:-3] + "y"
        elif low.endswith(("sses", "xes", "ches", "shes")):
            last = last[:-2]
        elif low.endswith("s") and not low.endswith("ss") and len(low) > 3:
            last = last[:-1]
        expanded[-1] = last
    return " ".join(w[:1].upper() + w[1:].lower() if w.islower() or w.isupper() else w for w in expanded) or table


class TermGenerationService:
    def __init__(self, settings, db: Database, catalog: CatalogService, glossary,
                 llm_factory: Callable[[], Any] | None):
        self.cfg = settings
        self.db = db
        self.catalog = catalog
        self.glossary = glossary
        self.llm_factory = llm_factory
        self._llm = None

    @property
    def template(self) -> str:
        path = self.cfg.prompt_path
        if not path or not Path(path).exists():
            raise NotConfigured(f"term generation prompt not found: {path}")
        return Path(path).read_text(encoding="utf-8")

    @property
    def prompt_version(self) -> str:
        return sha256_text(self.template)[:12]

    def available(self) -> tuple[bool, str | None]:
        if self.llm_factory is None:
            return False, "term_generation provider/model not configured"
        try:
            self.llm()
        except NotConfigured as exc:
            return False, exc.message
        return True, None

    def llm(self):
        if self._llm is None:
            if self.llm_factory is None:
                raise NotConfigured("term generation model is not configured")
            self._llm = self.llm_factory()
        return self._llm

    def generate_for_dataset(self, dataset_id: str, force: bool = False,
                             job: JobContext | None = None) -> dict[str, Any]:
        metrics = get_metrics()
        dataset = self.catalog.dataset_row(dataset_id)
        if not dataset["active"]:
            raise NotFound(f"dataset {dataset_id} is retired")
        columns = self.catalog.active_columns(dataset_id)
        version = self.prompt_version
        pending = [c for c in columns if force or c["term_status"] != "generated" or c["term_prompt_version"] != version]
        abbreviations = load_abbreviations(self.cfg)
        if not pending and dataset.get("entity_label"):
            if job:
                job.add_item(dataset_id, "succeeded", "terms unchanged", dataset=dataset["name"])
            return {"dataset_id": dataset_id, "status": "unchanged", "generated": 0, "failed": 0}
        run_id = new_id("terms")
        llm = self.llm()
        with span("terms.generate", "CHAIN", {"dataset.id": dataset_id, "terms.columns": len(pending),
                                               "llm.prompt_template.version": version}) as sp:
            names = [c["name"] for c in pending]
            records: dict[str, TermRecord] = {}
            issues: list[str] = []
            batches = [names[i:i + self.cfg.max_columns_per_request]
                       for i in range(0, len(names), self.cfg.max_columns_per_request)]
            for attempt in range(2):
                for batch in batches:
                    if not batch:
                        continue
                    prompt = render_prompt(self.template, dataset["name"], batch, self.cfg, abbreviations)
                    sp.set_attribute("llm.prompt.chars", len(prompt))
                    response = llm.generate(prompt, purpose="term_generation")
                    parsed, batch_issues = parse_term_output(response.text, dataset["name"], batch)
                    records.update(parsed)
                    issues.extend(batch_issues)
                missing = [n for n in names if n not in records]
                if not missing:
                    break
                batches = [missing[i:i + self.cfg.max_columns_per_request]
                           for i in range(0, len(missing), self.cfg.max_columns_per_request)]
            decoded = Counter(r.decoded_table for r in records.values()
                              if r.decoded_table and r.decoded_table.lower() != dataset["name"].lower())
            if decoded:
                entity_label, method = re.sub(r"[^\w ]+", " ", decoded.most_common(1)[0][0]).strip(), "llm_decoded_table"
            else:
                entity_label, method = heuristic_entity_label(dataset["name"], abbreviations), "heuristic"
            entity_label = " ".join(entity_label.split()[:6]) or dataset["name"]
            now = utcnow()
            failed = 0
            with self.db.transaction() as conn:
                for col in pending:
                    rec = records.get(col["name"])
                    if rec:
                        conn.execute(
                            "UPDATE columns SET term_label=?, term_slug=?, term_description=?, term_explanation=?,"
                            " term_run_id=?, term_prompt_version=?, term_status='generated', term_error=NULL,"
                            " term_sensitive=? WHERE column_id=?",
                            (rec.term, slugify(rec.term), rec.description, rec.explanation, run_id, version,
                             None if rec.sensitive is None else int(rec.sensitive), col["column_id"]))
                    else:
                        failed += 1
                        conn.execute("UPDATE columns SET term_status='failed', term_error=? WHERE column_id=?",
                                     ("model output did not include this column", col["column_id"]))
                conn.execute("UPDATE datasets SET entity_label=?, entity_slug=?, entity_method=?, entity_run_id=? "
                             "WHERE dataset_id=?", (entity_label, slugify(entity_label), method, run_id, dataset_id))
                conn.execute(
                    "INSERT INTO term_generation_runs(run_id, job_id, dataset_id, provider, model, prompt_version,"
                    " status, error, generated, failed, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, job.job_id if job else None, dataset_id, llm.provider, llm.model, version,
                     "partial" if failed else "succeeded", "; ".join(issues)[:2000] or None, len(records), failed, now))
            metrics.terms.add(len(records), {"outcome": "generated"})
            if failed:
                metrics.terms.add(failed, {"outcome": "failed"})
            self.glossary.upsert_dataset(dataset_id)
            sp.set_attribute("terms.generated", len(records))
            sp.set_attribute("terms.failed", failed)
        result = {"dataset_id": dataset_id, "run_id": run_id, "status": "partial" if failed else "succeeded",
                  "generated": len(records), "failed": failed, "entity_label": entity_label,
                  "entity_method": method, "issues": issues[:20]}
        if job:
            job.add_item(dataset_id, "failed" if failed and not records else "succeeded",
                         f"{failed} column(s) without a term" if failed else None,
                         dataset=dataset["name"], generated=len(records), failed_columns=failed,
                         entity_label=entity_label)
        return result
