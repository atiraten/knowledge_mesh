"""Business term (per column) and business entity (per table/dataset) generation (FR-12).

The prompt is ``prompts/term_generation.txt``; placeholders ``{industry_name}``,
``{language_name}``, ``{table}``, ``{columns}``, ``{class_words_section}`` and
``{industry_abbreviation}`` are filled from ``term_generation`` settings. ``{columns}`` lists one column
per line: ``name (TYPE) ["v1" "v2"]`` with the cataloged data type and up to 5 of the column's most common
profiled values (each cut at 60 characters; omitted when profiling stored none). A column whose current
business term is flagged sensitive gets no values. Connector secrets are never sent.

Output lines ``<table>,<column>,<term>,<sensitive>,<description>,<comments>`` are validated against
the cataloged columns before anything is published: unknown columns are dropped, missing
columns are retried once and then reported as failures, and terms must be 1-10 words.
``<sensitive>`` (Y/N) says whether the term can identify a physical person, alone or combined with
other information (PII / PHI / GDPR); it is stored as ``business_terms.sensitive`` and drives the
red marking of SQL results. A line without it (older prompt) leaves the flag unknown (NULL).

Terms and entities are recorded in the business glossary (``structured/glossary_store.py``): the column
gets ``term_slug`` and its decoding, the term row its label, description and flag. Edits made in the
glossary are kept: a column bound to a manual / edited term, or whose term was deleted in the glossary
(``term_status = 'removed'``), is not regenerated, and a dataset whose entity was edited or deleted
keeps it.

Review and correction (``prompts/term_correction.txt``, ``term_generation.correction_prompt_path``):
after generation the same model reviews the generated terms, one row per column
(``column|data_type|business term|description|decoding``) and answers ``column|APPROVE|`` or
``column|REJECT|<reason>`` per row; a row without a verdict counts as rejected. Rejected columns go
back to the generation prompt (only those columns, plus a "CORRECTIONS REQUIRED" section with the
previous term and the reason), then the corrected rows are reviewed again, at most
``max_correction_rounds`` times (default 3). A column still rejected after that is stored blank:
no term, ``term_status = 'rejected'`` and the last reason in ``term_error``. When the correction prompt
file does not exist the review is skipped and the run says so.

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
from ..common.ids import new_id, sha256_text, utcnow
from ..jobs import JobContext
from ..observability import get_metrics, span
from ..storage.db import Database, loads
from .catalog import CatalogService
from .glossary_store import MANUAL_ORIGINS, GlossaryStore

log = logging.getLogger(__name__)
_PLACEHOLDERS = ("industry_name", "language_name", "table", "columns", "class_words_section", "industry_abbreviation")
_REVIEW_PLACEHOLDERS = ("industry_name", "language_name", "class_words", "abbreviations", "table_name", "rows")
_REVIEW_HEADER = "column|data_type|business term|description|decoding"
_VERDICT_HEADER = re.compile(r"^\|?\s*column\s*\|\s*status\b", re.I)
NO_VERDICT = "the reviewer returned no verdict"


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


MAX_SAMPLES = 5
SAMPLE_CHARS = 60


def column_line(name: str, data_type: str | None = None, samples: list[Any] | None = None) -> str:
    """One ``{columns}`` entry: ``name (TYPE) ["v1" "v2"]``; the type and the sample values only when known."""
    out = sanitize_identifier(name)
    if data_type:
        out += f" ({_one_line(data_type)})"
    values = []
    for v in samples or []:
        text = _one_line(v)[:SAMPLE_CHARS]
        if text and len(values) < MAX_SAMPLES:
            values.append('"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"')
    if values:
        out += " [" + " ".join(values) + "]"
    return out


def render_prompt(template: str, table: str, columns: list[str], settings, abbreviations: dict[str, str],
                  details: dict[str, str] | None = None) -> str:
    """``details`` maps a column name to its ``column_line``; the list is then one column per line."""
    class_words = list(dict.fromkeys(settings.class_words or []))  # each class word once
    values = {
        "industry_name": settings.industry_name,
        "language_name": settings.language_name,
        "table": sanitize_identifier(table),
        "columns": ("\n" + "\n".join(details.get(c) or sanitize_identifier(c) for c in columns)
                    if details is not None else ", ".join(sanitize_identifier(c) for c in columns)),
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


def cut_off_note(response: Any, purpose: str) -> str | None:
    """Issue line for a model answer cut off at max_output_tokens (thinking tokens count against it)."""
    if not getattr(response, "truncated", False):
        return None
    return (f"{purpose} response cut off at max_output_tokens ({getattr(response, 'output_tokens', None) or 0} "
            f"visible + {getattr(response, 'thinking_tokens', None) or 0} thinking tokens)")


def _one_line(value: Any) -> str:
    """A value placed inside a pipe-separated row: no pipes, no line breaks."""
    return re.sub(r"\s+", " ", re.sub(r"[|\r\n]+", " ", str(value or ""))).strip()


def _abbreviation_lines(abbreviations: dict[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in sorted(abbreviations.items())) or "(none provided)"


def review_rows(records: list[TermRecord], data_types: dict[str, str | None]) -> list[str]:
    """``{rows}`` of the correction prompt: the generated output of each column with its cataloged data type."""
    return [_REVIEW_HEADER] + ["|".join([sanitize_identifier(r.column), _one_line(data_types.get(r.column)),
                                         _one_line(r.term), _one_line(r.description), _one_line(r.explanation)])
                               for r in records]


def render_review_prompt(template: str, table: str, rows: list[str], settings,
                         abbreviations: dict[str, str]) -> str:
    values = {
        "industry_name": settings.industry_name,
        "language_name": settings.language_name,
        "class_words": ", ".join(dict.fromkeys(settings.class_words or [])) or "(none provided)",
        "abbreviations": _abbreviation_lines(abbreviations),
        "table_name": sanitize_identifier(table),
        "rows": "\n".join(rows),
    }
    out = template
    for key in _REVIEW_PLACEHOLDERS:
        out = out.replace("{" + key + "}", values[key])
    return out


def parse_review_output(text: str, columns: list[str]) -> dict[str, tuple[str, str]]:
    """``{column: ("approve" | "reject", reason)}`` from ``column|STATUS|reason`` lines; other lines are ignored."""
    by_lower = {c.lower(): c for c in columns}
    verdicts: dict[str, tuple[str, str]] = {}
    for raw in text.splitlines():
        line = re.sub(r"^[-*]\s+", "", raw.strip().strip("`").strip()).replace("**", "")
        if not line or line.startswith("```") or "|" not in line or _VERDICT_HEADER.match(line):
            continue
        parts = [x.strip() for x in line.split("|")]
        if parts[0] == "":  # markdown table row: | col | STATUS | reason |
            parts = parts[1:]
        if len(parts) < 2:
            continue
        col = by_lower.get(parts[0].strip("'\"`").lower())
        status = parts[1].strip("'\"`").lower()
        if col is None or col in verdicts:
            continue
        if status in ("approve", "approved"):
            verdicts[col] = ("approve", "")
        elif status in ("reject", "rejected"):
            verdicts[col] = ("reject", " | ".join(x for x in parts[2:] if x)[:500] or "no reason given")
    return verdicts


def correction_section(rejected: dict[str, str], records: dict[str, TermRecord]) -> str:
    """Appended to the generation prompt of a correction call: what the reviewer rejected and why."""
    lines = ["", "### CORRECTIONS REQUIRED",
             "A reviewer rejected these business terms. Generate a corrected line for each column, "
             "fixing the reason given."]
    for col, reason in rejected.items():
        prev = records.get(col)
        lines.append(f"{sanitize_identifier(col)}|previous term: {_one_line(prev.term) if prev else '(none)'}"
                     f"|reason: {_one_line(reason)}")
    return "\n".join(lines)


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
    def review_template(self) -> str | None:
        """The term correction (review) prompt, or None when its file does not exist (review skipped)."""
        path = getattr(self.cfg, "correction_prompt_path", None)
        return Path(path).read_text(encoding="utf-8") if path and Path(path).exists() else None

    @property
    def prompt_version(self) -> str:
        review = self.review_template  # the terms depend on both prompts
        return sha256_text(self.template + ("\n" + review if review is not None else ""))[:12]

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
        details = self._column_details(dataset_id, columns)
        version = self.prompt_version
        # edits made in the business glossary are kept, even with force
        editable = [c for c in columns if c["term_status"] != "removed" and c.get("term_origin") not in MANUAL_ORIGINS]
        pending = [c for c in editable
                   if force or c["term_status"] != "generated" or c["term_prompt_version"] != version]
        keep_entity = dataset.get("entity_method") == "removed" or dataset.get("entity_origin") in MANUAL_ORIGINS
        abbreviations = load_abbreviations(self.cfg)
        if not pending and (dataset.get("entity_label") or keep_entity):
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
                    prompt = render_prompt(self.template, dataset["name"], batch, self.cfg, abbreviations, details)
                    sp.set_attribute("llm.prompt.chars", len(prompt))
                    response = llm.generate(prompt, purpose="term_generation")
                    if note := cut_off_note(response, "term_generation"):
                        issues.append(note)
                    parsed, batch_issues = parse_term_output(response.text, dataset["name"], batch)
                    records.update(parsed)
                    issues.extend(batch_issues)
                missing = [n for n in names if n not in records]
                if not missing:
                    break
                batches = [missing[i:i + self.cfg.max_columns_per_request]
                           for i in range(0, len(missing), self.cfg.max_columns_per_request)]
            rejected, rounds = self._review_and_correct(llm, dataset, columns, records, abbreviations, issues,
                                                        details)
            sp.set_attribute("terms.review_rounds", rounds)
            sp.set_attribute("terms.rejected", len(rejected))
            # first, so that the 2000-character cap of the run's error field never cuts them
            issues[:0] = [f"{name}: still rejected after {rounds} correction round(s), left blank: {reason}"
                          for name, reason in rejected.items()]
            decoded = Counter(r.decoded_table for r in records.values()
                              if r.decoded_table and r.decoded_table.lower() != dataset["name"].lower())
            if decoded:
                entity_label, method = re.sub(r"[^\w ]+", " ", decoded.most_common(1)[0][0]).strip(), "llm_decoded_table"
            else:
                entity_label, method = heuristic_entity_label(dataset["name"], abbreviations), "heuristic"
            entity_label = " ".join(entity_label.split()[:6]) or dataset["name"]
            if keep_entity:  # edited or deleted in the business glossary: kept as it is
                entity_label, method = dataset.get("entity_label"), dataset.get("entity_method")
            now = utcnow()
            failed = 0
            with self.db.transaction() as conn:
                for col in pending:
                    rec = records.get(col["name"])
                    slug = None
                    if rec and col["name"] not in rejected:
                        slug = GlossaryStore.generated_term(conn, rec.term, rec.description, rec.sensitive, now,
                                                            dataset_id)
                    if col["name"] in rejected:  # blank: no term, the reviewer's last reason is kept
                        failed += 1
                        conn.execute(
                            "UPDATE columns SET term_slug=NULL, term_explanation=NULL, term_run_id=?,"
                            " term_prompt_version=?, term_status='rejected', term_error=? WHERE column_id=?",
                            (run_id, version, f"rejected by review: {rejected[col['name']]}"[:2000],
                             col["column_id"]))
                    elif rec and slug is None:  # the generated term was deleted in the business glossary
                        conn.execute(
                            "UPDATE columns SET term_slug=NULL, term_explanation=NULL, term_run_id=?,"
                            " term_prompt_version=?, term_status='removed', term_error=? WHERE column_id=?",
                            (run_id, version, f"generated term {rec.term!r} was deleted from the business glossary",
                             col["column_id"]))
                    elif rec:
                        conn.execute(
                            "UPDATE columns SET term_slug=?, term_explanation=?, term_run_id=?, term_prompt_version=?,"
                            " term_status='generated', term_error=NULL WHERE column_id=?",
                            (slug, rec.explanation, run_id, version, col["column_id"]))
                    else:
                        failed += 1
                        conn.execute("UPDATE columns SET term_status='failed', term_error=? WHERE column_id=?",
                                     ("model output did not include this column", col["column_id"]))
                if not keep_entity:
                    entity_slug = GlossaryStore.generated_entity(conn, entity_label, now)
                    conn.execute("UPDATE datasets SET entity_slug=?, entity_method=?, entity_run_id=? WHERE dataset_id=?",
                                 (entity_slug, method if entity_slug else "removed", run_id, dataset_id))
                conn.execute(
                    "INSERT INTO term_generation_runs(run_id, job_id, dataset_id, provider, model, prompt_version,"
                    " status, error, generated, failed, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, job.job_id if job else None, dataset_id, llm.provider, llm.model, version,
                     "partial" if failed else "succeeded", "; ".join(issues)[:2000] or None,
                     len(records) - len(rejected), failed, now))
            metrics.terms.add(len(records) - len(rejected), {"outcome": "generated"})
            if rejected:
                metrics.terms.add(len(rejected), {"outcome": "rejected"})
            if failed - len(rejected):
                metrics.terms.add(failed - len(rejected), {"outcome": "failed"})
            self.glossary.upsert_dataset(dataset_id)
            sp.set_attribute("terms.generated", len(records) - len(rejected))
            sp.set_attribute("terms.failed", failed)
        result = {"dataset_id": dataset_id, "run_id": run_id, "status": "partial" if failed else "succeeded",
                  "generated": len(records) - len(rejected), "failed": failed, "rejected": sorted(rejected),
                  "review_rounds": rounds, "entity_label": entity_label,
                  "entity_method": method, "issues": issues[:20]}
        if job:
            job.add_item(dataset_id, "failed" if failed and not records else "succeeded",
                         f"{failed} column(s) without a term" if failed else None,
                         dataset=dataset["name"], generated=len(records) - len(rejected), failed_columns=failed,
                         rejected_columns=sorted(rejected), review_rounds=rounds, entity_label=entity_label)
        return result

    def _column_details(self, dataset_id: str, columns: list[dict[str, Any]]) -> dict[str, str]:
        """``column_line`` of each column: cataloged data type and the latest profile's most common values,
        except for a column whose current business term is flagged sensitive (no row values for it)."""
        values: dict[str, list[Any]] = {}
        for p in self.db.query("SELECT p.column_id, p.most_common_values_json FROM column_profiles p "
                               "JOIN profiling_runs r ON r.run_id = p.run_id WHERE p.dataset_id = ? "
                               "ORDER BY r.started_at", (dataset_id,)):
            values[p["column_id"]] = [v.get("value") for v in loads(p["most_common_values_json"], [])
                                      if isinstance(v, dict)]
        return {c["name"]: column_line(c["name"], c.get("data_type"),
                                       None if c.get("term_sensitive") == 1 else values.get(c["column_id"]))
                for c in columns}

    def _review_and_correct(self, llm, dataset: dict[str, Any], columns: list[dict[str, Any]],
                            records: dict[str, TermRecord], abbreviations: dict[str, str],
                            issues: list[str], details: dict[str, str] | None = None) -> tuple[dict[str, str], int]:
        """Review the generated terms, send rejected columns back for correction (up to
        ``max_correction_rounds`` times) and return ``({column: last reason} still rejected, rounds)``.
        ``records`` is updated in place with the corrected terms."""
        template = self.review_template
        if template is None:
            issues.append(f"term review skipped: correction prompt not found: {self.cfg.correction_prompt_path}")
            return {}, 0
        if not records:
            return {}, 0
        size = self.cfg.max_columns_per_request
        data_types = {c["name"]: c["data_type"] for c in columns}
        table = dataset["name"]

        reviews = 0

        def review(names: list[str]) -> dict[str, str]:
            nonlocal reviews
            reviews += 1
            verdicts: dict[str, tuple[str, str]] = {}
            for i in range(0, len(names), size):
                batch = names[i:i + size]
                prompt = render_review_prompt(template, table, review_rows([records[n] for n in batch], data_types),
                                              self.cfg, abbreviations)
                response = llm.generate(prompt, purpose="term_review")
                if note := cut_off_note(response, f"review {reviews}: term_review"):
                    issues.append(note)
                verdicts.update(parse_review_output(response.text, batch))
            out = {n: verdicts[n][1] if n in verdicts else NO_VERDICT
                   for n in names if verdicts.get(n, ("reject", ""))[0] == "reject"}
            issues.extend(f"review {reviews}: {n} rejected: {reason}" for n, reason in out.items())  # audit trail
            return out

        generation = self.template
        rejected = review(list(records))
        rounds = 0
        while rejected and rounds < self.cfg.max_correction_rounds:
            rounds += 1
            names = list(rejected)
            corrected: dict[str, TermRecord] = {}
            for i in range(0, len(names), size):
                batch = names[i:i + size]
                prompt = (render_prompt(generation, table, batch, self.cfg, abbreviations, details)
                          + "\n" + correction_section({n: rejected[n] for n in batch}, records))
                response = llm.generate(prompt, purpose="term_correction")
                if note := cut_off_note(response, f"correction round {rounds}: term_correction"):
                    issues.append(note)
                parsed, batch_issues = parse_term_output(response.text, table, batch)
                corrected.update(parsed)
                issues.extend(f"correction round {rounds}: {x}" for x in batch_issues)
            records.update(corrected)
            # columns the correction call did not return stay rejected (same reason) for the next round
            rejected = {**{n: rejected[n] for n in names if n not in corrected}, **review(list(corrected))}
        return rejected, rounds
