"""``generate_sql``: an LLM writes one read-only SELECT from the evidence the caller passes.

The caller (the search agent's executor through MCP, or any MCP client) passes ALL the evidence: the intent,
the target source, the tables with their columns and keys, the joins and the business facts. Nothing is looked
up in the catalog; only the SQL dialect comes from the source's configuration. The query is checked (read-only,
cataloged tables of that source) but never run. The model is the agent model (``agent`` settings) with
``agent.sql_generation_prompt_path``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from ..common.errors import NotConfigured
from .sql_exec import SqlExecutor, SqlRejected


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")  # items may be passed as get_dataset / query_knowledge_graph return them


class SqlColumn(_Evidence):
    name: str = Field(min_length=1, max_length=200)
    data_type: str | None = Field(default=None, max_length=100)
    business_term: str | None = Field(default=None, max_length=200)


class SqlKey(_Evidence):
    type: str = Field(min_length=1, max_length=20, description="primary, unique or foreign")
    columns: list[str] = Field(min_length=1, max_length=20)
    references: str | None = Field(default=None, max_length=400, description="Foreign keys: table(columns).")


class SqlTable(_Evidence):
    name: str = Field(min_length=1, max_length=300,
                      description="Table name as cataloged, schema-qualified when the table has a schema.")
    columns: list[SqlColumn] = Field(min_length=1, max_length=300)
    keys: list[SqlKey] = Field(default_factory=list, max_length=50)


class SqlColumnPair(_Evidence):
    from_column: str = Field(min_length=1, max_length=200)
    to_column: str = Field(min_length=1, max_length=200)


class SqlJoin(_Evidence):
    from_table: str = Field(min_length=1, max_length=300)
    to_table: str = Field(min_length=1, max_length=300)
    column_pairs: list[SqlColumnPair] = Field(min_length=1, max_length=20, description="Ordered column pairs.")


class SqlGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=64)
    intent: str = Field(min_length=1, max_length=2000)
    tables: list[SqlTable] = Field(min_length=1, max_length=20)
    joins: list[SqlJoin] = Field(default_factory=list, max_length=30)
    facts: list[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(default_factory=list, max_length=30)


def _table_line(t: SqlTable) -> str:
    cols = "; ".join(f"{c.name} {c.data_type or ''}".strip()
                     + (f" (term: {c.business_term})" if c.business_term else "") for c in t.columns)
    keys = "; ".join(f"{k.type} key ({', '.join(k.columns)})" + (f" -> {k.references}" if k.references else "")
                     for k in t.keys)
    return f"table {t.name}: columns: {cols}" + (f"; keys: {keys}" if keys else "")


def _join_line(j: SqlJoin) -> str:
    on = " AND ".join(f"{j.from_table}.{p.from_column} = {j.to_table}.{p.to_column}" for p in j.column_pairs)
    return f"join {j.from_table} -> {j.to_table} on {on}"


class SqlGenerator:
    def __init__(self, prompt_path: Path, sql: SqlExecutor, llm_factory: Callable[[], Any] | None):
        self.prompt_path = prompt_path
        self.sql = sql
        self._factory = llm_factory
        self._llm = None
        self._lock = threading.Lock()

    def _client(self):
        if self._factory is None:
            raise NotConfigured("no agent model is available for SQL generation")
        with self._lock:
            if self._llm is None:
                self._llm = self._factory()
            return self._llm

    def generate(self, req: SqlGenerationRequest, structured_source_ids: list[str] | None = None,
                 dataset_ids: list[str] | None = None) -> dict[str, Any]:
        from ..agent.plan import parse_json_object, prompt_version

        llm = self._client()
        try:
            system = Path(self.prompt_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise NotConfigured(f"SQL generation prompt not readable: {self.prompt_path}") from exc
        try:
            dialect = self.sql.dialect(req.source_id)
        except SqlRejected as exc:
            return {"status": "error", "error": {"code": exc.code, "message": exc.message}}
        facts = [_join_line(j) for j in req.joins] + list(req.facts)
        prompt = (f"Target source: {req.source_id}\nSQL dialect: {dialect}\n\nIntent:\n{req.intent}\n\n"
                  "Tables (only these exist; use their exact names):\n" + "\n".join(_table_line(t) for t in req.tables)
                  + "\n\nJoins and business facts from the knowledge graph and catalog:\n"
                  + ("\n".join(facts) or "(none)") + "\n\nReturn the JSON object.")
        resp = llm.generate(prompt, system=system, purpose="sql_generation", json_output=True,
                            prompt_version=prompt_version(system))
        try:
            reply = parse_json_object(resp.text)
        except ValueError as exc:
            return {"status": "error", "error": {"code": "invalid_reply", "message": str(exc)[:200]}}
        sql = str(reply.get("sql") or "").strip().rstrip(";").strip()
        if not sql:
            return {"status": "cannot_answer", "explanation": str(reply.get("explanation") or "")[:1000],
                    "hint": "The model found no way to answer this intent from the given tables."}
        try:
            checked = self.sql.validate(req.source_id, sql, structured_source_ids, dataset_ids)
        except SqlRejected as exc:
            return {"status": "invalid", "sql": sql, "dialect": dialect,
                    "validation": {"ok": False, "code": exc.code, "message": exc.message},
                    "hint": "Fix the problem (call generate_sql again with more evidence) before execute_sql."}
        return {"status": "ok", "sql": sql, "dialect": dialect, "source_id": req.source_id,
                "tables": [t["catalog_ref"] for t in checked["tables"]],
                "explanation": str(reply.get("explanation") or "")[:1000], "validation": {"ok": True},
                "hint": "Pass this sql and source_id to execute_sql."}
