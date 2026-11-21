"use strict";
// Knowledge Mesh browser UI. All server data is rendered with textContent (never innerHTML).
// The one exception, generate_html fragments, is rebuilt element by element through sanitizeFragment().

const view = document.getElementById("view");

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "href") node.setAttribute("href", safeHref(v));
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
function safeHref(v) {
  const s = String(v);
  return (s.startsWith("#") || s.startsWith("/") || /^https?:\/\//i.test(s)) ? s : "#";
}
function badge(text, kind) { return el("span", { class: `badge ${kind || ""}` }, text); }
function statusBadge(s) {
  const kind = { succeeded: "ok", indexed: "ok", extracted: "info", partial: "warn", running: "info",
                 queued: "info", skipped: "warn", failed: "bad", deleted: "bad",
                 cataloged: "ok", empty: "bad", unavailable: "bad" }[s] || "";
  return badge(s, kind);
}
function toast(msg, isError) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.background = isError ? "var(--bad)" : "";
  t.style.display = "block"; clearTimeout(toast._h); toast._h = setTimeout(() => (t.style.display = "none"), 5000);
}
function fmt(v) { return v === null || v === undefined || v === "" ? "—" : String(v); }

function sessionId() {  // one id per browser tab: Phoenix groups this tab's searches into one session
  let id = null;
  try { id = sessionStorage.getItem("meshSessionId"); } catch (e) { /* storage blocked */ }
  if (!id) {
    id = "ui-" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    try { sessionStorage.setItem("meshSessionId", id); } catch (e) { /* storage blocked */ }
  }
  return id;
}

async function api(method, path, body, isForm) {
  const headers = { "X-Session-Id": sessionId() };
  const key = sessionStorage.getItem("meshApiKey");
  if (key) headers["X-API-Key"] = key;
  let payload;
  if (isForm) payload = body;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(path, { method, headers, body: payload });
  if (res.status === 401) {
    const k = prompt("API key required");
    if (k) { sessionStorage.setItem("meshApiKey", k); return api(method, path, body, isForm); }
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error((data.error && data.error.message) || res.statusText);
    err.data = data; err.status = res.status; throw err;
  }
  return data;
}
function showError(container, err) {
  const details = err.data && err.data.error && err.data.error.details;
  container.append(el("div", { class: "panel error" },
    el("strong", {}, `Error: ${err.message}`),
    err.data && err.data.trace_id ? el("div", { class: "small mono" }, `trace ${err.data.trace_id}`) : null,
    details ? el("pre", {}, JSON.stringify(details, null, 2)) : null));
}

// ---------------------------------------------------------------- router
const routes = {
  search: renderSearch, documents: renderDocuments, document: renderDocument, chunk: renderChunk,
  crawl: renderCrawl, catalog: renderCatalog, dataset: renderDataset, term: renderTerm, entity: renderEntity,
  kg: renderKnowledgeGraph,
  definitions: renderDefinitions, definition: renderDefinition, jobs: renderJobs, job: renderJob,
  settings: renderSettings,
};
let refreshTimer = null;
let navSeq = 0;  // incremented on every navigation; a render that awaited data checks it is still the current one
function navGuard() { const seq = navSeq; return () => seq !== navSeq; }
// Catalog pages share one layout whose left facet stays in place while the right side changes.
const CATALOG_ROUTES = new Set(["catalog", "dataset", "kg"]);
function route() {
  clearTimeout(refreshTimer);
  navSeq += 1;
  closeContextMenu();
  const [name, ...rest] = (location.hash.slice(1) || "search").split("/");
  const arg = rest.map(decodeURIComponent).join("/");
  document.querySelectorAll("header a[data-tab]").forEach(a => a.classList.toggle("active", a.dataset.tab === name ||
    ((name === "dataset" || name === "kg") && a.dataset.tab === "catalog") || (name === "job" && a.dataset.tab === "jobs") ||
    (name === "definition" && a.dataset.tab === "definitions") || (name === "document" && a.dataset.tab === "documents")));
  if (!(CATALOG_ROUTES.has(name) && view.querySelector(".catalog-layout"))) view.replaceChildren();
  (routes[name] || renderSearch)(arg).catch(err => showError(view.querySelector(".catalog-main") || view, err));
}
window.addEventListener("hashchange", route);

async function loadHealth() {
  try {
    const h = await api("GET", "/api/health");
    const c = h.checks;
    document.getElementById("health").textContent =
      `${h.status} · docs index v${fmt(c.document_index.version)} · graph ${c.knowledge_graph.triples} triples · ` +
      `terms ${c.term_generation.available ? "on" : "off"} · agent ${c.agent && c.agent.enabled ? "on" : "off"}` +
      (c.observability && c.observability.online_judge ? " · judge on" : "") +
      (c.observability && c.observability.capture_sensitive_content ? " · ⚠ prompts captured in traces" : "");
  } catch (e) { document.getElementById("health").textContent = "health unavailable"; }
}

// ---------------------------------------------------------------- search (agent harness)
async function renderSearch() {
  const q = el("input", { type: "text", placeholder: "e.g. What is Customer Lifetime Value and which tables contain the data needed to calculate it?" });
  const go = el("button", {}, "Search");
  const htmlOut = el("input", { type: "checkbox", id: "html-output" });
  const note = el("p", { class: "small muted", style: "margin:6px 0 0" }, "");
  let agent = null;
  try { agent = (await api("GET", "/api/health")).checks.agent; } catch (e) { /* shown per search */ }
  note.textContent = agent && agent.enabled
    ? `Agent ${agent.provider}/${agent.model}: planner → tasks (executor loop, max ${agent.max_iterations} calls each) → evaluate / replan → cited answer`
    : `Direct retrieval only: ${(agent && agent.reason) || "agent status unavailable"}`;
  const out = el("div");
  const alerts = el("div");
  try {
    const firing = (await api("GET", "/api/ops/alerts?status=firing&limit=10")).alerts;
    if (firing.length) alerts.append(el("div", { class: "panel" }, el("strong", {}, "Firing alerts: "),
      firing.map(a => el("span", {}, badge(a.alertname, a.severity === "critical" ? "bad" : "warn"), " ", a.summary || "", " "))));
  } catch (e) { /* alerts are optional */ }
  const run = async () => {
    if (!q.value.trim()) { q.classList.add("invalid"); q.focus(); return; }
    go.disabled = true;
    out.replaceChildren(el("p", { class: "muted" }, agent && agent.enabled
      ? "Planning, running tools and evaluating results… this can take a minute." : "Searching…"));
    try {
      const data = await api("POST", "/api/search", { query: q.value, output_format: htmlOut.checked ? "html" : "text" });
      out.replaceChildren(data.mode === "agent" ? renderAgentResult(data) : renderRetrievalFallback(data));
    } catch (err) { out.replaceChildren(); showError(out, err); }
    finally { go.disabled = false; }
  };
  go.addEventListener("click", run);
  q.addEventListener("keydown", e => { if (e.key === "Enter") run(); });
  q.addEventListener("input", () => q.classList.remove("invalid"));
  view.append(el("h1", {}, "Search"), alerts,
    el("div", { class: "panel" }, el("div", { class: "row", style: "flex-wrap:nowrap" }, q, go),
      el("div", { class: "row", style: "margin-top:6px" }, el("label", { class: "inline", for: "html-output" }, htmlOut,
        " HTML output (the agent renders results with generate_html)")), note), out);
  q.focus();
}

function renderRetrievalFallback(data) {
  const wrap = el("div");
  if (data.agent && data.agent.reason)
    wrap.append(el("div", { class: "panel" }, badge("agent not used", "warn"), " ", data.agent.reason));
  wrap.append(renderEvidence(data));
  return wrap;
}

function citationList(citations) {
  return el("ul", {}, citations.map(c => el("li", {},
    el("strong", {}, `[${c.id}] `), `${(c.type || "").replace(/_/g, " ")} `,
    c.label ? `“${c.label}” ` : "", c.inferred ? badge("inferred", "warn") : null,
    c.ref ? el("a", { href: citationHash(c.ref) }, "source") : null)));
}

function taskStatusBadge(s) {
  return badge(s.replace(/_/g, " "),
    { answered: "ok", partial: "warn", not_found: "bad", max_iterations: "warn", failed: "bad", running: "info" }[s] || "");
}

function renderPlan(plan) {
  return el("div", { class: "panel" },
    el("div", { class: "row" }, el("h2", { style: "margin:0" }, `Plan v${plan.version}`),
      badge(plan.decompose ? "decomposed" : "single intent", "info"), badge(`execution: ${plan.execution}`),
      plan.fallback_reason ? badge("fallback", "warn") : null, plan.replan_reason ? badge("replanned", "warn") : null),
    plan.replan_reason ? el("p", { class: "small" }, `Replanned because: ${plan.replan_reason}`) : null,
    plan.fallback_reason ? el("p", { class: "small" }, plan.fallback_reason) : null,
    el("p", { class: "small" }, el("span", { class: "muted" }, "Original query: "), plan.original_query),
    el("p", { class: "small muted" }, plan.reasoning),
    el("ol", {}, plan.intents.map(i => el("li", {}, el("strong", {}, `${i.id} `), i.intent,
      i.depends_on && i.depends_on.length ? el("span", { class: "small muted" }, ` · depends on ${i.depends_on.join(", ")}`) : null,
      el("div", { class: "small mono muted" }, (i.retrieval_queries || []).map(q => `“${q}”`).join("  ")))))
  );
}

function renderDecisions(decisions) {
  const label = d => d.after && d.after.length ? `after ${d.after.join(", ")} → ${d.decision}` : d.decision;
  return el("div", { class: "panel" }, el("h2", {}, "Orchestration"),
    el("ol", {}, decisions.map(d => el("li", {},
      badge(d.decision, { next: "ok", replan: "warn", finish: "info" }[d.decision] || ""), " ", label(d),
      d.reason ? el("span", { class: "small muted" }, ` — ${d.reason}`) : null))));
}

function renderTask(t, open) {
  const rows = [];
  for (const s of t.steps) {
    const calls = s.tool_calls || [];
    if (!calls.length) {
      rows.push(el("tr", {}, el("td", {}, String(s.iteration)), el("td", {}, s.evaluation || "—"),
        el("td", {}, badge(s.action, s.action === "complete" ? "ok" : "bad")),
        el("td", {}, s.error ? el("span", { class: "error" }, s.error.message)
          : (s.cited && s.cited.length ? el("span", { class: "small muted" }, `cites ${s.cited.join(", ")}`) : "—"))));
      continue;
    }
    calls.forEach((c, n) => rows.push(el("tr", {},
      el("td", {}, n === 0 ? String(s.iteration) : ""),
      el("td", {}, n === 0 ? (s.evaluation || "—") : ""),
      el("td", {}, el("strong", {}, c.tool), el("div", { class: "mono small" }, JSON.stringify(c.arguments)),
        c.note ? el("div", { class: "small muted" }, c.note) : null),
      el("td", {}, c.status === "error" ? el("span", { class: "error" }, c.summary) : c.summary,
        c.evidence_ids && c.evidence_ids.length ? el("div", { class: "small muted" }, c.evidence_ids.join(", ")) : null))));
  }
  const table = el("table", {}, el("tr", {}, el("th", {}, "#"), el("th", {}, "Evaluation"), el("th", {}, "Tool call"), el("th", {}, "Result")), rows);
  return el("details", { class: "panel", open: open ? true : null },
    el("summary", {}, el("strong", {}, `${t.id} · ${t.intent} `), taskStatusBadge(t.status),
      el("span", { class: "small muted" }, ` attempt ${t.attempt} · plan v${t.plan_version} · ${t.iterations} iteration${t.iterations === 1 ? "" : "s"}`)),
    el("div", { class: "small mono muted" }, (t.retrieval_queries || []).map(q => `“${q}”`).join("  ")),
    t.error ? el("p", { class: "error" }, t.error) : null,
    t.answer ? el("div", { style: "margin:8px 0" }, el("h3", {}, "Task result"), el("div", { class: "answer" }, t.answer)) : null,
    el("h3", {}, "Executor steps"), t.steps.length ? table : el("p", { class: "muted" }, "No steps."));
}

function renderFeedback(traceId) {
  const box = el("div", { class: "row", style: "margin-top:8px" });
  const comment = el("input", { type: "text", placeholder: "Optional comment", style: "max-width:360px" });
  const send = async (rating) => {
    try {
      await api("POST", `/api/searches/${traceId}/feedback`, { rating, comment: comment.value || undefined });
      box.replaceChildren(el("span", { class: "small muted" }, `Thanks: feedback recorded (${rating === "up" ? "👍" : "👎"}).`));
    } catch (err) { showError(box, err); }
  };
  box.append(el("span", { class: "small muted" }, "Was this answer useful?"),
    el("button", { class: "secondary", onclick: () => send("up") }, "👍"),
    el("button", { class: "secondary", onclick: () => send("down") }, "👎"), comment);
  return box;
}

function scoreBadge(v) {
  if (v === null || v === undefined) return badge("—");
  return badge(`${Math.round(v * 100)}%`, v >= 0.8 ? "ok" : v >= 0.5 ? "warn" : "bad");
}

function renderEvaluation(data) {
  const ev = data.evaluation;
  const status = ev.judge ? ev.judge.status : "";
  const judge = el("div", {}, el("p", { class: "small muted" }, status === "pending" ? "LLM judge running…"
    : status === "disabled" ? "LLM judge disabled (observability.online_judge in config/mesh.yaml)." : ""));
  const panel = el("div", { class: "panel" },
    el("div", { class: "row" }, el("h2", { style: "margin:0" }, "Evaluation"),
      el("span", { class: "small muted" }, "rule compliance"), scoreBadge(ev.compliance),
      el("span", { class: "small muted" }, "citation coverage"), scoreBadge(ev.citation_coverage),
      (ev.violations || []).map(v => badge(v.replace(/_/g, " "), "bad"))),
    el("table", {}, el("tr", {}, el("th", {}, "Check"), el("th", {}, "Result"), el("th", {}, "Value"), el("th", {}, "Detail")),
      ev.checks.map(c => el("tr", {}, el("td", {}, c.name.replace(/_/g, " ")),
        el("td", {}, badge(c.passed ? "pass" : "fail", c.passed ? "ok" : "bad")),
        el("td", { class: "mono small" }, fmt(c.value)), el("td", { class: "small muted" }, fmt(c.detail))))),
    el("h3", {}, "LLM judge"), judge);
  if (status === "pending") setTimeout(() => pollJudge(data.trace_id, judge, 0), 1500);
  return panel;
}

async function pollJudge(traceId, box, n) {
  if (n > 60 || !document.body.contains(box)) return;  // gave up, or the user left the page
  let rec;
  try { rec = await api("GET", `/api/searches/${traceId}`); } catch (e) { return; }
  const status = rec.judge_status || "";
  if (status === "pending") { setTimeout(() => pollJudge(traceId, box, n + 1), 2000); return; }
  if (!status.startsWith("done")) { box.replaceChildren(el("p", { class: "error" }, `LLM judge: ${status}`)); return; }
  const j = rec.judge;
  const names = ["faithfulness", "answer_relevance", "completeness", "context_relevance", "plan_adherence"];
  box.replaceChildren(
    el("div", { class: "row" }, names.filter(k => j[k]).map(k => el("span", {},
      el("span", { class: "small muted" }, k.replace(/_/g, " ") + " "), scoreBadge(j[k].score), " "))),
    j.hallucination && j.hallucination.label === "hallucinated"
      ? el("div", {}, badge("unsupported claims", "bad"), el("pre", {}, j.hallucination.explanation || ""))
      : badge("no unsupported claims", "ok"),
    j.faithfulness && j.faithfulness.explanation ? el("p", { class: "small" }, j.faithfulness.explanation) : null);
}

// ---------------------------------------------------------------- HTML output (generate_html)
const HTML_TAGS = new Set(["DIV", "H2", "H3", "H4", "P", "UL", "OL", "LI", "TABLE", "CAPTION", "THEAD", "TBODY", "TR", "TH", "TD", "SPAN"]);
const HTML_CLASSES = new Set(["km-html", "km-citations", "km-note", "km-grid", "km-sensitive"]);
const HTML_DROP = new Set(["SCRIPT", "STYLE", "IFRAME", "OBJECT", "EMBED", "TEMPLATE", "NOSCRIPT", "LINK", "META", "SVG", "MATH"]);
// Rebuild a server fragment with an allowlist: known tags only, no attributes but known classes, text as text.
function sanitizeFragment(html) {
  const doc = new DOMParser().parseFromString(String(html || ""), "text/html");  // parsed inert: scripts never run
  const copy = node => {
    if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.textContent);
    if (node.nodeType !== Node.ELEMENT_NODE || HTML_DROP.has(node.tagName)) return null;
    const kids = Array.from(node.childNodes).map(copy).filter(Boolean);
    if (!HTML_TAGS.has(node.tagName)) { const f = document.createDocumentFragment(); f.append(...kids); return f; }
    const out = document.createElement(node.tagName.toLowerCase());
    const cls = (node.getAttribute("class") || "").split(/\s+/).filter(c => HTML_CLASSES.has(c));
    if (cls.length) out.className = cls.join(" ");
    out.append(...kids);
    return out;
  };
  const frag = document.createDocumentFragment();
  for (const n of Array.from(doc.body.childNodes)) { const c = copy(n); if (c) frag.append(c); }
  return frag;
}

function renderHtmlOutput(data) {
  if (!data.html || !data.html.length) return null;
  return el("div", { class: "panel" }, el("h2", {}, "HTML output"),
    data.html.map(h => el("div", { class: "html-output" }, el("div", { class: "small muted" }, `[${h.id}]`,
      h.truncated ? badge("truncated", "warn") : null), sanitizeFragment(h.html))));
}

function renderClarification(data) {
  const c = data.clarification;
  if (!c) return null;
  const reply = el("textarea", { placeholder: "Your answer", rows: 3 });
  const send = el("button", {}, "Continue");
  const out = el("div");
  const box = el("div", { class: "panel clarify" },
    el("h2", {}, "More information needed"),
    el("p", {}, c.question),
    el("p", { class: "small muted" }, `Question ${c.round} of at most ${c.max_rounds}. The original request and the results so far are kept` +
      (c.expires_in_hours ? ` for ${c.expires_in_hours} h.` : ".")),
    reply, el("div", { class: "row", style: "margin-top:8px" }, send), out);
  send.addEventListener("click", async () => {
    if (!reply.value.trim()) { reply.classList.add("invalid"); reply.focus(); return; }
    send.disabled = true;
    out.replaceChildren(el("p", { class: "muted" }, "Continuing the search with your answer…"));
    try {
      const next = await api("POST", `/api/searches/${encodeURIComponent(data.trace_id)}/clarification`, { reply: reply.value });
      reply.disabled = true;
      out.replaceChildren(el("h2", {}, "Continued search"), renderAgentResult(next));
    } catch (err) { out.replaceChildren(); showError(out, err); send.disabled = false; }
  });
  return box;
}

function renderAgentResult(data) {
  const wrap = el("div");
  if (data.continues) wrap.append(el("div", { class: "panel small" }, badge("continued", "info"),
    ` Continues search ${data.continues} with: `, (data.clarifications || []).filter(c => c.answer).map(c => `“${c.answer}” `)));
  wrap.append(el("div", { class: "panel" },
    el("h2", {}, data.answer ? "Answer" : "No answer"),
    data.answer ? el("div", { class: "answer" }, data.answer) : el("p", { class: "error" }, fmt(data.error)),
    data.citations && data.citations.length ? el("div", { style: "margin-top:10px" },
      el("h3", {}, "Citations"), citationList(data.citations)) : null,
    data.dropped_citations && data.dropped_citations.length
      ? el("p", { class: "small muted" }, `Removed unsupported citations: ${data.dropped_citations.join(", ")}`) : null,
    el("div", { class: "small mono muted" },
      `${data.usage.llm_calls} model calls · ${data.usage.tool_calls} tool calls · ` +
      `${data.usage.input_tokens + data.usage.output_tokens} tokens · ` + "$" + (data.usage.cost_usd || 0).toFixed(4) +
      ` · ${(data.usage.latency_ms / 1000).toFixed(1)} s · trace ${fmt(data.trace_id)} `,
      data.observability && data.observability.trace_url
        ? el("a", { href: data.observability.trace_url, target: "_blank", rel: "noopener" }, "open trace in Phoenix") : null),
    renderFeedback(data.trace_id)));
  const clarify = renderClarification(data);
  if (clarify) wrap.append(clarify);
  const htmlPanel = renderHtmlOutput(data);
  if (htmlPanel) wrap.append(htmlPanel);
  if (data.evaluation) wrap.append(renderEvaluation(data));
  for (const plan of data.plans || [data.plan]) wrap.append(renderPlan(plan));
  if (data.decisions && data.decisions.length) wrap.append(renderDecisions(data.decisions));
  const tasks = data.tasks || [];
  for (const t of tasks) wrap.append(renderTask(t, tasks.length === 1));

  const ev = el("div", { class: "panel" }, el("h2", {}, `Evidence gathered (${data.evidence.length})`));
  if (!data.evidence.length) ev.append(el("p", { class: "muted" }, "No evidence was found."));
  for (const e of data.evidence) {
    ev.append(el("div", { class: "result" },
      el("div", { class: "row" }, el("strong", {}, `[${e.id}]`), badge((e.type || "").replace(/_/g, " "), "info"),
        e.inferred ? badge("inferred", "warn") : null,
        e.ref ? el("a", { href: citationHash(e.ref) }, e.label || e.catalog_ref || e.ref) : (e.label || "")),
      e.excerpt ? el("div", { class: "small muted excerpt" }, e.excerpt) : null));
  }
  wrap.append(ev);
  return wrap;
}

function citationHash(ref) {
  if (!ref) return "#";
  const m = ref.match(/^\/api\/(chunks|datasets|business-definitions|business-terms|business-entities)\/(.+)$/);
  if (!m) return ref;
  const map = { chunks: "chunk", datasets: "dataset", "business-definitions": "definition",
                "business-terms": "term", "business-entities": "entity" };
  return `#${map[m[1]]}/${m[2]}`;
}

function renderEvidence(data) {
  const docs = el("div", { class: "panel" }, el("h2", {}, `Document passages (${data.results.length})`));
  if (!data.results.length) docs.append(el("p", { class: "muted" }, "No relevant passages."));
  for (const r of data.results) {
    const loc = [r.page ? `page ${r.page}` : null, r.section].filter(Boolean).join(" · ");
    const original = r.citation.original_ref + (r.page ? `#page=${r.page}` : "");
    docs.append(el("div", { class: "result" },
      el("div", { class: "row" }, el("strong", {}, `#${r.rank} ${r.title || r.source_uri}`),
        badge(`similarity ${r.score}`, "info"), loc ? el("span", { class: "small muted" }, loc) : null,
        (r.flags || []).map(f => badge(f.replace(/_/g, " "), "bad"))),
      el("div", { class: "excerpt" }, r.text),
      (r.parents || []).map(p => el("details", {},
        el("summary", { class: "small" }, `Parent · level ${p.level} `, el("span", { class: "mono muted" }, p.node_id)),
        p.expanded ? el("div", { class: "excerpt small" }, p.text)
          : el("p", { class: "small muted" }, `not expanded: ${p.reason}`))),
      el("div", { class: "small" }, el("a", { href: `#chunk/${r.chunk_id}` }, "passage"), " · ",
        el("a", { href: original, target: "_blank", rel: "noopener" }, "original"), " · ",
        el("a", { href: `#document/${r.document_id}` }, "document"), " · ",
        el("span", { class: "mono muted" }, r.chunk_id))));
  }
  const gl = el("div", { class: "panel" }, el("h2", {}, `Glossary matches (${data.glossary_results.length})`));
  if (!data.glossary_results.length) gl.append(el("p", { class: "muted" }, "No glossary matches."));
  for (const g of data.glossary_results) {
    gl.append(el("div", { class: "result" },
      el("div", { class: "row" }, el("a", { href: citationHash(g.ref) }, el("strong", {}, g.label)),
        badge(g.type.replace(/_/g, " ")), badge(`similarity ${g.score}`, "info")),
      g.description ? el("div", {}, g.description) : null,
      el("div", { class: "small" }, el("a", { href: `#dataset/${g.dataset_id}` }, g.catalog_ref))));
  }
  const kg = el("div", { class: "panel" }, el("h2", {}, `Graph evidence (${data.graph_evidence.length})`));
  if (!data.graph_evidence.length) kg.append(el("p", { class: "muted" }, "No graph evidence."));
  for (const k of data.graph_evidence) {
    kg.append(el("div", { class: "result" },
      el("div", { class: "row" }, badge(k.type.replace(/_/g, " "), "info"),
        k.inferred ? badge("inferred", "warn") : badge("declared/captured", "ok")),
      el("div", {}, k.text),
      el("div", { class: "small" }, k.source_ref ? el("a", { href: citationHash(k.source_ref) }, "provenance") : null,
        " ", el("span", { class: "mono muted" }, k.graph_ref))));
  }
  const wrap = el("div");
  if (data.reranking) wrap.append(renderReranking(data));
  if (data.insufficient_evidence) wrap.append(el("div", { class: "panel" }, el("strong", {}, "Insufficient evidence. "), data.message));
  wrap.append(el("div", { class: "grid3" }, docs, gl, kg),
    el("div", { class: "small mono muted" }, `trace ${fmt(data.trace_id)}`));
  return wrap;
}

function renderReranking(data) {
  const rr = data.reranking;
  if (!rr.applied) {
    return el("div", { class: "panel" }, badge("Laya reranking unavailable", "warn"),
      el("span", { class: "small muted" }, ` ${rr.error || ""} — passages are in similarity order.`));
  }
  const resp = data.response || { found: false, text: rr.message };
  const head = el("div", { class: "row" }, el("h2", { style: "margin:0" }, "Response"),
    resp.found ? badge("selected by Laya", "ok") : badge("no acceptable response", "bad"),
    el("span", { class: "small muted" }, `${rr.calls} Laya call(s) · ${rr.latency_ms} ms · ` +
      `probability > ${rr.min_probability}, top ${rr.top_n}, quality ≥ ${rr.min_quality}`));
  const rows = rr.decisions.map(d => el("tr", {},
    el("td", {}, el("a", { href: `#chunk/${d.chunk_id}` }, `#${d.retrieval_rank}`)),
    el("td", {}, d.choice_probability.toFixed(3), " ", d.passed_probability ? badge("passed", "ok") : badge("below threshold")),
    el("td", {}, d.quality ? [badge(d.quality, d.acceptable ? "ok" : "bad"), el("span", { class: "small muted" }, ` p=${d.quality_probability}, score ${d.quality_score}`)] : "—"),
    el("td", {}, d.selected ? badge("selected", "info") : "")));
  return el("div", { class: "panel" }, head,
    el("div", { class: "answer", style: "margin:10px 0" }, resp.text),
    resp.found ? el("div", { class: "small" }, el("a", { href: `#chunk/${resp.chunk_id}` }, "passage"), " · ",
      el("a", { href: `#document/${resp.document_id}` }, "document")) : null,
    rr.decisions.length ? el("details", {}, el("summary", { class: "small" }, "Laya decisions"),
      el("table", {}, el("tr", {}, ["Retrieved", "Choice probability", "Quality", ""].map(h => el("th", {}, h))), rows)) : null);
}

// ---------------------------------------------------------------- documents
async function renderDocuments() {
  const file = el("input", { type: "file", accept: ".pdf,.docx,.txt,.md,.html,.htm" });
  const idx = el("input", { type: "checkbox", checked: true });
  const list = el("div");
  const upload = async () => {
    if (!file.files.length) return toast("Choose a file first", true);
    const fd = new FormData(); fd.append("file", file.files[0]); fd.append("index", idx.checked ? "true" : "false");
    try {
      const r = await api("POST", "/api/documents", fd, true);
      toast(`${r.status}: ${r.document_id}${r.job_id ? " (indexing " + r.job_id + ")" : ""}`);
      setTimeout(load, 800);
    } catch (err) { toast(err.message, true); }
  };
  const ingestDir = async () => {
    try { const r = await api("POST", "/api/ingest-directory", {}); location.hash = `#job/${r.job_id}`; }
    catch (err) { toast(err.message, true); }
  };
  async function load() {
    const data = await api("GET", "/api/documents?limit=200");
    const rows = data.documents.map(d => el("tr", {},
      el("td", {}, el("a", { href: `#document/${d.document_id}` }, d.title || d.filename || d.document_id)),
      el("td", {}, d.source_type), el("td", {}, statusBadge(d.status), d.status_reason ? el("div", { class: "small muted" }, d.status_reason) : null),
      el("td", {}, fmt(d.leaf_count)), el("td", { class: "small" }, fmt(d.updated_at)),
      el("td", {}, el("button", { class: "danger", onclick: async () => {
        if (!confirm("Delete this document and its indexed chunks?")) return;
        await api("DELETE", `/api/documents/${d.document_id}`); load(); } }, "Delete"))));
    list.replaceChildren(el("table", {}, el("tr", {}, ["Document", "Source", "Status", "Leaf chunks", "Updated", ""].map(h => el("th", {}, h))), rows),
      el("p", { class: "small muted" }, `${data.total} documents`));
  }
  view.append(el("h1", {}, "Documents"),
    el("div", { class: "panel" }, el("h2", {}, "Upload"), el("div", { class: "row" }, file,
      el("label", { style: "margin:0" }, idx, " index after upload"), el("button", { onclick: upload }, "Upload")),
      el("p", { class: "small muted" }, "PDF, DOCX, TXT, Markdown or HTML. Size and type limits are enforced by the server."),
      el("button", { class: "secondary", onclick: ingestDir }, "Ingest configured inbox directory")),
    el("div", { class: "panel" }, el("div", { class: "row" }, el("h2", { style: "margin:0" }, "Ingested"),
      el("button", { class: "secondary", onclick: load }, "Refresh")), list));
  await load();
}

async function renderDocument(id) {
  const d = await api("GET", `/api/documents/${encodeURIComponent(id)}`);
  const text = await api("GET", `/api/documents/${encodeURIComponent(id)}/text?max_chars=6000`).catch(() => null);
  view.append(el("h1", {}, d.title || d.filename || d.document_id),
    el("div", { class: "panel" }, el("table", {}, Object.entries({
      "Document ID": d.document_id, "Source": d.source_uri, "Type": d.mime_type, "Status": d.status,
      "Reason": d.status_reason, "Content hash": d.content_hash, "Pages": d.page_count, "Sections": d.section_count,
      "Nodes / leaf chunks": `${d.node_count} / ${d.leaf_count}`, "Index version": d.index_version,
      "Ingested": d.ingested_at, "Indexed": d.indexed_at }).map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", { class: "mono" }, fmt(v))))),
      d.original_ref ? el("p", {}, el("a", { href: d.original_ref, target: "_blank", rel: "noopener" }, "Open original")) : null),
    text ? el("div", { class: "panel" }, el("h2", {}, "Extracted text (first 6,000 characters)"),
      text.available_sections.length ? el("p", { class: "small muted" }, "Sections: " + text.available_sections.join(" · ")) : null,
      el("div", { class: "excerpt" }, text.text)) : null);
}

async function renderChunk(id) {
  const c = await api("GET", `/api/chunks/${encodeURIComponent(id)}`);
  view.append(el("h1", {}, "Passage"),
    el("div", { class: "panel" },
      el("div", { class: "row" }, el("a", { href: `#document/${c.document_id}` }, c.title || c.document_id),
        c.page ? badge(`page ${c.page}`) : null, c.section ? badge(c.section) : null, badge(`level ${c.level}`)),
      el("div", { class: "excerpt" }, c.text),
      el("p", { class: "small" }, el("a", { href: c.original_ref + (c.page ? `#page=${c.page}` : ""), target: "_blank", rel: "noopener" }, "Open original"),
        " · ", el("span", { class: "mono muted" }, c.source_uri))),
    c.parents.map(p => el("div", { class: "panel" }, el("h3", {}, `Parent (level ${p.level}) `, el("span", { class: "mono muted" }, p.node_id)),
      el("div", { class: "excerpt small" }, p.text))));
}

// ---------------------------------------------------------------- crawl
async function renderCrawl() {
  const url = el("input", { type: "url", placeholder: "https://example.com/" });
  const depth = el("input", { type: "number", min: 1, max: 10, value: 3, style: "width:80px" });
  const pages = el("input", { type: "number", min: 1, max: 10000, value: 100, style: "width:100px" });
  const submit = async () => {
    try {
      const r = await api("POST", "/api/crawl", { url: url.value, max_depth: Number(depth.value), max_pages: Number(pages.value) });
      location.hash = `#job/${r.job_id}`;
    } catch (err) { toast(err.message, true); }
  };
  view.append(el("h1", {}, "Crawl a website"), el("div", { class: "panel" },
    el("label", {}, "Start URL (crawl stays on this hostname unless other hosts are allowlisted)"), url,
    el("div", { class: "row" }, el("label", {}, "Max depth (1 = this page only)"), depth, el("label", {}, "Max pages"), pages,
      el("button", { onclick: submit }, "Start crawl")),
    el("p", { class: "small muted" }, "robots.txt is honored; private, loopback and metadata addresses are blocked.")));
}

// ---------------------------------------------------------------- catalog
// The catalog pages share one layout: a left facet, always visible, with Databases (one per configured source) ->
// Schemas -> Tables, and the clicked element on the right:
//   #catalog                            overview of the sources
//   #catalog/db/<source>                a database: its description and its schemas in a grid
//   #catalog/schema/<source>/<schema>   a schema: its tables and their business entity in a grid
//   #dataset/<dataset_id>               a table: its columns in a grid, then its keys and relationships
//   #kg/<source|schema|table>/...       the knowledge graph of a database, schema or table
// Right-click (or the context-menu key / Shift+F10 on a focused item) opens the knowledge graph of that element.
let catalogTree = null;  // { layout, items: Map(key -> tree item), sources: Map, datasets: Map } of the shown facet

function hashParts() {
  return location.hash.slice(1).split("/").map(p => { try { return decodeURIComponent(p); } catch (e) { return p; } });
}
function catalogHash(kind, ...parts) { return "#catalog/" + [kind, ...parts].map(p => encodeURIComponent(p ?? "")).join("/"); }
function kgHash(kind, sourceId, extra) {
  return "#kg/" + [kind, sourceId, extra].filter(x => x !== undefined && x !== null).map(encodeURIComponent).join("/");
}
function showContextMenu(x, y, items) {
  closeContextMenu();
  const menu = el("div", { class: "ctx-menu", role: "menu" }, items.map(it => el("button", {
    class: "ctx-item", role: "menuitem", onclick: () => { closeContextMenu(); it.action(); } }, it.label)));
  menu.style.left = `${Math.max(4, Math.min(x, window.innerWidth - 240))}px`;
  menu.style.top = `${Math.max(4, Math.min(y, window.innerHeight - 60))}px`;
  document.body.append(menu);
  menu.querySelector("button").focus();
  setTimeout(() => document.addEventListener("click", closeContextMenu, { once: true }), 0);
}
function closeContextMenu() { document.querySelectorAll(".ctx-menu").forEach(m => m.remove()); }
document.addEventListener("keydown", e => { if (e.key === "Escape") closeContextMenu(); });

// A grid: bordered table with a header row. rows: arrays of cell contents, or ready <tr> elements.
function grid(headers, rows, label) {
  return el("div", { class: "grid-wrap" }, el("table", { class: "grid", "aria-label": label || null },
    el("thead", {}, el("tr", {}, headers.map(h => el("th", { scope: "col" }, h)))),
    el("tbody", {}, rows.map(r => r instanceof Node ? r : el("tr", {}, r.map(c => el("td", {}, c)))))));
}

function treeItem({ label, sub, href, kg, kgLabel, children, open }) {
  const caret = children ? el("button", { class: "caret", "aria-label": `expand ${kgLabel}`, "aria-expanded": open ? "true" : "false" }, open ? "▾" : "▸") : el("span", { class: "caret-space" });
  const name = el("a", { href, class: "tree-label", title: "Click: show on the right · right-click: knowledge graph" }, label);
  const menu = e => {
    e.preventDefault();
    const r = name.getBoundingClientRect();
    showContextMenu(e.clientX || r.left, e.clientY || r.bottom, [{ label: `Knowledge graph of ${kgLabel}`, action: () => { location.hash = kg; } }]);
  };
  name.addEventListener("contextmenu", menu);
  name.addEventListener("keydown", e => { if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) menu(e); });
  const li = el("li", { role: "treeitem", "aria-selected": "false", "aria-expanded": children ? (open ? "true" : "false") : null },
    el("div", { class: "tree-row" }, caret, name, sub ? el("span", { class: "small muted" }, ` ${sub}`) : null));
  if (children) {
    const ul = el("ul", { role: "group" }, children);
    ul.hidden = !open;
    li.setOpen = isOpen => {
      ul.hidden = !isOpen;
      caret.textContent = isOpen ? "▾" : "▸";
      caret.setAttribute("aria-expanded", String(isOpen));
      li.setAttribute("aria-expanded", String(isOpen));
    };
    caret.addEventListener("click", () => li.setOpen(ul.hidden));
    li.append(ul);
  }
  return li;
}

function catalogFacet(sources) {
  const items = new Map();
  const add = (key, li) => { items.set(JSON.stringify(key), li); return li; };
  const db = s => (s.settings && (s.settings.database || s.settings.catalog)) || (s.kind === "parquet" ? "parquet" : null);
  const tree = el("ul", { class: "facet-tree", role: "tree", "aria-label": "Databases, schemas and tables" },
    sources.map(s => add(["db", s.source_id], treeItem({
      label: db(s) ? `${s.source_id} (${db(s)})` : s.source_id, sub: `${s.dataset_count}`, href: catalogHash("db", s.source_id),
      kg: kgHash("source", s.source_id), kgLabel: `database ${s.source_id}`, open: true,
      children: s.schemas.map(sc => add(["schema", s.source_id, sc.schema || ""], treeItem({
        label: sc.schema || "(default)", sub: `${sc.datasets.length}`, href: catalogHash("schema", s.source_id, sc.schema || ""),
        kg: kgHash("schema", s.source_id, sc.schema || ""), kgLabel: `schema ${sc.schema || "(default)"}`, open: sources.length === 1,
        children: sc.datasets.map(d => add(["table", d.dataset_id], treeItem({
          label: d.name, href: `#dataset/${d.dataset_id}`, kg: kgHash("table", s.source_id, d.dataset_id), kgLabel: `table ${d.name}` }))) }))) }))));
  const facet = el("aside", { class: "facet panel", "aria-label": "Catalog navigation" },
    el("h2", { style: "margin-top:0" }, el("a", { href: "#catalog", class: "facet-home" }, "Databases")),
    sources.length ? tree : el("p", { class: "muted small" }, "No sources."),
    el("p", { class: "small muted" }, "Click a database, schema or table to show it on the right; right-click it for its knowledge graph."));
  return { facet, items };
}

// Builds the layout and facet once, then on each catalog navigation only marks the selection and returns a fresh
// right-hand pane. Returns null when the user navigated elsewhere while the catalog was loading.
async function catalogShell(selection) {
  const isStale = navGuard();
  let layout = view.querySelector(".catalog-layout");
  if (!layout || !catalogTree || catalogTree.layout !== layout) {
    const data = await api("GET", "/api/catalog");
    if (isStale()) return null;
    const { facet, items } = catalogFacet(data.sources);
    layout = el("div", { class: "catalog-layout" }, facet, el("div", { class: "catalog-main" }));
    catalogTree = { layout, items, sources: new Map(data.sources.map(s => [s.source_id, s])),
      datasets: new Map(data.sources.flatMap(s => s.schemas.flatMap(sc => sc.datasets.map(d =>
        [d.dataset_id, { ...d, source_id: s.source_id, schema: sc.schema }])))) };
    view.replaceChildren(el("h1", {}, "Structured catalog"), layout);
  }
  selectTreeItem(selection);
  const pane = el("div", { class: "catalog-pane" });
  layout.querySelector(".catalog-main").replaceChildren(pane);
  if (layout.getBoundingClientRect().top < 0) window.scrollTo({ top: 0 });
  return pane;
}
function selectTreeItem(key) {
  for (const li of catalogTree.layout.querySelectorAll('li[aria-selected="true"]')) {
    li.setAttribute("aria-selected", "false");
    const label = li.querySelector(":scope > .tree-row > .tree-label");
    label.classList.remove("selected"); label.removeAttribute("aria-current");
  }
  const li = key && catalogTree.items.get(JSON.stringify(key));
  if (!li) return;
  li.setAttribute("aria-selected", "true");
  const label = li.querySelector(":scope > .tree-row > .tree-label");
  label.classList.add("selected"); label.setAttribute("aria-current", "page");
  for (let p = li.parentElement.closest("li[role=treeitem]"); p; p = p.parentElement.closest("li[role=treeitem]")) {
    if (p.setOpen) p.setOpen(true);
  }
  label.scrollIntoView({ block: "nearest" });
}

function sourceHeader(s) {
  const start = async () => {
    try { const r = await api("POST", `/api/structured-sources/${s.source_id}/catalog`, {}); location.hash = `#job/${r.job_id}`; }
    catch (err) { toast(err.message, true); }
  };
  return [el("div", { class: "row" }, el("h2", { style: "margin:0" }, s.source_id), badge(s.kind, "info"), statusBadge(s.status || "configured"),
      el("span", { class: "small muted" }, `${s.dataset_count} datasets · last cataloged ${fmt(s.last_extracted_at)}`),
      el("button", { onclick: start }, "Catalog + discover keys + terms"),
      el("a", { href: kgHash("source", s.source_id) }, "knowledge graph")),
    s.status_reason
      ? el("div", { class: s.dataset_count ? "small" : "small error", style: "margin-top:6px" }, s.status_reason)
      : (s.dataset_count ? null : el("div", { class: "small muted", style: "margin-top:6px" },
          s.last_extracted_at ? "The last catalog run found no datasets." : "Not cataloged yet: click Catalog."))];
}

async function renderCatalog() {
  const [, kind, sourceId, schema] = hashParts();
  const key = kind === "db" ? ["db", sourceId] : kind === "schema" ? ["schema", sourceId, schema || ""] : null;
  const pane = await catalogShell(key);
  if (!pane) return;
  if (kind === "db" || kind === "schema") {
    const s = catalogTree.sources.get(sourceId);
    if (!s) { pane.append(el("div", { class: "panel error" }, `Unknown database: ${sourceId}`)); return; }
    if (kind === "db") databasePane(pane, s); else schemaPane(pane, s, schema || "");
    return;
  }
  const sources = [...catalogTree.sources.values()];
  if (!sources.length) pane.append(el("div", { class: "panel muted" }, "No structured sources are configured in config/mesh.yaml."));
  for (const s of sources) {
    pane.append(el("div", { class: "panel tree" }, sourceHeader(s),
      el("ul", {}, s.schemas.map(sc => el("li", {}, el("a", { href: catalogHash("schema", s.source_id, sc.schema || "") }, el("strong", {}, `schema ${sc.schema || "(default)"}`)),
        el("ul", {}, sc.datasets.map(d => el("li", {}, el("a", { href: `#dataset/${d.dataset_id}` }, d.name), " ",
          badge(d.kind), d.entity_label ? badge(`entity: ${d.entity_label}`, "ok") : null,
          el("span", { class: "small muted" }, ` ${fmt(d.column_count)} columns · ${fmt(d.row_count)} rows`)))))))));
  }
}

// Database: its own description, then its schemas with their description (not captured yet, so blank).
function databasePane(pane, s) {
  pane.append(el("div", { class: "panel" }, sourceHeader(s),
    el("p", { class: s.description ? null : "small muted" }, s.description || "No description in the source settings."),
    el("h3", {}, "Schemas"),
    s.schemas.length ? grid(["Schema", "Description"], s.schemas.map(sc => [
        el("a", { href: catalogHash("schema", s.source_id, sc.schema || "") }, sc.schema || "(default)"),
        el("span", { class: "muted", title: "Schema descriptions are not captured yet" }, "—")]), `Schemas of ${s.source_id}`)
      : el("p", { class: "muted" }, "No schemas: catalog this database first.")));
}

// Schema: its tables with the business entity each one represents.
function schemaPane(pane, s, schema) {
  const sc = s.schemas.find(x => (x.schema || "") === schema);
  if (!sc) { pane.append(el("div", { class: "panel error" }, `Unknown schema ${schema || "(default)"} in ${s.source_id}`)); return; }
  pane.append(el("div", { class: "panel" },
    el("div", { class: "row" }, el("h2", { style: "margin:0" }, `Schema ${schema || "(default)"}`),
      el("span", { class: "small muted" }, "database ", el("a", { href: catalogHash("db", s.source_id) }, s.source_id), ` · ${sc.datasets.length} tables`),
      el("a", { href: kgHash("schema", s.source_id, schema) }, "knowledge graph")),
    grid(["Table", "Business entity"], sc.datasets.map(d => [
      el("a", { href: `#dataset/${d.dataset_id}` }, d.name),
      d.entity_label ? (d.entity_slug ? el("a", { href: `#entity/${d.entity_slug}` }, d.entity_label) : d.entity_label) : el("span", { class: "muted" }, "—")]),
      `Tables of schema ${schema || "(default)"}`)));
}

// ---------------------------------------------------------------- knowledge graph view (cytoscape.js, vendored)
let cytoscapeLoading = null;
function loadCytoscape() {
  if (window.cytoscape) return Promise.resolve(window.cytoscape);
  if (!cytoscapeLoading) cytoscapeLoading = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/static/vendor/cytoscape.min.js";
    s.onload = () => resolve(window.cytoscape);
    s.onerror = () => { cytoscapeLoading = null; reject(new Error("could not load the graph library")); };
    document.head.append(s);
  });
  return cytoscapeLoading;
}
const KG_COLORS = { table: "#2f5bd3", column: "#7b8496", primary_key: "#1f8a4c", unique_key: "#4c9f70", foreign_key: "#b26a00",
                    entity: "#8e44ad", term: "#c0367a" };
const KG_SHAPES = { table: "round-rectangle", column: "ellipse", primary_key: "diamond", unique_key: "diamond", foreign_key: "diamond",
                    entity: "hexagon", term: "round-tag" };

async function renderKnowledgeGraph() {
  const [, kind, sourceId, extra] = hashParts();
  const pane = await catalogShell(kind === "table" ? ["table", extra] : kind === "schema" ? ["schema", sourceId, extra || ""] : ["db", sourceId]);
  if (!pane) return;
  const params = new URLSearchParams({ source_id: sourceId || "" });
  if (kind === "schema") params.set("schema", extra || "");
  if (kind === "table") params.set("dataset_id", extra || "");
  const ds = kind === "table" ? catalogTree.datasets.get(extra) : null;
  const title = kind === "table" ? `table ${ds ? ds.name : extra}` : kind === "schema" ? `schema ${extra || "(default)"}` : `database ${sourceId}`;
  const back = kind === "table" ? `#dataset/${extra}` : kind === "schema" ? catalogHash("schema", sourceId, extra || "") : catalogHash("db", sourceId);
  pane.append(el("div", { class: "row", style: "margin-bottom:10px" }, el("h2", { style: "margin:0" }, `Knowledge graph · ${title}`),
    el("a", { href: back }, `← ${title}`)));
  const data = await api("GET", `/api/graph/view?${params.toString()}`);
  const info = el("div", { class: "small kg-info" }, "Hover a node or edge for details; click a table, entity or term to open it.");
  const canvas = el("div", { class: "kg-canvas", role: "img", "aria-label": `Knowledge graph with ${data.nodes.length} nodes and ${data.edges.length} edges` });
  const legend = el("div", { class: "row small" }, Object.entries(KG_COLORS).map(([k, c]) => {
    const dot = el("span", { class: "legend-dot" }); dot.style.background = c;
    return el("span", {}, dot, k.replace(/_/g, " "));
  }), el("span", { class: "muted" }, "· dashed = inferred · red = rejected"));
  pane.append(el("div", { class: "panel" },
    el("div", { class: "row" }, badge(`${data.nodes.length} nodes`, "info"), badge(`${data.edges.length} edges`, "info"),
      data.truncated ? badge("truncated (knowledge_graph.view_max_nodes)", "warn") : null,
      el("span", { class: "small muted" }, `graph version ${data.graph.version} · ${data.graph.published_at}`)),
    legend, canvas, info));
  // accessible listing of the same content
  pane.append(el("details", { class: "panel" }, el("summary", {}, "Nodes and edges as a list"),
    el("table", {}, el("tr", {}, ["Type", "Node", "Status"].map(h => el("th", {}, h))),
      data.nodes.map(n => el("tr", {}, el("td", {}, n.type.replace(/_/g, " ")),
        el("td", {}, n.ref ? el("a", { href: n.ref }, n.label) : n.label), el("td", {}, fmt(n.status))))),
    el("table", {}, el("tr", {}, ["Edge", "From", "To", "Status"].map(h => el("th", {}, h))),
      data.edges.map(e => el("tr", {}, el("td", {}, `${e.type.replace(/_/g, " ")}: ${e.label}`),
        el("td", { class: "small" }, labelOf(data, e.source)), el("td", { class: "small" }, labelOf(data, e.target)), el("td", {}, fmt(e.status)))))));
  let cytoscape;
  try { cytoscape = await loadCytoscape(); }
  catch (err) { canvas.replaceChildren(el("p", { class: "error" }, err.message)); return; }
  if (!canvas.isConnected) return;  // the user selected something else meanwhile
  const text = getComputedStyle(document.body).getPropertyValue("--text").trim() || "#1d2330";
  const line = getComputedStyle(document.body).getPropertyValue("--muted").trim() || "#5d6677";
  const cy = cytoscape({
    container: canvas,
    elements: [...data.nodes.map(n => ({ data: { ...n } })), ...data.edges.map(e => ({ data: { ...e } }))],
    wheelSensitivity: 0.2,
    style: [
      { selector: "node", style: { label: "data(label)", "font-size": 9, color: text, "text-valign": "bottom", "text-margin-y": 3,
        "background-color": n => KG_COLORS[n.data("type")] || "#888", shape: n => KG_SHAPES[n.data("type")] || "ellipse",
        width: n => n.data("type") === "column" ? 10 : 22, height: n => n.data("type") === "column" ? 10 : 22 } },
      { selector: "edge", style: { width: 1.2, "line-color": line, "target-arrow-color": line, "target-arrow-shape": "triangle",
        "curve-style": "bezier", label: e => ["has_column", "has_key", "has_term"].includes(e.data("type")) ? "" : e.data("label"),
        "font-size": 8, color: text, "text-rotation": "autorotate" } },
      { selector: "edge[status = 'inferred']", style: { "line-style": "dashed" } },
      { selector: "edge[status = 'rejected'], node[status = 'rejected']", style: { "line-color": "#c0362c", "target-arrow-color": "#c0362c" } },
      { selector: "edge[type = 'semantic_relationship']", style: { width: 2.2, "line-color": "#8e44ad", "target-arrow-color": "#8e44ad" } },
      { selector: "edge[type = 'term_relation']", style: { "line-color": "#c0367a", "target-arrow-color": "#c0367a" } },
    ],
    layout: { name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 70, padding: 20 },
  });
  const describe = d => [d.type.replace(/_/g, " ") + ": " + d.label, d.status ? `status ${d.status}` : null,
    d.origin ? `origin ${d.origin}` : null, d.confidence ? `confidence ${d.confidence}` : null, d.data_type ? `type ${d.data_type}` : null,
    d.definition || null, d.synonyms && d.synonyms.length ? `synonyms: ${d.synonyms.join(", ")}` : null].filter(Boolean).join(" · ");
  cy.on("mouseover", "node, edge", ev => { info.textContent = describe(ev.target.data()); });
  cy.on("tap", "node", ev => { const ref = ev.target.data("ref"); if (ref) location.hash = ref; });
}
function labelOf(data, id) { const n = data.nodes.find(x => x.id === id); return n ? n.label : id; }

// Verify / reject inferred knowledge (FKs, relationships, term relations, table->entity and column->term links).
// Business terms that can identify a person (PII / PHI / GDPR), flagged by the term model.
function sensitiveTag(flag) {
  return flag ? el("span", { class: "badge bad", title: "Can identify a person, alone or combined with other information" }, "sensitive") : null;
}
function statusTag(status) {
  return status ? badge(status, { verified: "ok", inferred: "warn", rejected: "bad" }[status] || "") : null;
}
function reviewButtons(objectType, objectId, status, after) {
  const act = async (next) => {
    const note = next === "rejected" ? prompt("Why is it wrong? (optional)") : null;
    try {
      await api("POST", "/api/reviews", { object_type: objectType, object_id: objectId, status: next, note: note || undefined });
      toast(`${objectType.replace(/_/g, " ")} ${next === "inferred" ? "reset to inferred" : next}`);
      after();
    } catch (err) { toast(err.message, true); }
  };
  return el("span", { class: "review" },
    status !== "verified" ? el("button", { class: "secondary tiny", title: "Mark as verified", onclick: () => act("verified") }, "Verify") : null,
    status !== "rejected" ? el("button", { class: "danger tiny", title: "Mark as rejected", onclick: () => act("rejected") }, "Reject") : null,
    status === "verified" || status === "rejected" ? el("button", { class: "secondary tiny", title: "Clear the decision", onclick: () => act("inferred") }, "Reset") : null);
}

async function relationshipsPanel(d, reload) {
  const panel = el("div", { class: "panel" });
  let r;
  try { r = await api("GET", `/api/datasets/${encodeURIComponent(d.dataset_id)}/relationships`); }
  catch (err) { showError(panel, err); return panel; }
  const propose = async () => {
    try { const job = await api("POST", "/api/datasets/generate-relationships", { dataset_ids: [d.dataset_id] }); location.hash = `#job/${job.job_id}`; }
    catch (err) { toast(err.message, true); }
  };
  panel.append(el("div", { class: "row" }, el("h2", { style: "margin:0" }, "Relationships & term relations"),
      el("button", { class: "secondary", onclick: propose }, "Propose relationships"),
      r.last_run ? el("span", { class: "small muted" }, `last proposal run ${r.last_run.created_at} · ${r.last_run.status}`) : null),
    el("p", { class: "small muted" }, "Proposed by the term-generation model for this table's foreign keys and terms. " +
      "They describe how data can be joined, not facts about records; inferred until verified."),
    r.relationships.length ? el("table", {}, el("tr", {}, ["Relationship", "Definition", "Synonyms", "Status", ""].map(h => el("th", {}, h))),
      r.relationships.map(x => el("tr", {},
        el("td", {}, el("strong", {}, `${x.source_entity} ${x.name} ${x.target_entity}`),
          x.inverse_name ? el("div", { class: "small muted" }, `inverse: ${x.inverse_name}`) : null,
          x.cardinality ? el("div", { class: "small muted" }, x.cardinality) : null),
        el("td", { class: "small" }, fmt(x.definition)), el("td", { class: "small" }, x.synonyms.join(", ") || "—"),
        el("td", {}, statusTag(x.status)), el("td", {}, reviewButtons("relationship", x.relationship_id, x.status, reload)))))
      : el("p", { class: "muted small" }, "No proposed relationship."),
    r.term_relations.length ? [el("h3", {}, "Term relations"), el("table", {},
      r.term_relations.map(x => el("tr", {}, el("td", {}, `${x.subject_label} `, el("strong", {}, x.predicate), ` ${x.object_label}`),
        el("td", { class: "small" }, fmt(x.definition)), el("td", {}, statusTag(x.status)),
        el("td", {}, reviewButtons("term_relation", x.relation_id, x.status, reload)))))] : null,
    Object.keys(r.term_synonyms).length ? [el("h3", {}, "Term synonyms (skos:altLabel, inferred)"),
      el("ul", { class: "small" }, Object.entries(r.term_synonyms).map(([slug, syns]) => el("li", {},
        el("a", { href: `#term/${slug}` }, slug), ": ", syns.join(", "))))] : null);
  return panel;
}

async function renderDataset(id) {
  const pane = await catalogShell(["table", id]);
  if (pane) await datasetPane(pane, id);
}

// Table: its columns in a grid, then the table details, column profiles, keys and relationships.
async function datasetPane(pane, id) {
  const reload = () => {
    const fresh = el("div", { class: "catalog-pane" });
    pane.replaceWith(fresh);
    datasetPane(fresh, id).catch(err => showError(fresh, err));
  };
  const d = await api("GET", `/api/datasets/${encodeURIComponent(id)}`);
  const act = async (path, body) => {
    try { const r = await api("POST", path, body); location.hash = `#job/${r.job_id}`; } catch (err) { toast(err.message, true); }
  };
  const termCell = c => c.business_term
    ? [el("a", { href: `#term/${c.business_term.slug}`, title: c.business_term.description || null }, c.business_term.label), " ",
       sensitiveTag(c.business_term.sensitive), statusTag(c.business_term.status),
       reviewButtons("term_mapping", c.column_id, c.business_term.status, reload)]
    : el("span", { class: "small muted" }, c.business_term_error || "—");
  const formats = p => p && p.format_patterns.length
    ? p.format_patterns.slice(0, 3).map(f => el("div", { class: "mono" }, `${f.pattern} (${f.count})`)) : "—";
  const columnGrid = grid(["Column name", "Data type", "Column data type", "Content type", "Data format", "Business term"],
    d.columns.map(c => {
      const p = c.profile;
      return el("tr", {}, el("td", { class: "mono" }, c.name), el("td", {}, fmt(p && p.type_category)),
        el("td", { class: "mono" }, fmt(c.data_type)), el("td", {}, fmt(p && p.content_type)), el("td", {}, formats(p)),
        el("td", {}, termCell(c)));
    }), `Columns of ${d.name}`);
  const profiled = d.columns.filter(c => c.profile);
  const gridNote = !profiled.length
    ? "Data type, content type and data format come from profiling: run Discover keys."
    : profiled.some(c => !c.profile.type_category) ? "Data type is filled in by the next Discover keys run." : null;
  const colProfiles = d.columns.map(c => {
    const p = c.profile;
    return [el("span", { class: "mono" }, c.name), c.nullable === null ? "—" : c.nullable ? "yes" : "no",
      p ? fmt(p.row_count) : "—", p ? fmt(p.distinct_count) : "—", p ? fmt(p.null_count) : "—", p ? fmt(p.empty_count) : "—",
      p && p.most_common_values.length ? el("span", { class: "small" }, p.most_common_values.map(v => `${v.value} (${v.count})`).join(", ")) : "—"];
  });
  // Inferred FKs: confidence from the current business terms (high = same term, medium = same name, low = values only).
  const confBadge = c => c ? el("span", { title: (c.reasons || []).join("\n") },
    badge(`${c.level} confidence`, { high: "ok", medium: "info", low: "warn" }[c.level])) : null;
  const keyRow = k => el("tr", {},
    el("td", {}, badge(k.key_type), k.origin === "declared" ? badge("declared", "ok") : badge("inferred", "warn"),
      k.primary_key_candidate ? badge("PK candidate", "info") : null, confBadge(k.confidence),
      k.key_type === "foreign" && k.origin === "inferred" ? [statusTag(k.status), reviewButtons("foreign_key", k.key_id, k.status, reload)] : null,
      k.confidence ? el("div", { class: "small muted" }, k.confidence.reasons.join("; ")) : null),
    el("td", { class: "mono" }, k.columns.join(", ")),
    el("td", {}, k.key_type === "foreign" ? [el("a", { href: `#dataset/${k.referenced_dataset_id}` }, fmt(k.referenced_dataset)),
      el("div", { class: "small mono" }, (k.mapping || []).map(m => `${m.from}→${m.to}`).join(", "))] : "—"),
    el("td", { class: "small muted" }, k.scope ? `${k.scope.rows_evaluated ?? ""} rows${k.scope.sampled ? " (sample)" : ""} · run ${k.profiling_run_id}` : "source constraint"));
  pane.append(
    el("div", { class: "panel" },
      el("div", { class: "row" }, el("h2", { style: "margin:0" }, d.name), el("span", { class: "mono muted" }, d.catalog_ref)),
      columnGrid, gridNote ? el("p", { class: "small muted", style: "margin:6px 0 0" }, gridNote) : null),
    el("div", { class: "panel" }, el("h3", { style: "margin-top:0" }, "Table details"), el("div", { class: "row" },
      badge(d.kind), d.business_entity ? [el("a", { href: `#entity/${d.business_entity.slug}` }, badge(`entity: ${d.business_entity.label}`, "ok")),
        statusTag(d.business_entity.status), reviewButtons("entity_mapping", d.dataset_id, d.business_entity.status, reload)] : null,
      el("a", { href: kgHash("table", d.source_id, d.dataset_id) }, "knowledge graph"),
      el("span", { class: "small muted" }, `${fmt(d.row_count)} rows · ${d.physical_ref}`)),
      d.definitions.length ? el("p", {}, "Definitions: ", d.definitions.map(x => el("a", { href: `#definition/${x.definition_id}`, style: "margin-right:8px" }, x.name))) : null,
      el("div", { class: "row", style: "margin-top:8px" },
        el("button", { onclick: () => act("/api/datasets/discover-keys", { dataset_ids: [d.dataset_id] }) }, "Discover keys"),
        el("button", { class: "secondary", onclick: () => act("/api/datasets/generate-terms", { dataset_ids: [d.dataset_id], force: true }) }, "Regenerate terms"))),
    el("details", { class: "panel" }, el("summary", {}, "Column profiles"),
      grid(["Column", "Nullable", "Rows", "Distinct", "Nulls", "Empty", "Most common values"], colProfiles, `Column profiles of ${d.name}`)),
    el("div", { class: "panel" }, el("h2", { style: "margin-top:0" }, "Keys"), d.keys.length ? el("table", {},
      el("tr", {}, ["Type", "Columns", "References", "Provenance"].map(h => el("th", {}, h))), d.keys.map(keyRow)) : el("p", { class: "muted" }, "No keys recorded."),
      d.referenced_by.length ? [el("h3", {}, "Referenced by"), el("table", {}, d.referenced_by.map(k => el("tr", {},
        el("td", {}, el("a", { href: `#dataset/${k.dataset_id}` }, k.dataset_id)),
        el("td", { class: "mono" }, (k.mapping || []).map(m => `${m.from}→${m.to}`).join(", ")),
        el("td", {}, k.origin === "declared" ? badge("declared", "ok") : badge("inferred", "warn"), confBadge(k.confidence), statusTag(k.origin === "inferred" ? k.status : null)))))] : null));
  pane.append(await relationshipsPanel(d, reload));
}

async function renderTerm(slug) {
  const t = await api("GET", `/api/business-terms/${encodeURIComponent(slug)}`);
  view.append(el("h1", {}, t.label || slug, " ", sensitiveTag(t.sensitive)), el("div", { class: "panel" }, el("div", { class: "mono muted" }, t.graph_ref),
    el("h2", {}, "Columns"), el("table", {}, el("tr", {}, ["Column", "Dataset", "Description", "Decoding"].map(h => el("th", {}, h))),
      t.columns.map(c => el("tr", {}, el("td", { class: "mono" }, c.catalog_ref), el("td", {}, el("a", { href: `#dataset/${c.dataset_id}` }, c.dataset)),
        el("td", {}, fmt(c.term_description)), el("td", { class: "small muted" }, fmt(c.term_explanation))))),
    t.definitions.length ? [el("h2", {}, "Definitions"), el("ul", {}, t.definitions.map(x => el("li", {}, el("a", { href: `#definition/${x.definition_id}` }, x.name))))] : null),
    graphPanel(t.graph));
}

async function renderEntity(slug) {
  const e = await api("GET", `/api/business-entities/${encodeURIComponent(slug)}`);
  view.append(el("h1", {}, e.label), el("div", { class: "panel" }, el("div", { class: "mono muted" }, e.graph_ref),
    el("h2", {}, "Datasets"), e.datasets.length ? el("ul", {}, e.datasets.map(d => el("li", {}, el("a", { href: `#dataset/${d.dataset_id}` }, d.catalog_ref)))) : el("p", { class: "muted" }, "None"),
    el("h2", {}, "Document mentions"), e.document_mentions.length ? el("ul", {}, e.document_mentions.map(m => el("li", {}, el("a", { href: `#chunk/${m.chunk_id}` }, m.chunk_id), " in ", el("a", { href: `#document/${m.document_id}` }, m.document_id)))) : el("p", { class: "muted" }, "None"),
    e.definitions.length ? [el("h2", {}, "Definitions"), el("ul", {}, e.definitions.map(x => el("li", {}, el("a", { href: `#definition/${x.definition_id}` }, x.name))))] : null),
    graphPanel(e.graph));
}

function graphPanel(g) {
  if (!g || !g.evidence || !g.evidence.length) return null;
  return el("div", { class: "panel" }, el("h2", {}, "Graph evidence"), g.evidence.map(k => el("div", { class: "result" },
    badge(k.type.replace(/_/g, " "), "info"), k.inferred ? badge("inferred", "warn") : null, el("div", {}, k.text),
    k.source_ref ? el("a", { class: "small", href: citationHash(k.source_ref) }, "provenance") : null)));
}

// ---------------------------------------------------------------- definitions
async function renderDefinitions() {
  const name = el("input", { type: "text", placeholder: "Active Customer" });
  const text = el("textarea", { placeholder: "A customer with a completed order in the past 90 days." });
  const ents = el("input", { type: "text", placeholder: "Customer, Order" });
  const terms = el("input", { type: "text", placeholder: "Customer Identifier" });
  const author = el("input", { type: "text", placeholder: "your name" });
  const src = el("input", { type: "text", placeholder: "policy document, ticket, URL…" });
  const result = el("div");
  const split = s => s.value.split(",").map(x => x.trim()).filter(Boolean);
  const save = async () => {
    if (!checkRequired(true)) return;
    result.replaceChildren();
    try {
      const d = await api("POST", "/api/business-definitions", { name: name.value, definition: text.value,
        entities: split(ents), terms: split(terms), author: author.value || null, source_ref: src.value || null });
      toast(`Saved ${d.name}`); location.hash = `#definition/${d.definition_id}`;
    } catch (err) { showError(result, err); }
  };
  const list = await api("GET", "/api/business-definitions");
  view.append(el("h1", {}, "Business definitions"),
    el("div", { class: "panel" }, el("h2", {}, "Capture a definition"),
      el("label", {}, "Name"), name, el("label", {}, "Definition"), text,
      el("label", {}, "Business entities it describes (comma separated)"), ents,
      el("label", {}, "Business terms it describes (comma separated)"), terms,
      el("div", { class: "grid3" }, el("div", {}, el("label", {}, "Author"), author), el("div", {}, el("label", {}, "Source reference"), src)),
      el("div", { style: "margin-top:10px" }, el("button", { onclick: save }, "Save definition")),
      el("p", { class: "small muted" }, "Definitions are validated against SHACL shapes before publication; a definition must link to at least one entity or term."),
      result),
    el("div", { class: "panel" }, el("h2", {}, `Captured (${list.definitions.length})`),
      el("table", {}, list.definitions.map(d => el("tr", {}, el("td", {}, el("a", { href: `#definition/${d.definition_id}` }, d.name)),
        el("td", {}, d.definition), el("td", {}, [...d.entities, ...d.terms].map(x => badge(x.label))))))));
}

async function renderDefinition(id) {
  const d = await api("GET", `/api/business-definitions/${encodeURIComponent(id)}`);
  view.append(el("h1", {}, d.name), el("div", { class: "panel" }, el("div", { class: "answer" }, d.definition),
    el("h3", {}, "Describes"), el("div", {}, d.entities.map(e => el("a", { href: `#entity/${e.ref.split("/").pop()}`, style: "margin-right:8px" }, badge(`entity: ${e.label}`, "ok"))),
      d.terms.map(t => el("a", { href: `#term/${t.ref.split("/").pop()}`, style: "margin-right:8px" }, badge(`term: ${t.label}`, "info")))),
    el("table", { style: "margin-top:10px" }, Object.entries({ Author: d.author, "Source reference": d.source_ref, Created: d.created_at,
      Updated: d.updated_at, "Graph ref": d.graph_ref }).map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", { class: "mono" }, fmt(v))))),
    el("div", { style: "margin-top:10px" }, el("button", { class: "danger", onclick: async () => {
      if (!confirm("Delete this definition?")) return; await api("DELETE", d.ref); location.hash = "#definitions"; } }, "Delete"))));
}

// ---------------------------------------------------------------- settings
const KIND_LABELS = { database: "Database", parquet: "Parquet repository", databricks: "Databricks" };

async function renderSettings(editId) {
  const data = await api("GET", "/api/settings");
  const editing = editId ? data.sources.find(s => s.source_id === editId && s.editable) : null;
  view.append(el("h1", {}, "Settings"));
  view.append(sourcesPanel(data));
  view.append(sourceForm(data, editing || null));
}

function credBadge(c) {
  if (c.store === "none") return badge("no credentials needed");
  const where = c.store === "env" ? `env $${c.reference}` : "OS keychain";
  return badge(`${where} · ${c.status}`, c.status === "present" ? "ok" : "bad");
}

function testResult(r) {
  if (!r.ok) return el("div", { class: "error" }, `Connection failed: ${r.error.message}`);
  const names = r.datasets.slice(0, 10).join(", ") + (r.dataset_count > 10 ? " …" : "");
  return el("div", {}, badge("connected", "ok"),
    r.dataset_count ? `${r.dataset_count} dataset(s) visible` : badge("0 datasets", "bad"),
    r.datasets.length ? el("div", { class: "small muted mono" }, names) : null,
    r.warning ? el("div", { class: r.dataset_count ? "small" : "small error", style: "margin-top:4px" }, r.warning) : null);
}

async function testSaved(id, out) {
  out.replaceChildren(el("span", { class: "muted" }, "Testing…"));
  try {
    const r = await api("POST", `/api/settings/sources/${encodeURIComponent(id)}/test`);
    out.replaceChildren(testResult(r));
  } catch (err) {
    out.replaceChildren(el("span", { class: "error" }, err.message));
  }
}

async function catalogNow(id) {
  try {
    const r = await api("POST", `/api/structured-sources/${encodeURIComponent(id)}/catalog`, {});
    location.hash = `#job/${r.job_id}`;
  } catch (err) {
    toast(err.message, true);
  }
}

function goSettings() {
  if (location.hash === "#settings") route();
  else location.hash = "#settings";
}

async function removeSource(id) {
  if (!confirm(`Remove source ${id}? Its catalog records, glossary entries and graph evidence will be retired.`)) return;
  try {
    const r = await api("DELETE", `/api/settings/sources/${encodeURIComponent(id)}`);
    toast(`Removed ${id} (${r.retired_datasets} dataset(s) retired)`);
    goSettings();
  } catch (err) {
    toast(err.message, true);
  }
}

function sourcesPanel(data) {
  const header = el("tr", {}, ["Source", "Type", "Target", "Credentials", "Defined in", "Catalog", ""].map(h => el("th", {}, h)));
  const rows = data.sources.map(s => {
    const out = el("div", { class: "small test-result" });
    const actions = el("div", { class: "row" },
      el("button", { class: "secondary", onclick: () => testSaved(s.source_id, out) }, "Test"),
      el("button", { class: "secondary", onclick: () => catalogNow(s.source_id) }, "Catalog now"),
      s.editable ? el("button", { class: "secondary", onclick: () => { location.hash = `#settings/${encodeURIComponent(s.source_id)}`; } }, "Edit") : null,
      s.editable ? el("button", { class: "danger", onclick: () => removeSource(s.source_id) }, "Remove") : null);
    return el("tr", {},
      el("td", {}, el("strong", {}, s.source_id), s.description ? el("div", { class: "small muted" }, s.description) : null),
      el("td", {}, KIND_LABELS[s.kind] || s.kind),
      el("td", { class: "mono small" }, s.target),
      el("td", {}, credBadge(s.credentials)),
      el("td", {}, s.origin === "config_file" ? badge("config/mesh.yaml") : badge("Settings page", "info")),
      el("td", {}, statusBadge(s.status || "configured"), el("div", { class: "small muted" }, `${s.dataset_count} datasets`)),
      el("td", {}, actions, out));
  });
  const store = data.secret_store;
  return el("div", { class: "panel" },
    el("h2", {}, "Structured sources"),
    data.sources.length ? el("table", {}, header, rows) : el("p", { class: "muted" }, "No structured sources yet. Add one below."),
    el("p", { class: "hint" }, "Sources defined in config/mesh.yaml are read-only here. ",
      `Secret store: ${store.backend}${store.available ? "" : " (unavailable: " + (store.reason || "unknown") + ")"}.`));
}

function credentialChoice(name, initial, keychainAvailable) {
  const envR = el("input", { type: "radio", name, value: "env", checked: initial === "env" ? true : null });
  const secR = el("input", { type: "radio", name, value: "secret", checked: initial !== "env" ? true : null,
    disabled: keychainAvailable ? null : true });
  const node = el("div", { class: "cred-choice" },
    el("label", {}, envR, "Environment variable"),
    el("label", {}, secR, "Store in OS keychain"));
  return { envR, secR, node };
}

function sourceForm(data, editing) {
  const st = editing ? editing.settings : {};
  const hasSecret = editing && editing.credentials.store === "secret_store";
  let kind = editing ? editing.kind : "database";
  const f = {};
  const text = (key, placeholder, value) => (f[key] = el("input", { type: "text", placeholder, value: value || "" }));
  const field = (label, node, hint, required) => el("div", {},
    el("label", { class: required ? "required" : null }, label), node, hint ? el("p", { class: "hint" }, hint) : null);
  const joined = v => (v || []).join(", ");
  const split = s => s.split(",").map(x => x.trim()).filter(Boolean);
  const keychain = data.secret_store.available;
  const initialCred = editing ? (editing.credentials.store === "env" ? "env" : "secret") : (keychain ? "secret" : "env");

  text("id", "crm", editing ? editing.source_id : "");
  if (editing) f.id.disabled = true;
  text("description", "CRM production replica", st.description);
  text("datasets", "optional: customers, sales.orders", joined(st.datasets));
  const dbTypes = data.database_types || {};
  f.db_type = el("select", {},
    Object.entries(dbTypes).map(([k, t]) => el("option", { value: k }, t.label)),
    el("option", { value: "other" }, "Other (full SQLAlchemy connection URL, e.g. SQLite)"));
  f.db_type.value = st.db_type || (editing && editing.kind === "database" ? "other" : "postgresql");
  text("database", "crm", st.database);
  text("host", "db.example.com", st.host);
  text("port", "", st.port ? String(st.port) : "");
  text("schemas", "public", joined(st.schemas));
  text("username", "readonly_user", st.username);
  text("password_env", "CRM_DB_PASSWORD", st.password_env);
  f.password = el("input", { type: "password", autocomplete: "off",
    placeholder: hasSecret && st.host ? "leave blank to keep the stored password" : "password" });
  text("connection_url_env", "CRM_DATABASE_URL", st.connection_url_env);
  f.connection_url = el("input", { type: "password", autocomplete: "off",
    placeholder: hasSecret && !st.host ? "leave blank to keep the stored URL" : "sqlite:////absolute/path/crm.db" });
  text("path", `${data.allowed_source_roots[0] || "/data/sources"}/exports`, st.path);
  text("server_hostname", "adb-1234567890123456.7.azuredatabricks.net", st.server_hostname);
  text("http_path", "/sql/1.0/warehouses/abc123def456", st.http_path);
  text("catalog", "sales", st.catalog);
  text("schema", "public", st.schema);
  text("token_env", "DATABRICKS_TOKEN", st.token_env);
  f.token = el("input", { type: "password", autocomplete: "off",
    placeholder: hasSecret ? "leave blank to keep the stored token" : "dapi…" });

  const dbCred = credentialChoice("db-cred", initialCred, keychain);
  const pwCred = credentialChoice("pw-cred", st.password_env ? "env" : (keychain ? "secret" : "env"), keychain);
  const dbxCred = credentialChoice("dbx-cred", initialCred, keychain);
  const envHint = "The variable must be set in the server's environment before it starts.";
  const keyHint = `Saved to the OS keychain (${data.secret_store.backend}); it is never shown again or written to the app's files.`;
  const dbEnv = field("Connection URL environment variable", f.connection_url_env, envHint, true);
  const dbSecret = field("Connection URL", f.connection_url, keyHint, !(hasSecret && !st.host));
  const pwEnv = field("Password environment variable", f.password_env, envHint);
  const pwSecret = field("Password", f.password, keyHint + " Leave empty if the database needs no password.");
  const dbxEnv = field("Token environment variable", f.token_env, envHint, true);
  const dbxSecret = field("Personal access token", f.token, keyHint, !hasSecret);
  const dbNameLabel = el("label", { class: "required" }, "Database name");
  const portHint = el("p", { class: "hint" }, "");

  const dbFields = el("div", { class: "form-grid" },
    el("div", {}, dbNameLabel, f.database),
    field("Hostname", f.host, null, true),
    el("div", {}, el("label", {}, "Port"), f.port, portHint));
  const dbUrl = el("div", {}, el("label", {}, "Connection URL"), dbCred.node, dbEnv, dbSecret,
    el("p", { class: "hint" }, "Any SQLAlchemy URL, e.g. sqlite:////absolute/path/crm.db. Use a read-only account."));
  const dbLogin = el("div", {},
    el("div", { class: "form-grid" }, field("Username", f.username, "Use a read-only account.")),
    el("label", {}, "Password"), pwCred.node, pwEnv, pwSecret);

  const groups = {
    database: el("div", {},
      el("div", { class: "form-grid" }, field("Database type", f.db_type, null, true)),
      dbFields, dbUrl,
      el("div", { class: "form-grid" }, field("Schema (optional)", f.schemas, "Leave empty to scan all schemas of the database. Separate several with commas.")),
      dbLogin),
    parquet: el("div", {},
      field("File or folder", f.path, "A .parquet file or a folder of them (each file is one dataset). Allowed under: " +
        data.allowed_source_roots.join(", ") + " — add more roots in storage.allowed_source_roots.", true)),
    databricks: el("div", {},
      el("div", { class: "form-grid" },
        field("Server hostname", f.server_hostname, "SQL warehouse → Connection details.", true),
        field("HTTP path", f.http_path, null, true),
        field("Catalog", f.catalog, null, true),
        field("Schema", f.schema, "\"default\" when empty.")),
      el("label", {}, "Credentials"), dbxCred.node, dbxEnv, dbxSecret,
      el("p", { class: "hint" }, "Use a token for a principal with read-only (USE/SELECT) privileges.")),
  };

  const refresh = () => {
    for (const [k, g] of Object.entries(groups)) g.style.display = k === kind ? "" : "none";
    const urlMode = f.db_type.value === "other";
    dbFields.style.display = urlMode ? "none" : "";
    dbLogin.style.display = urlMode ? "none" : "";
    dbUrl.style.display = urlMode ? "" : "none";
    const t = dbTypes[f.db_type.value];
    dbNameLabel.textContent = f.db_type.value === "oracle" ? "Service name" : "Database name";
    f.port.placeholder = t ? String(t.port) : "";
    portHint.textContent = t ? `Default ${t.port} when empty.` : "";
    dbEnv.style.display = dbCred.envR.checked ? "" : "none";
    dbSecret.style.display = dbCred.envR.checked ? "none" : "";
    pwEnv.style.display = pwCred.envR.checked ? "" : "none";
    pwSecret.style.display = pwCred.envR.checked ? "none" : "";
    dbxEnv.style.display = dbxCred.envR.checked ? "" : "none";
    dbxSecret.style.display = dbxCred.envR.checked ? "none" : "";
    picker.querySelectorAll("label").forEach(l => l.classList.toggle("selected", l.dataset.kind === kind));
  };
  [dbCred.envR, dbCred.secR, pwCred.envR, pwCred.secR, dbxCred.envR, dbxCred.secR].forEach(r => r.addEventListener("change", refresh));
  f.db_type.addEventListener("change", refresh);

  const picker = el("div", { class: "kind-picker" }, Object.entries(KIND_LABELS).map(([k, label]) => {
    const radio = el("input", { type: "radio", name: "source-kind", value: k, checked: k === kind ? true : null,
      disabled: editing ? true : null });
    radio.addEventListener("change", () => { kind = k; refresh(); });
    const l = el("label", {}, radio, label);
    l.dataset.kind = k;
    return l;
  }));

  const payload = () => {
    const p = { id: f.id.value.trim(), kind, description: f.description.value.trim() || null, datasets: split(f.datasets.value) };
    if (kind === "database") {
      p.schemas = split(f.schemas.value);
      if (f.db_type.value === "other") {
        if (dbCred.envR.checked) p.connection_url_env = f.connection_url_env.value.trim();
        else if (f.connection_url.value) p.connection_url = f.connection_url.value;
      } else {
        p.db_type = f.db_type.value;
        p.database = f.database.value.trim();
        p.host = f.host.value.trim();
        const port = f.port.value.trim();
        if (port && !/^\d+$/.test(port)) throw Object.assign(new Error("Port must be a number"), { fields: ["port"] });
        p.port = port ? Number(port) : null;
        p.username = f.username.value.trim() || null;
        if (pwCred.envR.checked) p.password_env = f.password_env.value.trim() || null;
        else if (f.password.value) p.password = f.password.value;
      }
    } else if (kind === "parquet") {
      p.path = f.path.value.trim();
    } else {
      p.server_hostname = f.server_hostname.value.trim();
      p.http_path = f.http_path.value.trim();
      p.catalog = f.catalog.value.trim();
      p.schema = f.schema.value.trim() || null;
      if (dbxCred.envR.checked) p.token_env = f.token_env.value.trim();
      else if (f.token.value) p.token = f.token.value;
    }
    return p;
  };

  const result = el("div", { class: "test-result" });
  const LABELS = { id: "Source ID", db_type: "Database type", host: "Hostname", port: "Port", path: "File or folder",
    connection_url: "Connection URL", connection_url_env: "Connection URL environment variable",
    server_hostname: "Server hostname", http_path: "HTTP path", catalog: "Catalog", token: "Personal access token",
    token_env: "Token environment variable", schemas: "Schema", username: "Username", password: "Password" };
  const labelOf = k => k === "database" ? dbNameLabel.textContent : (LABELS[k] || k);
  const visible = k => f[k] && f[k].offsetParent !== null;
  const markInvalid = names => {
    Object.values(f).forEach(n => n.classList && n.classList.remove("invalid"));
    const shown = names.filter(visible);
    shown.forEach(k => f[k].classList.add("invalid"));
    if (shown.length) f[shown[0]].focus();
    return shown;
  };
  const requiredNow = forSave => {
    const req = [];
    if (forSave && !editing) req.push("id");
    if (kind === "database") {
      if (f.db_type.value === "other") req.push(dbCred.envR.checked ? "connection_url_env" : (hasSecret && !st.host ? null : "connection_url"));
      else req.push("database", "host");
    } else if (kind === "parquet") {
      req.push("path");
    } else {
      req.push("server_hostname", "http_path", "catalog", dbxCred.envR.checked ? "token_env" : (hasSecret ? null : "token"));
    }
    return req.filter(Boolean);
  };
  const checkRequired = forSave => {
    const missing = requiredNow(forSave).filter(k => !String(f[k].value || "").trim());
    markInvalid(missing);
    if (missing.length) {
      result.replaceChildren(el("div", { class: "error" }, `Required: ${missing.map(labelOf).join(", ")}`));
      return false;
    }
    return true;
  };
  const fail = err => {
    const details = (err.data && err.data.error && err.data.error.details) || {};
    const shown = markInvalid(err.fields || details.fields || []);
    result.replaceChildren();
    if (err.data) showError(result, err);
    else result.append(el("div", { class: "error" }, err.message));
    if (!shown.length && (details.fields || []).length) result.append(el("div", { class: "small" }, "Field(s): " + details.fields.map(labelOf).join(", ")));
  };
  Object.values(f).forEach(n => n.addEventListener && n.addEventListener("input", () => n.classList.remove("invalid")));
  const test = async () => {
    if (!checkRequired(false)) return;
    result.replaceChildren(el("span", { class: "muted" }, "Testing connection…"));
    try {
      const r = await api("POST", "/api/settings/sources/test", payload());
      if (!r.ok && r.error.fields) markInvalid(r.error.fields);
      result.replaceChildren(testResult(r));
    } catch (err) {
      fail(err);
    }
  };
  const save = async () => {
    result.replaceChildren();
    try {
      const p = payload();
      const saved = editing
        ? await api("PUT", `/api/settings/sources/${encodeURIComponent(editing.source_id)}`, p)
        : await api("POST", "/api/settings/sources", p);
      f.connection_url.value = "";
      f.password.value = "";
      f.token.value = "";
      if (!editing && confirm(`Saved ${saved.source_id}. Catalog it now (schemas, keys and business terms)?`)) {
        return catalogNow(saved.source_id);
      }
      toast(`Saved ${saved.source_id}`);
      goSettings();
    } catch (err) {
      fail(err);
    }
  };

  const panel = el("div", { class: "panel" },
    el("h2", {}, editing ? `Edit source ${editing.source_id}` : "Add a source"),
    editing ? null : el("label", {}, "Type"), picker,
    el("div", { class: "form-grid" },
      field("Source ID", f.id, editing ? "The ID cannot be changed." : "Letters, digits, _ . - (used in catalog references).", !editing),
      field("Description", f.description)),
    groups.database, groups.parquet, groups.databricks,
    field("Datasets to include", f.datasets, "Optional allowlist (table or schema.table). Empty = every dataset in scope."),
    el("div", { class: "row", style: "margin-top:12px" },
      el("button", { class: "secondary", onclick: test }, "Test connection"),
      el("button", { onclick: save }, editing ? "Save changes" : "Save source"),
      editing ? el("button", { class: "secondary", onclick: () => { location.hash = "#settings"; } }, "Cancel") : null),
    result);
  refresh();
  return panel;
}

// ---------------------------------------------------------------- jobs
async function renderJobs() {
  const data = await api("GET", "/api/jobs?limit=100");
  view.append(el("h1", {}, "Jobs"), el("div", { class: "panel" }, el("table", {},
    el("tr", {}, ["Job", "Kind", "Status", "Counts", "Created"].map(h => el("th", {}, h))),
    data.jobs.map(j => el("tr", {}, el("td", {}, el("a", { href: `#job/${j.job_id}` }, j.job_id)), el("td", {}, j.kind),
      el("td", {}, statusBadge(j.status)), el("td", { class: "small mono" }, JSON.stringify(j.counts)), el("td", { class: "small" }, j.created_at))))));
}

async function renderJob(id, offset) {
  offset = offset || 0;
  const job = await api("GET", `/api/jobs/${encodeURIComponent(id)}`);
  const items = await api("GET", `/api/jobs/${encodeURIComponent(id)}/items?offset=${offset}&limit=100`);
  view.replaceChildren(el("h1", {}, `Job ${job.job_id}`),
    el("div", { class: "panel" }, el("div", { class: "row" }, badge(job.kind, "info"), statusBadge(job.status),
      el("span", { class: "small mono" }, JSON.stringify(job.counts))),
      job.error ? el("pre", { class: "error" }, JSON.stringify(job.error, null, 2)) : null,
      Object.keys(job.result || {}).length ? el("details", {}, el("summary", {}, "Result summary"), el("pre", {}, JSON.stringify(job.result, null, 2))) : null,
      el("div", { class: "small mono muted" }, `trace ${fmt(job.trace_id)} · created ${job.created_at} · finished ${fmt(job.finished_at)}`)),
    el("div", { class: "panel" }, el("h2", {}, `Manifest (${items.total} items)`), el("table", {},
      el("tr", {}, ["#", "Reference", "Status", "Reason", "Detail"].map(h => el("th", {}, h))),
      items.items.map(i => el("tr", {}, el("td", {}, i.seq), el("td", { class: "mono small" }, linkFor(i)), el("td", {}, statusBadge(i.status)),
        el("td", { class: "small" }, fmt(i.reason)), el("td", { class: "small mono muted" }, Object.keys(i.detail).length ? JSON.stringify(i.detail) : "")))),
      el("div", { class: "row", style: "margin-top:8px" },
        offset > 0 ? el("button", { class: "secondary", onclick: () => renderJob(id, Math.max(0, offset - 100)) }, "Previous") : null,
        items.next_offset !== null ? el("button", { class: "secondary", onclick: () => renderJob(id, items.next_offset) }, "Next") : null)));
  if (job.status === "queued" || job.status === "running") refreshTimer = setTimeout(() => renderJob(id, offset).catch(() => {}), 2000);
}
function linkFor(item) {
  const ref = item.ref || "";
  if (ref.startsWith("ds-")) return el("a", { href: `#dataset/${ref}` }, ref);
  if (ref.startsWith("doc-")) return el("a", { href: `#document/${ref}` }, ref);
  if (item.detail && item.detail.document_id) return el("a", { href: `#document/${item.detail.document_id}` }, ref);
  return ref;
}

loadHealth();
setInterval(loadHealth, 30000);
route();
