import pytest

from knowledge_mesh.common.errors import Forbidden, InvalidArgument, PayloadTooLarge, UnsupportedMediaType
from knowledge_mesh.graph.ontology import MESH, RDF, RDFS, entity_uri
from tests.test_loaders_chunking import make_docx, make_pdf

POLICY = (
    "# Customer Identification\n"
    "Assign each customer a unique identifier when the account is opened. "
    "The customer identifier never changes.\n"
    "## Orders\n"
    "A customer places orders through the web store. Each order contains order lines.\n"
)


def ingest_text(mesh, text, name="policy.md", wait=30):
    return mesh.ingest_document(content=text.encode(), filename=name, wait_seconds=wait)


def leaf_count(mesh):
    return mesh.doc_index.stats()["embedded_nodes"]


def test_upload_index_and_status(mesh_factory):
    mesh = mesh_factory()
    out = ingest_text(mesh, POLICY)
    assert out["status"] == "new" and out["job_status"] == "succeeded"
    doc = mesh.documents.get(out["document_id"])
    assert doc["status"] == "indexed" and doc["leaf_count"] > 0
    assert doc["content_hash"] and doc["mime_type"] == "text/markdown" and doc["source_type"] == "upload"
    res = mesh.retrieve_context("how is each customer identifier assigned", evidence_types=["documents"])
    top = res["results"][0]
    assert top["document_id"] == out["document_id"]
    assert top["section"] == "Customer Identification"
    assert top["score_type"] == "cosine_similarity"
    assert top["citation"]["chunk_ref"] == f"/api/chunks/{top['chunk_id']}"


def test_pdf_and_docx_ingestion_with_page_citations(mesh_factory):
    mesh = mesh_factory()
    pdf = mesh.ingest_document(content=make_pdf(["Overview of the program.", "Refunds are paid within ten days."]),
                               filename="refunds.pdf", wait_seconds=30)
    docx_out = mesh.ingest_document(content=make_docx(), filename="policy.docx", wait_seconds=30)
    assert mesh.documents.get(pdf["document_id"])["status"] == "indexed"
    assert mesh.documents.get(docx_out["document_id"])["status"] == "indexed"
    res = mesh.retrieve_context("when are refunds paid", evidence_types=["documents"])
    hit = next(r for r in res["results"] if r["document_id"] == pdf["document_id"])
    assert hit["page"] == 2
    page2 = mesh.read_document(pdf["document_id"], page=2)
    assert "Refunds" in page2["text"] and page2["available_pages"] == [1, 2]


def test_reingest_unchanged_is_noop_and_update_replaces_chunks(mesh_factory):
    mesh = mesh_factory()
    first = ingest_text(mesh, POLICY)
    n = leaf_count(mesh)
    again = ingest_text(mesh, POLICY)
    assert again["status"] == "unchanged" and "job_id" not in again
    assert leaf_count(mesh) == n
    updated = ingest_text(mesh, POLICY + "\n## Payments\nA customer pays invoices by card.\n")
    assert updated["status"] == "updated" and updated["document_id"] == first["document_id"]
    chunks = mesh.db.query("SELECT chunk_id FROM chunks WHERE document_id = ? AND is_leaf = 1", (first["document_id"],))
    assert leaf_count(mesh) == len(chunks)  # stale chunks replaced, not accumulated


def test_duplicate_content_under_other_name_not_embedded_twice(mesh_factory):
    mesh = mesh_factory()
    ingest_text(mesh, POLICY, "a.md")
    n = leaf_count(mesh)
    dup = ingest_text(mesh, POLICY, "b.md")
    assert dup["status"] == "duplicate" and dup["duplicate_of"]
    assert leaf_count(mesh) == n


def test_delete_removes_chunks_and_graph_context(mesh_factory):
    mesh = mesh_factory()
    out = ingest_text(mesh, POLICY)
    assert mesh.graph.stats()["relationships"] >= 1
    mesh.delete_document(out["document_id"])
    assert leaf_count(mesh) == 0
    assert mesh.documents.get(out["document_id"])["status"] == "deleted"
    assert mesh.graph.stats()["relationships"] == 0
    assert mesh.retrieve_context("customer identifier", evidence_types=["documents"])["results"] == []


def test_persistence_across_restart(mesh_factory, config_factory):
    cfg = config_factory()
    mesh = mesh_factory(cfg)
    out = ingest_text(mesh, POLICY)
    mesh.capture_definition("Active Customer", "A customer with a completed order in the past 90 days.",
                            entities=["Customer"], author="tester")
    mesh.close()
    restarted = mesh_factory(cfg)
    res = restarted.retrieve_context("active customer identifier")
    assert any(r["document_id"] == out["document_id"] for r in res["results"])
    assert any(g["type"] == "business_definition" for g in res["graph_evidence"])
    assert restarted.documents.get(out["document_id"])["status"] == "indexed"


LONG_GUIDE = "# Guide\n" + " ".join(f"Rule {i}: customers place orders and receive invoices." for i in range(300))


def test_leaf_with_parent_chain(mesh_factory):
    mesh = mesh_factory()
    ingest_text(mesh, LONG_GUIDE, "guide.md")
    res = mesh.retrieve_context("customers receive invoices", top_k=1, evidence_types=["documents"],
                                expand_parents=True)
    top = res["results"][0]
    assert top["parent_id"].startswith("node-")
    assert [p["level"] for p in top["parents"]] == [1, 0] and all(p["expanded"] for p in top["parents"])
    assert top["parents"][0]["node_id"] == top["parent_id"]
    assert len(top["text"]) < len(top["parents"][0]["text"]) <= len(top["parents"][1]["text"])
    chunk = mesh.get_chunk(top["chunk_id"])
    assert [p["node_id"] for p in chunk["parents"]] == [p["node_id"] for p in top["parents"]]
    plain = mesh.retrieve_context("customers receive invoices", evidence_types=["documents"], expand_parents=False)
    assert "parents" not in plain["results"][0]


def test_parent_budget_fills_nearest_level_first_and_counts_shared_parents_once(mesh_factory):
    mesh = mesh_factory()
    ingest_text(mesh, LONG_GUIDE, "guide.md")
    query = "customers receive invoices"
    docs = mesh.retrieve_context(query, top_k=5, evidence_types=["documents"])["results"]
    assert len(docs) >= 3
    level0 = {p["node_id"] for d in docs for p in d["parents"] if p["level"] == 0}
    assert len(level0) < len(docs)  # results share a 2048-token parent
    level1 = {p["node_id"]: len(p["text"]) for d in docs for p in d["parents"] if p["level"] == 1}
    needed = sum(len(d["text"]) for d in docs) + sum(level1.values())
    mesh.retrieval.cfg.max_context_tokens = -(-needed // 4)  # room for leaves + each distinct level-1 parent once
    for d in mesh.retrieve_context(query, top_k=5, evidence_types=["documents"])["results"]:
        by_level = {p["level"]: p for p in d["parents"]}
        assert by_level[1]["expanded"] and by_level[1]["text"]
        assert not by_level[0]["expanded"] and by_level[0]["reason"] == "context budget exceeded"
        assert "text" not in by_level[0]


def test_top_k_default_override_and_bounds(mesh_factory):
    mesh = mesh_factory()
    for i in range(8):
        ingest_text(mesh, f"# Note {i}\nCustomer orders note number {i} about orders and customers.", f"n{i}.md")
    assert len(mesh.retrieve_context("customer orders", evidence_types=["documents"])["results"]) == 5
    assert len(mesh.retrieve_context("customer orders", top_k=7, evidence_types=["documents"])["results"]) == 7
    with pytest.raises(InvalidArgument):
        mesh.retrieve_context("customer orders", top_k=21)


def test_insufficient_evidence(mesh_factory):
    mesh = mesh_factory()
    ingest_text(mesh, POLICY)
    res = mesh.retrieve_context("zebra quantum xylophone telescope")
    assert res["insufficient_evidence"] is True
    assert res["results"] == [] and res["glossary_results"] == [] and res["graph_evidence"] == []


def test_document_entities_and_relationship_in_graph(mesh_factory):
    mesh = mesh_factory()
    out = ingest_text(mesh, "Suppliers ship products to the warehouse. A customer places orders online.")
    ds = mesh.graph.dataset
    assert (entity_uri("customer"), RDF.type, MESH.BusinessEntity) in ds
    assert (entity_uri("order"), RDF.type, MESH.BusinessEntity) in ds
    rels = [r for r in ds.subjects(RDF.type, MESH.Relationship)
            if str(ds.value(r, MESH.relationshipType)) == "places"]
    assert rels
    rel = rels[0]
    assert ds.value(rel, MESH.sourceEntity) == entity_uri("customer")
    assert ds.value(rel, MESH.targetEntity) == entity_uri("order")
    passage = ds.value(rel, URIRef_prov("wasDerivedFrom"))
    assert str(ds.value(passage, MESH.documentId)) == out["document_id"]
    chunk_id = str(ds.value(passage, MESH.chunkId))
    assert mesh.get_chunk(chunk_id)["document_id"] == out["document_id"]
    assert str(ds.value(entity_uri("customer"), RDFS.label)) == "Customer"


def URIRef_prov(name):
    from knowledge_mesh.graph.ontology import PROV

    return PROV[name]


def test_local_directory_ingestion_and_path_restrictions(mesh_factory, inbox, tmp_path):
    mesh = mesh_factory()
    root = inbox(mesh)
    (root / "a.txt").write_text("Payments are settled daily by the account team.")
    (root / "b.md").write_text("# Supplier\nSuppliers provide products.")
    (root / "c.exe").write_bytes(b"MZ")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 broken")
    job = mesh.ingest_document(path=str(root), wait_seconds=30)
    assert job["kind"] == "ingest_directory"
    counts = job["counts"]
    assert counts.get("indexed") == 2 and counts.get("skipped") == 1 and counts.get("failed") == 1
    assert job["status"] == "partial"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(Forbidden):
        mesh.ingest_document(path=str(outside))
    with pytest.raises(Forbidden):
        mesh.ingest_document(path=str(root / ".." / ".." / "outside.txt"))


def test_upload_limits(mesh_factory, config_factory):
    mesh = mesh_factory(config_factory({"storage": {"max_upload_bytes": 100}}))
    with pytest.raises(PayloadTooLarge):
        mesh.ingest_document(content=b"x" * 101, filename="big.txt")
    with pytest.raises(UnsupportedMediaType):
        mesh.ingest_document(content=b"abc", filename="evil.sh")
    out = mesh.ingest_document(content=b"Customer note.", filename="../../etc/passwd.txt", index=False)
    row = mesh.documents.get_row(out["document_id"])
    assert row["filename"] == "passwd.txt"
    assert str(mesh.config.storage.originals_dir) in row["original_path"]
