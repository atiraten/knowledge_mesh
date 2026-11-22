import json

from knowledge_mesh.evaluation.retrieval_eval import _metrics_for, evaluate, load_testset


def test_metric_math():
    m = _metrics_for(["a", "x", "b"], {"a", "b"}, 3)
    assert m["recall"] == 1.0 and m["hit"] == 1.0 and m["rr"] == 1.0 and 0 < m["ndcg"] <= 1
    assert _metrics_for(["x", "a"], {"a"}, 1)["recall"] == 0.0
    assert _metrics_for(["x", "a"], {"a"}, 2)["rr"] == 0.5


def test_offline_evaluation_run_is_recorded(mesh_factory, tmp_path):
    mesh = mesh_factory()
    a = mesh.ingest_document(content=b"# Refunds\nRefunds are paid within ten days.", filename="r.md", wait_seconds=30)
    b = mesh.ingest_document(content=b"# Shipping\nParcels ship within two days.", filename="s.md", wait_seconds=30)
    path = tmp_path / "set.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [
        {"id": "q1", "question": "when are refunds paid", "relevant_document_ids": [a["document_id"]]},
        {"id": "q2", "question": "how fast do parcels ship", "relevant_document_ids": [b["document_id"]]},
    ]))
    out = evaluate(mesh, load_testset(path), k=5, out_dir=tmp_path / "evals")
    assert out["summary"]["recall@5"] == 1.0 and out["summary"]["mrr"] == 1.0
    assert out["run"]["embedding_model"] == "test-hashing-embedding" and out["run"]["config_sha256"]
    saved = json.loads(open(out["output_path"]).read())
    assert saved["summary"] == out["summary"]


def test_rebuild_indexes_after_model_change(mesh_factory, config_factory):
    cfg = config_factory()
    mesh = mesh_factory(cfg)
    out = mesh.ingest_document(content=b"# Refunds\nRefunds are paid within ten days.", filename="r.md", wait_seconds=30)
    mesh.close()
    from knowledge_mesh.common.errors import IncompatibleIndex
    from knowledge_mesh.indexing.embeddings import HashingEmbedding

    other = HashingEmbedding(model_name="other-model", dim=128)
    changed = mesh_factory(cfg, embed_model=other)
    import pytest

    with pytest.raises(IncompatibleIndex):
        changed.retrieve_documents(["refunds"])
    result = changed.rebuild_indexes()
    assert result["documents_indexed"] == 1 and result["previous_indexes"]["documents"].endswith(
        tuple("0123456789"))
    assert changed.doc_index.stats()["embedding_model"] == "other-model"
    hits = changed.retrieve_documents(["when are refunds paid"], evidence_types=["documents"])["results"]
    assert hits and hits[0]["document_id"] == out["document_id"]
