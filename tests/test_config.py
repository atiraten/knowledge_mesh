from pathlib import Path

import pytest

from knowledge_mesh.config import ConfigError, config_from_dict, load_config

REPO = Path(__file__).resolve().parents[1]


def test_repository_config_loads_and_resolves_paths():
    cfg = load_config(REPO / "config" / "mesh.yaml")
    assert cfg.embedding.model == "sentence-transformers/all-mpnet-base-v2"
    assert cfg.embedding.provider == "sentence_transformers"
    assert cfg.retrieval.top_k == 5
    assert cfg.chunking.chunk_sizes_tokens == [2048, 1024, 256]
    assert cfg.crawler.max_depth == 3 and cfg.crawler.max_pages == 100 and cfg.crawler.concurrency_per_host == 2
    assert cfg.storage.metadata_db == REPO / "data" / "metadata.sqlite"
    assert cfg.term_generation.provider == "google"
    assert cfg.term_generation.api_key_file == REPO / "config" / "model.key"
    assert cfg.term_generation.prompt_path == REPO / "prompts" / "term_generation.txt"
    assert not cfg.answer_generation.enabled or cfg.answer_generation.configured


@pytest.mark.parametrize("bad", [
    {"retrieval": {"top_k": 50, "max_top_k": 20}},
    {"chunking": {"chunk_sizes_tokens": [256, 1024]}},
    {"structured_sources": [{"id": "x", "kind": "database"}]},
    {"answer_generation": {"enabled": True}},
    {"unknown_section": {}},
])
def test_invalid_configuration_is_rejected(tmp_path, bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad, tmp_path)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
