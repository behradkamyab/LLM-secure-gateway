from pathlib import Path

import pytest

from gateway.common.config import load_config


def test_load_e0_zeroshot_config():
    cfg = load_config("configs/e0_zeroshot.yaml")
    assert cfg["experiment"]["id"] == "E0_zeroshot"
    assert cfg["experiment"]["seed"] == 42
    assert cfg["model"]["ner_engine"]["name"] == "yonigo/deberta-v3-base-pii-en"
    assert cfg["model"]["router_engine"]["name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg["data"]["test_path"] == "data/processed/test.jsonl"
    assert cfg["eval"]["target_entities"] == ["B-PII", "I-PII"]


def test_load_config_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
