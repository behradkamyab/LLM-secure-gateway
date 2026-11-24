from pathlib import Path

import pytest
from pydantic import ValidationError

from gateway.common.jsonl import load_jsonl, save_jsonl
from gateway.common.schema import PIIEntity, PIIRecord
from gateway.data.generate import parse_llm_batch
from gateway.infer.redactor import EMAIL_PATTERN


def test_jsonl_typed_roundtrip_and_generator_streaming(tmp_path: Path):
    target = tmp_path / "records.jsonl"
    original = [
        PIIRecord(
            id="it-1",
            original_prompt="Sono Giulia Conti e scrivo da Milano.",
            entities=[PIIEntity(text="Giulia Conti", label="PERSON")],
        ),
        PIIRecord(
            id="it-2",
            original_prompt="La mia email è giulia.c@azienda.it.",
            entities=[PIIEntity(text="giulia.c@azienda.it", label="EMAIL")],
        ),
    ]

    # Pass a generator expression to verify streaming Iterable support
    save_jsonl((rec for rec in original), target, schema=PIIRecord)
    assert target.exists()
    assert not target.with_suffix(".jsonl.tmp").exists()

    loaded = load_jsonl(target, schema=PIIRecord)
    assert loaded == original
    assert isinstance(loaded[0], PIIRecord)


def test_jsonl_atomic_write_preserves_existing_file_on_failure(tmp_path: Path):
    target = tmp_path / "safe.jsonl"
    valid_record = PIIRecord(id="ok-1", original_prompt="Ticket valido.")
    save_jsonl([valid_record], target, schema=PIIRecord)

    bad_batch = [
        valid_record,
        {"id": "bad-2", "original_prompt": "   "},
    ]
    with pytest.raises(ValidationError):
        save_jsonl(bad_batch, target, schema=PIIRecord)

    assert not target.with_suffix(".jsonl.tmp").exists()
    still_valid = load_jsonl(target, schema=PIIRecord)
    assert len(still_valid) == 1
    assert still_valid[0].id == "ok-1"


def test_load_jsonl_line_number_note_on_corrupt_line(tmp_path: Path):
    target = tmp_path / "broken.jsonl"
    target.write_text(
        '{"id": "ok-1", "original_prompt": "Ciao", "entities": []}\n'
        '{"id": "bad-2", "original_prompt": ""}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as exc_info:
        load_jsonl(target, schema=PIIRecord)

    assert any(":2" in note for note in getattr(exc_info.value, "__notes__", []))


def test_load_jsonl_untyped_and_missing_file(tmp_path: Path):
    target = tmp_path / "raw.jsonl"
    save_jsonl([{"a": 1}, {"a": 2}], target)
    raw = load_jsonl(target)
    assert raw == [{"a": 1}, {"a": 2}]

    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "does_not_exist.jsonl")


def test_parse_llm_batch_filters_hallucinated_spans():
    raw_payload = """
    {
      "records": [
        {
          "id": "dup-id",
          "original_prompt": "Sono Marco Carta e la mia email e marco@azienda.it.",
          "entities": [
            {"text": "Marco Carta", "label": "PERSON"},
            {"text": "marco@azienda.it", "label": "EMAIL"}
          ]
        },
        {
          "id": "dup-id",
          "original_prompt": "Ho bisogno di supporto VPN.",
          "entities": [
            {"text": "Nome Inventato", "label": "PERSON"}
          ]
        }
      ]
    }
    """
    records = parse_llm_batch(raw_payload, start_index=10)
    assert len(records) == 1
    assert records[0].id == "syn-10"
    assert len(records[0].entities) == 2


def test_email_regex_strips_trailing_sentence_period():
    sentence = "Per favore contatta Antonietta Gotti all'indirizzo moorelouis@example.com."
    matches = EMAIL_PATTERN.findall(sentence)
    assert matches == ["moorelouis@example.com"]


def test_dockerfile_and_dockerignore_production_rules():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in dockerfile
    assert "uv pip install --system" in dockerfile
    assert "libgomp1" in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    assert "gateway.infer.main:app" in dockerfile

    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    for blocked in (".venv", ".git", ".env", "outputs", "data/*"):
        assert blocked in dockerignore
    assert "!data/router_intents.json" in dockerignore


def test_pyproject_declares_gateway_serve_entrypoint():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'gateway-serve = "gateway.cli:serve_main"' in pyproject
