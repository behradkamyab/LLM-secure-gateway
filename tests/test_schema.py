import pytest
from pydantic import ValidationError

from gateway.common.schema import (
    PIIEntity,
    PIIRecord,
    RedactedResponse,
    RedactionRequest,
    RouteResponse,
    SyntheticDataRecord,
    TokenizedNERRecord,
)


def test_pii_entity_default_and_custom_label():
    default_ent = PIIEntity(text="Mario Rossi")
    assert default_ent.text == "Mario Rossi"
    assert default_ent.label == "PII"

    custom_ent = PIIEntity(text="mario@azienda.it", label="EMAIL")
    assert custom_ent.label == "EMAIL"


def test_pii_entity_rejects_whitespace_only():
    with pytest.raises(ValidationError):
        PIIEntity(text="   ")


def test_pii_record_valid_and_empty_entities():
    record = PIIRecord(
        id="rec-1",
        original_prompt="Chiamami al numero aziendale.",
    )
    assert record.id == "rec-1"
    assert record.entities == []
    assert SyntheticDataRecord is PIIRecord


def test_pii_record_rejects_empty_or_whitespace_prompt():
    with pytest.raises(ValidationError):
        PIIRecord(id="rec-1", original_prompt="")

    with pytest.raises(ValidationError):
        PIIRecord(id="rec-1", original_prompt="   ")


def test_tokenized_ner_record_valid_and_length_mismatch():
    rec = TokenizedNERRecord(
        id="rec-1",
        input_ids=[101, 2023, 102],
        attention_mask=[1, 1, 1],
        labels=[-100, 0, -100],
    )
    assert len(rec.input_ids) == 3

    with pytest.raises(ValidationError):
        TokenizedNERRecord(
            id="rec-1",
            input_ids=[101, 2023, 102],
            attention_mask=[1, 1],
            labels=[-100, 0, -100],
        )


def test_redaction_request_bounds():
    valid = RedactionRequest(text="Ciao sono Lorenzo.")
    assert valid.text == "Ciao sono Lorenzo."

    with pytest.raises(ValidationError):
        RedactionRequest(text="   ")

    with pytest.raises(ValidationError):
        RedactionRequest(text="a" * 2001)


def test_redacted_and_route_responses():
    redacted = RedactedResponse(
        original_text="Sono Mario",
        redacted_text="Sono [REDACTED]",
        entities_found=1,
    )
    assert redacted.entities_found == 1

    with pytest.raises(ValidationError):
        RedactedResponse(
            original_text="Sono Mario",
            redacted_text="Sono [REDACTED]",
            entities_found=-1,
        )

    route = RouteResponse(destination="LLM_PIPELINE", confidence_score=0.42)
    assert route.destination == "LLM_PIPELINE"
