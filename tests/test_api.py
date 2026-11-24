import os

import pytest
from fastapi.testclient import TestClient

from gateway.infer.main import app


@pytest.fixture(scope="module")
def client():
    os.environ["HF_HUB_OFFLINE"] = "1"
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_gateway_faq_cache_hit_skips_redaction(client: TestClient):
    res = client.post(
        "/api/v1/gateway",
        json={"text": "How do I reset my Windows password?"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["route"]["destination"] == "IT_SUPPORT"
    assert body["route"]["confidence_score"] >= 0.60
    assert body["redacted_payload"] is None


def test_gateway_llm_pipeline_redacts_italian_pii(client: TestClient):
    res = client.post(
        "/api/v1/gateway",
        json={"text": "Mi chiamo Lorenzo Bianchi e la mia email è lorenzo.b@azienda.it."},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["route"]["destination"] == "LLM_PIPELINE"
    redacted = body["redacted_payload"]
    assert redacted is not None
    assert "lorenzo.b@azienda.it" not in redacted["redacted_text"]
    assert "[REDACTED]" in redacted["redacted_text"]
    assert redacted["entities_found"] >= 1


def test_gateway_rejects_blank_text(client: TestClient):
    res = client.post("/api/v1/gateway", json={"text": "   "})
    assert res.status_code == 422
