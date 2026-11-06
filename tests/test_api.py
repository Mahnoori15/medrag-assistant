import os

os.environ.pop("ANTHROPIC_API_KEY", None)  # tests run offline in extractive mode

from fastapi.testclient import TestClient

from app.main import app
from eval.run_eval import evaluate

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["chunks"] > 0


def test_ask_returns_citation():
    r = client.post("/ask", json={"question": "How is atrial fibrillation diagnosed?"})
    body = r.json()
    assert r.status_code == 200
    assert body["citations"][0]["doc_id"] == "S3"
    assert "Not medical advice" in body["disclaimer"]


def test_unrelated_question_is_refused():
    r = client.post("/ask", json={"question": "Who won the football world cup?"})
    assert r.json()["mode"] == "no_match"


def test_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_retrieval_quality():
    assert evaluate(k=3)["hit_rate@3"] >= 0.8
