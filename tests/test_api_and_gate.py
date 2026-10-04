from fastapi.testclient import TestClient

from papertrail.api import app
from papertrail.eval.gate import check_regression
from papertrail.retrieval import reciprocal_rank_fusion

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_refuses_until_pipeline_exists():
    r = client.post("/ask", json={"question": "What does DTR evict first?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "refused"
    assert body["refusal_reason"]


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_gate_flags_regression_beyond_margin():
    result = check_regression(
        current={"recall@5": 0.80, "mrr": 0.70},
        baseline={"recall@5": 0.85, "mrr": 0.71},
        margins={"recall@5": 0.02, "mrr": 0.02},
    )
    assert not result.passed
    assert len(result.failures) == 1 and result.failures[0].startswith("recall@5")


def test_gate_ignores_new_metrics():
    assert check_regression({"x": 0.1}, {}, {"x": 0.01}).passed


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[:2] == ["a", "b"]
    assert set(fused) == {"a", "b", "c", "d"}
