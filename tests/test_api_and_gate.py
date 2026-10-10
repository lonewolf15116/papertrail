import pytest
from fastapi.testclient import TestClient

from papertrail import api as api_module
from papertrail.api import app
from papertrail.eval.gate import check_regression
from papertrail.retrieval import reciprocal_rank_fusion
from papertrail.schemas import AnswerStatus, AskResponse, Citation

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


class StubPipeline:
    def __init__(self, resp: AskResponse | None = None, boom: bool = False) -> None:
        self.resp, self.boom = resp, boom
        self.seen: list[tuple[str, int | None]] = []

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        self.seen.append((question, top_k))
        if self.boom:
            raise RuntimeError("upstream down")
        assert self.resp is not None
        return self.resp


@pytest.fixture
def with_pipeline(monkeypatch):
    def _install(stub: StubPipeline) -> StubPipeline:
        monkeypatch.setattr(api_module, "get_pipeline", lambda: stub)
        return stub

    return _install


def test_ask_returns_the_pipelines_answer(with_pipeline):
    cite = Citation(paper_id="dtr2021", section="3", page=3, chunk_id="dtr2021:003", quote="q" * 20)
    stub = with_pipeline(
        StubPipeline(AskResponse(status=AnswerStatus.ANSWERED, answer="A.", citations=[cite]))
    )
    r = client.post("/ask", json={"question": "What does DTR evict?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered" and body["citations"][0]["page"] == 3
    assert stub.seen == [("What does DTR evict?", 3)]


def test_ask_passes_refusals_through(with_pipeline):
    with_pipeline(
        StubPipeline(AskResponse(status=AnswerStatus.REFUSED, refusal_reason="Not in the corpus."))
    )
    body = client.post("/ask", json={"question": "Who won the 2022 World Cup?"}).json()
    assert body["status"] == "refused" and body["refusal_reason"] == "Not in the corpus."


def test_ask_maps_pipeline_failure_to_502(with_pipeline):
    with_pipeline(StubPipeline(boom=True))
    r = client.post("/ask", json={"question": "What does DTR evict?"})
    assert r.status_code == 502 and "upstream down" not in r.text


def test_ask_is_503_when_the_pipeline_cannot_be_built(monkeypatch):
    def broken() -> None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(api_module, "_pipeline", None)
    monkeypatch.setattr(api_module, "build_default_pipeline", broken)
    r = client.post("/ask", json={"question": "What does DTR evict?"})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert client.get("/health").status_code == 200  # health never depends on the pipeline


def test_ask_validates_input_before_touching_the_pipeline(monkeypatch):
    def broken() -> None:
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(api_module, "_pipeline", None)
    monkeypatch.setattr(api_module, "build_default_pipeline", broken)
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


def test_gate_ignores_metadata_keys_in_baseline():
    baseline = {"recall@5": 0.70, "mrr": 0.50, "_retriever": "hybrid_header", "_questions": 64}
    current = {"recall@5": 0.69, "mrr": 0.49, "_retriever": "hybrid_header"}
    assert check_regression(current, baseline, {"recall@5": 0.02, "mrr": 0.02}).passed
    worse = {"recall@5": 0.60, "mrr": 0.49}
    assert not check_regression(worse, baseline, {"recall@5": 0.02, "mrr": 0.02}).passed
