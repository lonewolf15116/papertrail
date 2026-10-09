from typing import Any

import pytest

from papertrail.answer import (
    TOOL_SCHEMA,
    AnthropicClient,
    Generation,
    build_response,
    render_user_message,
    validate_citations,
)
from papertrail.config import Settings
from papertrail.pipeline import Pipeline
from papertrail.schemas import AnswerStatus, Chunk

DTR = Chunk(
    chunk_id="dtr2021:003",
    paper_id="dtr2021",
    section="3 Dynamic Tensor Rematerialization",
    page=3,
    text="DTR evicts the tensor that is stalest, largest, and cheapest to rematerialize, "
    "using the heuristic h_DTR computed over a tensor's metadata.",
)
CHK = Chunk(
    chunk_id="checkmate2020:007",
    paper_id="checkmate2020",
    section="4 Optimal rematerialization",
    page=5,
    text="Checkmate formulates rematerialization as a mixed integer linear program and solves "
    "it with an off-the-shelf solver to obtain an optimal schedule.",
)
PRICES = (1.0, 5.0)


def gen(**data: Any) -> Generation:
    return Generation(data=data, input_tokens=1000, output_tokens=200)


def test_valid_citation_is_kept_with_chunk_metadata():
    resp = build_response(
        [DTR, CHK],
        gen(
            can_answer=True,
            answer="DTR evicts stale, large, cheap tensors.",
            citations=[{"source": 1, "quote": "evicts the tensor that is stalest, largest"}],
        ),
        PRICES,
    )
    assert resp.status is AnswerStatus.ANSWERED
    (c,) = resp.citations
    assert (c.paper_id, c.page, c.chunk_id) == ("dtr2021", 3, "dtr2021:003")
    assert c.section.startswith("3 Dynamic")
    assert resp.retrieved_chunk_ids == ["dtr2021:003", "checkmate2020:007"]
    assert resp.dropped_citations == 0


def test_quote_matching_ignores_case_spacing_and_hyphenation():
    got, dropped = validate_citations(
        [{"source": 1, "quote": "EVICTS   the tensor that is  stal-est, largest"}], [DTR]
    )
    assert len(got) == 1 and dropped == 0


def test_fabricated_quote_is_dropped_and_answer_withheld():
    resp = build_response(
        [DTR],
        gen(
            can_answer=True,
            answer="DTR uses an ILP solver.",
            citations=[{"source": 1, "quote": "DTR solves an integer linear program exactly"}],
        ),
        PRICES,
    )
    assert resp.status is AnswerStatus.REFUSED
    assert resp.dropped_citations == 1
    assert "could be verified" in (resp.refusal_reason or "")
    assert resp.answer is None and resp.citations == []


def test_quote_from_wrong_source_is_not_accepted():
    # The words are in the Checkmate chunk but the model cited source 1 (DTR).
    got, dropped = validate_citations(
        [{"source": 1, "quote": "formulates rematerialization as a mixed integer linear program"}],
        [DTR, CHK],
    )
    assert got == [] and dropped == 1


def test_partial_validity_answers_and_counts_the_dropped():
    resp = build_response(
        [DTR],
        gen(
            can_answer=True,
            answer="DTR evicts by a metadata heuristic.",
            citations=[
                {
                    "source": 1,
                    "quote": "using the heuristic h_DTR computed over a tensor's metadata",
                },
                {"source": 1, "quote": "this sentence is not in the paper at all"},
            ],
        ),
        PRICES,
    )
    assert resp.status is AnswerStatus.ANSWERED
    assert len(resp.citations) == 1 and resp.dropped_citations == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"source": 0, "quote": "evicts the tensor that is stalest"},
        {"source": 3, "quote": "evicts the tensor that is stalest"},
        {"source": True, "quote": "evicts the tensor that is stalest"},
        {"source": "1", "quote": "evicts the tensor that is stalest"},
        {"source": 1, "quote": 42},
        {"source": 1},
        {"source": 1, "quote": "stalest"},  # too short to verify anything
        "not a dict",
    ],
)
def test_malformed_citations_are_dropped(bad: object):
    got, dropped = validate_citations([bad], [DTR])
    assert got == [] and dropped == 1


def test_duplicate_citations_are_merged_not_counted_as_dropped():
    q = "evicts the tensor that is stalest, largest"
    got, dropped = validate_citations(
        [{"source": 1, "quote": q}, {"source": 1, "quote": q.upper()}], [DTR]
    )
    assert len(got) == 1 and dropped == 0


def test_non_list_citations_are_ignored():
    assert validate_citations(None, [DTR]) == ([], 0)
    assert validate_citations("x", [DTR]) == ([], 0)


def test_refusal_keeps_the_models_reason():
    resp = build_response(
        [DTR],
        gen(can_answer=False, refusal_reason="The sources do not mention Capuchin."),
        PRICES,
    )
    assert resp.status is AnswerStatus.REFUSED
    assert resp.refusal_reason == "The sources do not mention Capuchin."


def test_refusal_without_reason_gets_a_default():
    resp = build_response([DTR], gen(can_answer=False), PRICES)
    assert resp.status is AnswerStatus.REFUSED and resp.refusal_reason


def test_empty_tool_call_is_a_refusal():
    resp = build_response([DTR], Generation(data={}, input_tokens=5, output_tokens=0), PRICES)
    assert resp.status is AnswerStatus.REFUSED


def test_answer_without_text_is_refused_even_with_valid_citation():
    resp = build_response(
        [DTR],
        gen(
            can_answer=True,
            answer="  ",
            citations=[{"source": 1, "quote": "evicts the tensor that is stalest"}],
        ),
        PRICES,
    )
    assert resp.status is AnswerStatus.REFUSED


def test_cost_is_computed_from_token_prices():
    resp = build_response([DTR], gen(can_answer=False), PRICES)
    assert resp.input_tokens == 1000 and resp.output_tokens == 200
    assert resp.estimated_cost_usd == pytest.approx(0.002)


def test_user_message_numbers_sources_and_names_papers():
    msg = render_user_message("What does DTR evict?", [DTR, CHK], {"dtr2021": "DTR"})
    assert msg.startswith("Question: What does DTR evict?")
    assert "[1] DTR (dtr2021), 3 Dynamic Tensor Rematerialization, page 3" in msg
    assert "[2] checkmate2020 (checkmate2020), 4 Optimal rematerialization, page 5" in msg


class FakeRetriever:
    name = "fake"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.asked: list[tuple[str, int]] = []

    def retrieve(self, question: str, k: int) -> list[Chunk]:
        self.asked.append((question, k))
        return self.chunks[:k]


class FakeClient:
    def __init__(self, **data: Any) -> None:
        self.data = data
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str, tool: dict[str, Any], max_tokens: int) -> Generation:
        self.calls.append((system, user))
        return Generation(data=self.data, input_tokens=300, output_tokens=40)


def make_pipeline(retriever: FakeRetriever, client: FakeClient) -> Pipeline:
    return Pipeline(retriever, client, {"dtr2021": "DTR"}, Settings(top_k=2))


def test_pipeline_answers_and_reports_timings():
    client = FakeClient(
        can_answer=True,
        answer="Stale, large, cheap tensors.",
        citations=[{"source": 1, "quote": "evicts the tensor that is stalest, largest"}],
    )
    retriever = FakeRetriever([DTR, CHK, DTR])
    resp = make_pipeline(retriever, client).ask("What does DTR evict?")
    assert resp.status is AnswerStatus.ANSWERED
    assert retriever.asked == [("What does DTR evict?", 2)]  # settings.top_k
    assert resp.latency_ms is not None and resp.retrieval_ms is not None
    assert resp.latency_ms >= resp.retrieval_ms >= 0
    assert "Sources:" in client.calls[0][1]


def test_pipeline_top_k_override():
    retriever = FakeRetriever([DTR, CHK])
    make_pipeline(retriever, FakeClient(can_answer=False)).ask("a question", top_k=1)
    assert retriever.asked[0][1] == 1


def test_pipeline_refuses_without_calling_the_model_when_nothing_is_retrieved():
    client = FakeClient(can_answer=True)
    resp = make_pipeline(FakeRetriever([]), client).ask("anything at all")
    assert resp.status is AnswerStatus.REFUSED
    assert client.calls == []


class _Block:
    def __init__(self, type: str, name: str = "", input: dict[str, Any] | None = None) -> None:
        self.type, self.name, self.input = type, name, input or {}


class _Message:
    def __init__(self, content: list[_Block]) -> None:
        self.content = content
        self.usage = type("U", (), {"input_tokens": 321, "output_tokens": 45})()


class _Messages:
    def __init__(self, reply: _Message) -> None:
        self.reply = reply
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _Message:
        self.kwargs = kwargs
        return self.reply


def _client_with(reply: _Message, **kw: Any) -> tuple[AnthropicClient, _Messages]:
    client = AnthropicClient("some-model", api_key="test-key", **kw)
    messages = _Messages(reply)
    client._client = type("C", (), {"messages": messages})()
    return client, messages


def test_anthropic_client_forces_the_tool_and_parses_its_arguments():
    reply = _Message([_Block("text"), _Block("tool_use", "submit_answer", {"can_answer": False})])
    client, messages = _client_with(reply)
    got = client.generate("sys", "user msg", TOOL_SCHEMA, 500)
    assert got == Generation({"can_answer": False}, 321, 45)
    k = messages.kwargs
    assert k["model"] == "some-model" and k["max_tokens"] == 500
    assert k["system"] == "sys" and k["messages"] == [{"role": "user", "content": "user msg"}]
    assert k["tool_choice"] == {"type": "tool", "name": "submit_answer"}
    assert k["tools"] == [TOOL_SCHEMA] and k["temperature"] == 0.0


def test_anthropic_client_can_leave_temperature_unset():
    client, messages = _client_with(_Message([]), temperature=None)
    assert client.generate("s", "u", TOOL_SCHEMA, 10).data == {}  # no tool call -> empty
    assert "temperature" not in messages.kwargs
