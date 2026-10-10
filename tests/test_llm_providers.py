"""The OpenAI client, provider settings and the client factory (no network: fakes only)."""

import json
from typing import Any

import pytest

from papertrail.answer import (
    TOOL_SCHEMA,
    AnthropicClient,
    Generation,
    OpenAIClient,
    make_client,
)
from papertrail.config import PROVIDERS, Settings


class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.function = _Function(name, arguments)


class _Reply:
    def __init__(self, tool_calls: list[_ToolCall] | None) -> None:
        message = type("M", (), {"tool_calls": tool_calls})()
        self.choices = [type("Ch", (), {"message": message})()]
        self.usage = type("U", (), {"prompt_tokens": 210, "completion_tokens": 33})()


class _Completions:
    def __init__(self, reply: _Reply) -> None:
        self.reply = reply
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _Reply:
        self.kwargs = kwargs
        return self.reply


def _openai_with(reply: _Reply, **kw: Any) -> tuple[OpenAIClient, _Completions]:
    client = OpenAIClient("some-model", api_key="test-key", **kw)
    completions = _Completions(reply)
    chat = type("Chat", (), {"completions": completions})()
    client._client = type("C", (), {"chat": chat})()
    return client, completions


def test_openai_client_forces_the_function_and_parses_its_arguments():
    args = json.dumps({"can_answer": True, "answer": "x", "citations": []})
    client, completions = _openai_with(_Reply([_ToolCall("submit_answer", args)]))
    got = client.generate("sys", "user msg", TOOL_SCHEMA, 500)
    assert got == Generation(json.loads(args), 210, 33)
    k = completions.kwargs
    assert k["model"] == "some-model" and k["max_completion_tokens"] == 500
    assert k["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user msg"},
    ]
    assert k["tool_choice"] == {"type": "function", "function": {"name": "submit_answer"}}
    (tool,) = k["tools"]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "submit_answer"
    assert tool["function"]["parameters"] == TOOL_SCHEMA["input_schema"]
    assert k["temperature"] == 0.0


def test_openai_client_can_leave_temperature_unset():
    client, completions = _openai_with(_Reply(None), temperature=None)
    assert client.generate("s", "u", TOOL_SCHEMA, 10).data == {}
    assert "temperature" not in completions.kwargs


@pytest.mark.parametrize(
    "calls",
    [
        None,
        [],
        [_ToolCall("some_other_tool", '{"can_answer": true}')],
        [_ToolCall("submit_answer", '{"can_answer": tru')],  # truncated JSON
        [_ToolCall("submit_answer", "[1, 2]")],  # valid JSON, wrong shape
    ],
)
def test_openai_client_returns_empty_data_when_the_call_is_unusable(calls):
    client, _ = _openai_with(_Reply(calls))
    assert client.generate("s", "u", TOOL_SCHEMA, 10).data == {}


def test_make_client_picks_the_provider():
    assert isinstance(make_client("anthropic", "m", api_key="k"), AnthropicClient)
    assert isinstance(make_client("openai", "m", api_key="k"), OpenAIClient)
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_client("nope", "m")


def test_default_settings_are_unchanged_for_anthropic():
    s = Settings(_env_file=None)
    answer, judge = s.answer_llm(), s.judge_llm()
    assert (answer.provider, answer.model, answer.key_env) == (
        "anthropic",
        "claude-haiku-4-5-20251001",
        "ANTHROPIC_API_KEY",
    )
    assert (judge.provider, judge.model) == ("anthropic", "claude-sonnet-5-5")
    assert (answer.input_price_per_mtok, answer.output_price_per_mtok) == (1.0, 5.0)


def test_openai_provider_uses_openai_defaults_for_answer_and_judge():
    s = Settings(_env_file=None, llm_provider="openai")
    answer, judge = s.answer_llm(), s.judge_llm()
    assert answer.provider == judge.provider == "openai"
    assert answer.key_env == judge.key_env == "OPENAI_API_KEY"
    assert answer.model != judge.model  # the judge is never the answerer by default


def test_model_and_price_overrides_beat_provider_defaults():
    s = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_model="my-model",
        llm_input_price_per_mtok=0.0,
        judge_provider="anthropic",
        judge_model="my-judge",
    )
    answer, judge = s.answer_llm(), s.judge_llm()
    assert answer.model == "my-model" and answer.input_price_per_mtok == 0.0
    assert answer.output_price_per_mtok == PROVIDERS["openai"].answer[2]
    assert (judge.provider, judge.model, judge.key_env) == (
        "anthropic",
        "my-judge",
        "ANTHROPIC_API_KEY",
    )


def test_unknown_provider_is_rejected_with_the_choices():
    with pytest.raises(ValueError, match="anthropic"):
        Settings(_env_file=None, llm_provider="nope").answer_llm()


def test_a_reply_cut_off_at_the_token_limit_is_reported_as_such_not_as_unsupported():
    from papertrail.answer import build_response
    from papertrail.schemas import AnswerStatus

    resp = build_response([], Generation({}, 100, 600, truncated=True), (0.0, 0.0))
    assert resp.status is AnswerStatus.REFUSED
    assert "token limit" in (resp.refusal_reason or "")


def test_openai_client_flags_truncation():
    reply = _Reply(None)
    reply.choices[0].finish_reason = "length"
    client, _ = _openai_with(reply)
    assert client.generate("s", "u", TOOL_SCHEMA, 10).truncated is True


def test_prompt_tells_the_model_to_refuse_near_miss_questions():
    # Guards against the rule being edited out; it does not prove the model obeys it. Behaviour
    # is measured by the unanswerable questions in the answer evaluation (unans-005/006/009/010
    # were answered by swapping H100->V100, TPU->GPU, GPT-3->ResNet-50, transformers->VGG-16).
    from papertrail.answer import SYSTEM_PROMPT

    assert "exact thing" in SYSTEM_PROMPT and "similar thing" in SYSTEM_PROMPT


def _chunk(text: str):
    from papertrail.schemas import Chunk

    return Chunk(chunk_id="p:1", paper_id="p", section="s", page=1, text=text)


@pytest.mark.parametrize(
    "quote, passage",
    [
        (
            r"a memory budget of B = \Omega(\sqrt{N}) to train",
            "DTR needs a memory budget of B = Ω(√N) to train a network.",
        ),
        (r"memory \le B \leq 2B", "We require memory ≤ B ≤ 2B at all times in the schedule."),
        (
            r"\text{cost} \times \mathrm{size}",
            "The heuristic multiplies cost × size for each tensor.",
        ),
        (
            r"combined as \alpha + \beta before normalising",
            "the weights are combined as α + β before normalising",
        ),
    ],
)
def test_quote_check_treats_latex_and_extracted_unicode_math_alike(quote, passage):
    from papertrail.answer import quote_in_chunk

    assert quote_in_chunk(quote, _chunk(passage))


def test_quote_check_still_rejects_a_different_formula():
    from papertrail.answer import quote_in_chunk

    chunk = _chunk("DTR needs a memory budget of B = Ω(√N) to train a network.")
    assert not quote_in_chunk(r"B = \Omega(N^2)", chunk)  # different math, must not match
    assert not quote_in_chunk(r"\Omega", chunk)  # too short to verify anything


def test_dropped_quotes_are_returned_for_debugging():
    from papertrail.answer import build_response
    from papertrail.schemas import AnswerStatus

    chunk = _chunk("DTR evicts the tensor that is stalest, largest, and cheapest to rematerialize.")
    gen = Generation(
        {
            "can_answer": True,
            "answer": "x",
            "citations": [
                {"source": 1, "quote": "evicts the tensor that is stalest, largest"},
                {"source": 1, "quote": "this sentence is not in the passage at all"},
            ],
        },
        10,
        10,
    )
    resp = build_response([chunk], gen, (0.0, 0.0))
    assert resp.status is AnswerStatus.ANSWERED
    assert resp.dropped_citations == 1
    assert resp.dropped_quotes == ["this sentence is not in the passage at all"]
