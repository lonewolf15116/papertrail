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
