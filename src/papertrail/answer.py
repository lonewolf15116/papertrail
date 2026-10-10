"""Grounded answer generation: prompt, structured output, and citation validation.

The model sees numbered sources and must return either an answer whose every claim carries a
citation (source number + a verbatim quote) or an explicit refusal. The quote is the contract:
a citation only survives if the quote really appears in the cited chunk, so a model cannot cite
a passage it did not use. An answer left with no verifiable citation becomes a refusal.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from papertrail.eval.evidence import squash
from papertrail.schemas import AnswerStatus, AskResponse, Chunk, Citation

TOOL_NAME = "submit_answer"
MIN_QUOTE_CHARS = 12  # alphanumeric characters; shorter quotes verify nothing

SYSTEM_PROMPT = """\
You answer questions about machine-learning systems research papers using ONLY the numbered \
sources in the user message.

Rules:
1. Use only what the sources say. Do not add outside knowledge, even if you know the answer.
2. Every claim in your answer must be backed by a citation: the number of the source and a quote \
copied exactly (verbatim, at most 40 words) from that source. Cite each source you rely on.
3. Papers often use the same words for different things (DTR, Checkmate and Capuchin all discuss \
rematerialization). Attribute each claim to the paper it comes from and never carry a claim from \
one paper over to another. If the question names a paper or system, use only that paper's sources.
4. If the sources do not support an answer to the question, refuse: set can_answer to false and \
say in one sentence what is missing. This includes the case where the sources are about a \
different paper or system than the one asked about. If the sources support only part of the \
question, answer only that part and say what they do not cover. Never guess.
5. Be concise: one to four sentences, in your own words, no preamble.
Respond only by calling the submit_answer tool."""

TOOL_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": "Submit a cited answer, or a refusal when the sources do not support one.",
    "input_schema": {
        "type": "object",
        "properties": {
            "can_answer": {
                "type": "boolean",
                "description": "True only if the sources support an answer to the question.",
            },
            "answer": {
                "type": "string",
                "description": "The answer, grounded in the sources. Empty when refusing.",
            },
            "citations": {
                "type": "array",
                "description": "Evidence for the answer. At least one when can_answer is true.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "integer", "description": "Source number, from 1."},
                        "quote": {
                            "type": "string",
                            "description": "Exact words copied from that source.",
                        },
                    },
                    "required": ["source", "quote"],
                },
            },
            "refusal_reason": {
                "type": "string",
                "description": "When can_answer is false: what the sources are missing.",
            },
        },
        "required": ["can_answer"],
    },
}


@dataclass(frozen=True)
class Generation:
    """What an LLM backend returns: the tool-call arguments plus token usage."""

    data: dict[str, Any]
    input_tokens: int
    output_tokens: int


class AnswerClient(Protocol):
    """Anything that can produce a Generation: Anthropic, OpenAI, or a fake in tests."""

    def generate(
        self, system: str, user: str, tool: dict[str, Any], max_tokens: int
    ) -> Generation: ...


class AnthropicClient:
    """Forced tool call against the Anthropic Messages API (structured output)."""

    def __init__(
        self, model: str, api_key: str | None = None, temperature: float | None = 0.0
    ) -> None:
        import anthropic

        self.model = model
        self.temperature = temperature  # None leaves the API default (some models fix sampling)
        self._client: Any = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def generate(self, system: str, user: str, tool: dict[str, Any], max_tokens: int) -> Generation:
        options: dict[str, Any] = (
            {} if self.temperature is None else {"temperature": self.temperature}
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            **options,
        )
        data: dict[str, Any] = {}
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                data = dict(block.input)
                break
        return Generation(
            data=data,
            input_tokens=int(msg.usage.input_tokens),
            output_tokens=int(msg.usage.output_tokens),
        )


class OpenAIClient:
    """Forced function call against the OpenAI Chat Completions API (structured output)."""

    def __init__(
        self, model: str, api_key: str | None = None, temperature: float | None = 0.0
    ) -> None:
        import openai

        self.model = model
        self.temperature = temperature  # None leaves the API default (some models fix sampling)
        self._client: Any = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def generate(self, system: str, user: str, tool: dict[str, Any], max_tokens: int) -> Generation:
        # The tool schema is written in Anthropic's shape; OpenAI wants the same JSON Schema
        # under "function.parameters".
        function = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        options: dict[str, Any] = (
            {} if self.temperature is None else {"temperature": self.temperature}
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[{"type": "function", "function": function}],
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
            **options,
        )
        data: dict[str, Any] = {}
        calls = getattr(resp.choices[0].message, "tool_calls", None) or []
        for call in calls:
            if call.function.name == tool["name"]:
                try:
                    parsed = json.loads(call.function.arguments)
                except (TypeError, ValueError):
                    parsed = None  # truncated or malformed arguments: treated as a refusal
                if isinstance(parsed, dict):
                    data = parsed
                break
        usage = resp.usage
        return Generation(
            data=data,
            input_tokens=int(usage.prompt_tokens),
            output_tokens=int(usage.completion_tokens),
        )


def make_client(
    provider: str, model: str, api_key: str | None = None, temperature: float | None = 0.0
) -> AnswerClient:
    """Build the LLM client for a provider name ("anthropic" or "openai")."""
    if provider == "anthropic":
        return AnthropicClient(model, api_key, temperature)
    if provider == "openai":
        return OpenAIClient(model, api_key, temperature)
    raise ValueError(f"unknown LLM provider {provider!r}")


def render_user_message(question: str, chunks: Sequence[Chunk], titles: dict[str, str]) -> str:
    parts = [f"Question: {question}", "", "Sources:"]
    for i, c in enumerate(chunks, start=1):
        title = titles.get(c.paper_id, c.paper_id)
        parts.append(f"[{i}] {title} ({c.paper_id}), {c.section}, page {c.page}")
        parts.append(c.text.strip())
        parts.append("")
    return "\n".join(parts)


def quote_in_chunk(quote: str, chunk: Chunk) -> bool:
    """Whether `quote` appears in the chunk, ignoring case, spacing, hyphenation and punctuation."""
    q = squash(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in squash(chunk.text)


def validate_citations(raw: object, chunks: Sequence[Chunk]) -> tuple[list[Citation], int]:
    """Keep citations whose source number is valid and whose quote is really in that chunk.

    Returns (valid citations, number dropped). Duplicates of the same (source, quote) are merged
    silently, not counted as dropped.
    """
    if not isinstance(raw, list):
        return [], 0
    valid: list[Citation] = []
    seen: set[tuple[int, str]] = set()
    dropped = 0
    for item in raw:
        source = item.get("source") if isinstance(item, dict) else None
        quote = item.get("quote") if isinstance(item, dict) else None
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or not isinstance(quote, str)
            or not 1 <= source <= len(chunks)
            or not quote_in_chunk(quote, chunks[source - 1])
        ):
            dropped += 1
            continue
        key = (source, squash(quote))
        if key in seen:
            continue
        seen.add(key)
        c = chunks[source - 1]
        valid.append(
            Citation(
                paper_id=c.paper_id,
                section=c.section,
                page=c.page,
                chunk_id=c.chunk_id,
                quote=quote.strip(),
            )
        )
    return valid, dropped


def build_response(
    chunks: Sequence[Chunk], generation: Generation, cost_per_mtok: tuple[float, float]
) -> AskResponse:
    """Turn the model's tool call into a validated AskResponse (answered or refused)."""
    data = generation.data
    in_price, out_price = cost_per_mtok
    common: dict[str, Any] = {
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "estimated_cost_usd": (
            generation.input_tokens * in_price + generation.output_tokens * out_price
        )
        / 1_000_000,
        "retrieved_chunk_ids": [c.chunk_id for c in chunks],
    }
    answer = data.get("answer")
    answer = answer.strip() if isinstance(answer, str) else ""
    citations, dropped = validate_citations(data.get("citations"), chunks)

    if data.get("can_answer") is True and answer and citations:
        return AskResponse(
            status=AnswerStatus.ANSWERED,
            answer=answer,
            citations=citations,
            dropped_citations=dropped,
            **common,
        )

    reason = data.get("refusal_reason")
    if data.get("can_answer") is True:
        reason = (
            "The model drafted an answer but none of its citations could be verified "
            "against the retrieved passages, so it was withheld."
        )
    elif not isinstance(reason, str) or not reason.strip():
        reason = "The retrieved passages do not support an answer."
    return AskResponse(
        status=AnswerStatus.REFUSED,
        refusal_reason=reason.strip(),
        dropped_citations=dropped,
        **common,
    )
