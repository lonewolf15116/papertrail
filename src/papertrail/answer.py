"""Grounded answer generation: prompt, structured output, and citation validation.

The model sees numbered sources and must return either an answer whose every claim carries a
citation (source number + a verbatim quote) or an explicit refusal. The quote is the contract:
a citation only survives if the quote really appears in the cited chunk, so a model cannot cite
a passage it did not use. An answer left with no verifiable citation becomes a refusal.
"""

from __future__ import annotations

import json
import re
import unicodedata
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
4a. Check the specifics of the question against the sources before answering. If the question \
names a particular model, dataset, hardware, metric or setting (for example a GPU type, a model \
size, an accelerator, a benchmark) and the sources do not report that exact thing, refuse. Do not \
answer with a similar thing from the sources instead: a result for a different GPU, model, \
dataset or metric does not answer the question, even if it is closely related.
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
    truncated: bool = False  # the reply hit the max-token limit, so its arguments may be cut off


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
            truncated=getattr(msg, "stop_reason", None) == "max_tokens",
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
            truncated=getattr(resp.choices[0], "finish_reason", None) == "length",
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


_MATH_SYMBOLS = {
    "√": "sqrt",
    "∞": "infty",
    "×": "times",
    "≤": "leq",
    "≥": "geq",
    "≈": "approx",
    "∑": "sum",
    "∏": "prod",
    "·": "cdot",
    "⋅": "cdot",
}
# LaTeX that only changes how text looks; PDF text extraction never contains it.
_LATEX_STYLE = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|mathcal|operatorname|left|right|big|Big)(?![a-zA-Z])"
)


def squash_math(text: str) -> str:
    """Like `squash`, but also treats LaTeX and the Unicode symbols a PDF extracts for it alike.

    A model often writes `B = \\Omega(\\sqrt{N})` for a passage the PDF extracted as `B = Ω(√N)`;
    plain `squash` turns those into different strings, so a correct quote is dropped. Both sides are
    mapped to the same ASCII names (`omega`, `sqrt`, ...). Used only to verify citations, never to
    resolve gold labels, so the retrieval metrics and the CI gate are unaffected.
    """
    t = unicodedata.normalize("NFKC", text)
    t = _LATEX_STYLE.sub("", t)
    t = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", t)
    t = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", t)
    out: list[str] = []
    for ch in t:
        if ch in _MATH_SYMBOLS:
            out.append(_MATH_SYMBOLS[ch])
        elif unicodedata.name(ch, "").startswith("GREEK "):
            out.append(unicodedata.name(ch).split()[-1].lower())  # GREEK CAPITAL LETTER OMEGA
        else:
            out.append(ch)
    return squash("".join(out))


_ELLIPSIS = re.compile(r"\.{3,}|\u2026")


def quote_in_chunk(quote: str, chunk: Chunk) -> bool:
    """Whether `quote` appears in the chunk, ignoring case, spacing, hyphenation, punctuation
    and the LaTeX-versus-Unicode spelling of math.

    A quote may join passages with an ellipsis ("A ... B"), which models do constantly. Each
    fragment long enough to verify anything must then appear verbatim in the chunk, in order;
    fragments shorter than MIN_QUOTE_CHARS carry no weight, and a quote with no long fragment
    is rejected, so an ellipsis cannot be used to smuggle in unverified text of any length.
    """
    fragments = [squash_math(f) for f in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if len(f) >= MIN_QUOTE_CHARS]
    if not fragments:
        return False
    haystack = squash_math(chunk.text)
    pos = 0
    for f in fragments:
        i = haystack.find(f, pos)
        if i < 0:
            return False
        pos = i + len(f)
    return True


def check_citations(raw: object, chunks: Sequence[Chunk]) -> tuple[list[Citation], list[str]]:
    """Keep citations whose source number is valid and whose quote is really in that chunk.

    Returns (valid citations, the quotes dropped). Duplicates of the same (source, quote) are
    merged silently, not counted as dropped.
    """
    if not isinstance(raw, list):
        return [], []
    valid: list[Citation] = []
    seen: set[tuple[int, str]] = set()
    dropped: list[str] = []
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
            dropped.append(quote[:200] if isinstance(quote, str) else repr(item)[:200])
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


def validate_citations(raw: object, chunks: Sequence[Chunk]) -> tuple[list[Citation], int]:
    """`check_citations`, reporting only how many citations were dropped."""
    valid, dropped = check_citations(raw, chunks)
    return valid, len(dropped)


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
    citations, dropped_quotes = check_citations(data.get("citations"), chunks)
    dropped = len(dropped_quotes)

    if data.get("can_answer") is True and answer and citations:
        return AskResponse(
            status=AnswerStatus.ANSWERED,
            answer=answer,
            citations=citations,
            dropped_citations=dropped,
            dropped_quotes=dropped_quotes,
            **common,
        )

    reason = data.get("refusal_reason")
    if data.get("can_answer") is True:
        reason = (
            "The model drafted an answer but none of its citations could be verified "
            "against the retrieved passages, so it was withheld."
        )
    elif generation.truncated and not data:
        reason = "The model's reply was cut off at the token limit, so no answer was produced."
    elif not isinstance(reason, str) or not reason.strip():
        reason = "The retrieved passages do not support an answer."
    return AskResponse(
        status=AnswerStatus.REFUSED,
        refusal_reason=reason.strip(),
        dropped_citations=dropped,
        dropped_quotes=dropped_quotes,
        **common,
    )
