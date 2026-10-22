"""ModelAdapter contract (contracts/interfaces.md) shared by every model family."""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field

DEPTHS = ("low", "medium", "high", "xhigh", "max")
PROB_FLOOR = 1e-6

# Models without a native reasoning-effort setting get depth as an explicit reasoning stage:
# the model first writes free-form reasoning capped at this many tokens, then answers. `low`
# answers directly. Claude uses its native `effort` instead.
REASONING_TOKENS = {"low": 0, "medium": 256, "high": 1024, "xhigh": 2048, "max": 4096}


def reasoning_tokens(depth: str) -> int:
    """Reasoning-stage cap for a depth. `CEILINGS_REASONING_TOKENS="0,64,128,256,512"` overrides
    the five levels (low..max), e.g. to keep local test runs short and cool."""
    import os

    env = os.environ.get("CEILINGS_REASONING_TOKENS")
    if env:
        values = [int(v) for v in env.split(",")]
        return dict(zip(DEPTHS, values, strict=True))[depth]
    return REASONING_TOKENS[depth]
REASON_PROMPT = (
    "Do not give your final answer yet. First think it through step by step and write out only "
    "your reasoning. You will be asked for the final answer in the next turn."
)
FINAL_ANSWER_PROMPT = "Now give your final answer in the required format."
FINAL_TEXT_PROMPT = "Now write only the requested reply, with nothing else."
TRUNCATED_NOTE = "(Your reasoning was cut off at its length limit.) "
ANSWER_RULES = (
    "Give a probability for every option; the probabilities must add up to 1. Your answer must be "
    "the option you gave the highest probability."
)


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None
    dollars: float = 0.0


@dataclass
class AdapterResult:
    final_answer: str | None
    probabilities: dict[str, float]
    refusal: dict | None
    usage: Usage
    request_id: str
    text: str = ""
    valid: bool = True
    extra: dict = field(default_factory=dict)


def option_order(answer_space: list[str], seed: int | None) -> list[str]:
    """The options in a seeded random order, so a model's position bias cannot line up with the
    truth. `None` keeps the given order."""
    if seed is None:
        return list(answer_space)
    import random

    order = list(answer_space)
    random.Random(seed).shuffle(order)
    return order


def answer_schema(answer_space: list[str], seed: int | None = None) -> dict:
    """Structured-output JSON schema: a probability for every option first, then the answer.

    Probabilities come first because constrained decoders emit properties in schema order; asking
    for the answer first makes the model commit before weighing the options. With a seed, the
    options are listed in a shuffled order.
    """
    order = option_order(answer_space, seed)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["probabilities", "answer"],
        "properties": {
            "probabilities": {
                "type": "object",
                "additionalProperties": False,
                "required": order,
                "properties": {a: {"type": "number"} for a in order},
            },
            "answer": {"type": "string", "enum": order},
        },
    }


def probability_diagnostics(text: str, answer: str | None, probs: dict[str, float],
                            answer_space: list[str]) -> dict:
    """How far the raw probabilities were from summing to 1, and whether the stated answer is the
    most probable option (both measured before the harness renormalizes)."""
    raw_sum = None
    try:
        raw = json.loads(text).get("probabilities")
        raw_sum = float(sum(float(raw.get(a, 0) or 0) for a in answer_space))
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        pass
    top = max(probs.values()) if probs else 0.0
    mismatch = answer is not None and probs.get(answer, 0.0) < top - 1e-9
    return {"raw_prob_sum": raw_sum, "answer_not_argmax": bool(mismatch)}


def normalize_probabilities(
    raw: dict[str, float] | None, answer_space: list[str]
) -> dict[str, float]:
    """Clip to [PROB_FLOOR, 1] over exactly `answer_space`, then renormalize; uniform if missing."""
    if not raw:
        return {a: 1 / len(answer_space) for a in answer_space}
    clipped = {}
    for a in answer_space:
        try:
            v = float(raw.get(a, 0.0) or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        clipped[a] = min(1.0, max(PROB_FLOOR, v))
    total = sum(clipped.values())
    return {a: v / total for a, v in clipped.items()}


def parse_answer(text: str, answer_space: list[str]) -> tuple[str | None, dict[str, float], str]:
    """Parse structured output; fall back to the first option named in free text.

    Returns (answer or None, normalized probabilities, how it was parsed: json|text|failed).
    """
    try:
        data = json.loads(text)
        answer = data.get("answer") if isinstance(data, dict) else None
        probs = data.get("probabilities") if isinstance(data, dict) else None
        if answer in answer_space:
            return answer, normalize_probabilities(probs, answer_space), "json"
        if isinstance(probs, dict) and probs:
            norm = normalize_probabilities(probs, answer_space)
            return max(norm, key=norm.get), norm, "json_argmax"
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    for a in sorted(answer_space, key=len, reverse=True):
        if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text or ""):
            rest = (1 - 0.6) / max(1, len(answer_space) - 1)
            return a, {o: (0.6 if o == a else rest) for o in answer_space}, "text"
    return None, normalize_probabilities(None, answer_space), "failed"


def approx_tokens(text: str) -> int:
    """Provider-free token estimate (about 4 characters per token)."""
    return 0 if not text else max(1, (len(text) + 3) // 4)


class ModelAdapter(abc.ABC):
    """One model family behind a uniform call."""

    model_id: str
    supports_batch: bool = False

    @abc.abstractmethod
    def complete(
        self,
        messages: list[dict],
        depth: str,
        answer_space: list[str] | None,
        seed: int,
        max_tokens: int = 16000,
    ) -> AdapterResult:
        """Answer (answer_space given) or produce free text (answer_space None)."""

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)


@dataclass
class ChatReply:
    text: str
    input_tokens: int
    output_tokens: int
    request_id: str
    model_served: str
    stop: str = ""


class ChatAdapter(ModelAdapter):
    """Base for chat APIs without native reasoning effort: depth becomes a reasoning stage."""

    answer_max_tokens = 400
    text_max_tokens = 300

    @abc.abstractmethod
    def _chat(self, messages: list[dict], schema: dict | None, max_tokens: int,
              seed: int) -> ChatReply:
        """One chat call; `schema` constrains the output to JSON when given."""

    def complete(self, messages, depth, answer_space, seed, max_tokens=16000):
        budget = reasoning_tokens(depth)
        convo = [dict(m) for m in messages]
        in_tok = out_tok = 0
        reasoning = ""
        reasoning_stop = ""
        ids = []
        served = set()
        sent: list[list[dict]] = []
        if budget > 0:
            convo[-1]["content"] = f"{convo[-1]['content']}\n\n{REASON_PROMPT}"
            sent.append([dict(m) for m in convo])
            r = self._chat(convo, None, min(budget, max_tokens), seed)
            in_tok, out_tok = in_tok + r.input_tokens, out_tok + r.output_tokens
            reasoning = r.text
            reasoning_stop = r.stop
            ids.append(r.request_id)
            served.add(r.model_served)
            final = FINAL_ANSWER_PROMPT if answer_space else FINAL_TEXT_PROMPT
            if r.stop == "length":
                final = TRUNCATED_NOTE + final
            convo = convo + [{"role": "assistant", "content": r.text},
                             {"role": "user", "content": final}]
        schema = answer_schema(answer_space, seed) if answer_space else None
        cap = self.answer_max_tokens if answer_space else self.text_max_tokens
        sent.append([dict(m) for m in convo])
        reply = self._chat(convo, schema, cap, seed + 1)
        in_tok, out_tok = in_tok + reply.input_tokens, out_tok + reply.output_tokens
        ids.append(reply.request_id)
        served.add(reply.model_served)
        valid = all(_same_model(s, self.model_name) for s in served if s)
        usage = Usage(in_tok, out_tok, reasoning_tokens=None)
        extra = {"raw": reply.text, "reasoning": reasoning, "reasoning_stop": reasoning_stop,
                 "stop": reply.stop, "sent": sent}
        if answer_space is None:
            return AdapterResult(None, {}, None, usage, ";".join(ids), text=reply.text,
                                 valid=valid, extra=extra)
        answer, probs, how = parse_answer(reply.text, answer_space)
        extra["parse"] = how
        extra.update(probability_diagnostics(reply.text, answer, probs, answer_space))
        return AdapterResult(answer, probs, None, usage, ";".join(ids), text=reply.text,
                             valid=valid, extra=extra)

    model_name: str = ""


def _same_model(served: str, requested: str) -> bool:
    """Providers may append a tag or date (e.g. 'qwen2.5:1.5b' vs 'qwen2.5:1.5b-instruct')."""
    return served == requested or served.startswith(requested) or requested.startswith(served)
