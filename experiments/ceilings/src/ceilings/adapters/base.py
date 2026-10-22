"""ModelAdapter contract (contracts/interfaces.md) shared by every model family."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

DEPTHS = ("low", "medium", "high", "xhigh", "max")
PROB_FLOOR = 1e-6


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


def answer_schema(answer_space: list[str]) -> dict:
    """Structured-output JSON schema: one chosen answer plus a probability per answer."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "probabilities"],
        "properties": {
            "answer": {"type": "string", "enum": list(answer_space)},
            "probabilities": {
                "type": "object",
                "additionalProperties": False,
                "required": list(answer_space),
                "properties": {a: {"type": "number"} for a in answer_space},
            },
        },
    }


def normalize_probabilities(
    raw: dict[str, float] | None, answer_space: list[str]
) -> dict[str, float]:
    """Clip to [PROB_FLOOR, 1] over exactly `answer_space`, then renormalize; uniform if missing."""
    if not raw:
        return {a: 1 / len(answer_space) for a in answer_space}
    clipped = {a: min(1.0, max(PROB_FLOOR, float(raw.get(a, 0.0) or 0.0))) for a in answer_space}
    total = sum(clipped.values())
    return {a: v / total for a, v in clipped.items()}


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
