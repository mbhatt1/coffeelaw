"""Agent policies: what one agent does with the slice of the record it can see.

Two implementations share one interface:

- `SimPolicy` ("sim-bounded"): a simulated agent with an explicit step budget per depth level.
  It is not a language model. It exists so the whole pipeline can be exercised for free, and so
  the analysis can be checked against a system whose ceilings are known exactly. It reads only
  the structured mirror of the text its agent is shown (`View.public` / `View.private` / the
  record), never the hidden answer.
- `LLMPolicy`: renders the same view as chat messages and calls a `ModelAdapter`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ceilings.adapters.base import (
    AdapterResult,
    ModelAdapter,
    Usage,
    approx_tokens,
    normalize_probabilities,
)
from ceilings.records import content_hash

STEP_BUDGET = {"low": 4, "medium": 8, "high": 16, "xhigh": 32, "max": 64}
CONFIDENT = 0.98
PROGRESS_RE = re.compile(r"PROGRESS (\S+) (\d) (\d+)")


@dataclass
class View:
    """Everything one agent can see at one moment."""

    name: str
    family_id: str
    scenario: str
    shared_text: str
    private_text: str
    public: dict
    private: dict
    answer_space: list[str]
    tool_results: list[tuple[object, str]] = field(default_factory=list)
    inbox: list[str] = field(default_factory=list)
    extra_text: str = ""

    def render(self) -> str:
        parts = [self.scenario, self.shared_text]
        if self.private_text:
            parts.append(f"Only you (agent {self.name}) can see this:\n{self.private_text}")
        if self.tool_results:
            parts.append(
                "\n".join(
                    f"Probe result: bit {q} of the failing host's ID reads {v}."
                    for q, v in self.tool_results
                )
            )
        if self.inbox:
            parts.append("Messages you received:\n" + "\n".join(self.inbox))
        if self.extra_text:
            parts.append(self.extra_text)
        return "\n\n".join(p for p in parts if p)


class Policy:
    model_id: str

    def choose_probe(self, view: View, depth: str, seed: int) -> tuple[object, Usage]:
        raise NotImplementedError

    def compose(
        self, view: View, depth: str, mode: str, budget: int, retry: str | None, seed: int
    ) -> tuple[str, Usage]:
        raise NotImplementedError

    def answer(self, view: View, depth: str, seed: int) -> AdapterResult:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return approx_tokens(text)


# --- Simulated bounded agent -------------------------------------------------------------------


def _confident(answer: str, space: list[str]) -> dict[str, float]:
    rest = (1 - CONFIDENT) / (len(space) - 1)
    return {a: (CONFIDENT if a == answer else rest) for a in space}


def _uniform(space: list[str]) -> dict[str, float]:
    return {a: 1 / len(space) for a in space}


class SimPolicy(Policy):
    """Bounded agent: solves exactly what its step budget and its view allow, otherwise guesses."""

    model_id = "sim-bounded"

    def _usage(self, view: View, budget_steps: int, extra_out: int = 0) -> Usage:
        return Usage(
            input_tokens=approx_tokens(view.render()),
            output_tokens=10 * budget_steps + 20 + extra_out,
        )

    def choose_probe(self, view, depth, seed):
        k = view.public["k"]
        counts = [0] * k
        for q, _ in view.tool_results:
            if isinstance(q, int) and 0 <= q < k:
                counts[q] += 1
        return counts.index(min(counts)), Usage(approx_tokens(view.render()), 5)

    def compose(self, view, depth, mode, budget, retry, seed):
        steps = STEP_BUDGET[depth]
        usage = self._usage(view, steps)
        fam = view.family_id
        if fam.startswith("comm_"):
            bits = view.private.get("bits")
            if bits is None or budget == 0:
                return "", usage
            if fam == "comm_parity":
                return str(sum(bits) % 2), usage
            return "".join(str(b) for b in bits[:budget]), usage
        if fam == "reason":
            progress = self._latest_progress(view)
            _, (person, shift, done) = _reason_solve(view.public, steps, progress)
            return f"PROGRESS {person} {shift} {done}", usage
        return "", usage

    @staticmethod
    def _latest_progress(view: View) -> tuple[str, int, int] | None:
        best = None
        for msg in view.inbox:
            m = PROGRESS_RE.search(msg)
            if m and (best is None or int(m.group(3)) > best[2]):
                best = (m.group(1), int(m.group(2)), int(m.group(3)))
        return best

    def answer(self, view, depth, seed):
        steps = STEP_BUDGET[depth]
        usage = self._usage(view, steps)
        space = view.answer_space
        fam = view.family_id
        probs = _uniform(space)
        if fam == "info":
            probs = _info_posterior(view, steps)
        elif fam == "reason":
            value, _ = _reason_solve(view.public, steps, self._latest_progress(view))
            if value is not None:
                probs = _confident(space[value], space)
        elif fam.startswith("comm_"):
            probs = _comm_posterior(view, steps)
        best = max(probs, key=probs.get)
        # An uninformed agent guesses the same way whenever it sees the same text, as a model
        # given the same prompt tends to; this keeps paired comparisons efficient.
        guess = space[
            int(content_hash([view.scenario, view.shared_text, view.private_text])[:8], 16)
            % len(space)
        ]
        answer = best if probs[best] > 1 / len(space) + 1e-9 else guess
        return AdapterResult(
            answer, normalize_probabilities(probs, space), None, usage, f"sim-{seed}"
        )


def _reason_solve(public, steps, progress):
    from ceilings.tasks.reason import solve

    return solve(public, steps, progress)


def _info_posterior(view: View, steps: int) -> dict[str, float]:
    k, p = view.public["k"], view.public["p"]
    results = [(q, v) for q, v in view.tool_results if v in ("0", "1")]
    usable = results[: steps * 4]  # Bayes updates cost 1 step per 4 results
    weights = []
    for theta in range(2**k):
        w = 1.0
        for q, v in usable:
            w *= (1 - p) if ((theta >> q) & 1) == int(v) else p
        weights.append(w)
    total = sum(weights)
    return {view.answer_space[i]: w / total for i, w in enumerate(weights)}


def _comm_posterior(view: View, steps: int) -> dict[str, float]:
    space = view.answer_space
    own = view.private.get("bits")
    n = view.public["n"]
    if own is None or steps < math.ceil(n / 4):  # one step per 4 bits combined
        return _uniform(space)
    msg = ""
    for m in view.inbox:
        if re.fullmatch(r"[01]*", m):
            msg = m
    if view.family_id == "comm_parity":
        if not msg:
            return _uniform(space)
        value = (sum(own) + int(msg[0])) % 2
        return _confident(space[value], space)
    L = len(msg)
    partial = sum(a * int(b) for a, b in zip(own[:L], msg, strict=True)) % 2
    if all(b == 0 for b in own[L:]):
        return _confident(space[partial], space)
    return _uniform(space)


# --- Language-model agent ----------------------------------------------------------------------

SYSTEM = (
    "You are agent {name}, one member of a small team solving a task. Use only what you are shown. "
    "Be concise."
)


class LLMPolicy(Policy):
    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter
        self.model_id = adapter.model_id

    def _messages(self, view: View, instruction: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM.format(name=view.name)},
            {"role": "user", "content": f"{view.render()}\n\n{instruction}"},
        ]

    def choose_probe(self, view, depth, seed):
        k = view.public["k"]
        res = self.adapter.complete(
            self._messages(
                view,
                f"Choose which bit to probe next. Reply with only an integer from 0 to {k - 1}.",
            ),
            depth,
            None,
            seed,
            max_tokens=4000,
        )
        m = re.search(r"-?\d+", res.text or "")
        return (int(m.group()) if m else res.text), res.usage

    def compose(self, view, depth, mode, budget, retry, seed):
        if mode == "bits":
            rule = (
                f"Reply with only a string of 0s and 1s, at most {budget} characters long. "
                "Agent A will read it."
            )
        else:
            rule = f"Write a message for agent A of at most {budget} tokens."
        if retry:
            rule = f"{retry} {rule}"
        res = self.adapter.complete(self._messages(view, rule), depth, None, seed, max_tokens=4000)
        return (res.text or "").strip(), res.usage

    def answer(self, view, depth, seed):
        return self.adapter.complete(
            self._messages(view, "Give your final answer and a probability for every option."),
            depth,
            view.answer_space,
            seed,
        )

    def count_tokens(self, text: str) -> int:
        return self.adapter.count_tokens(text)


def make_policy(model_id: str) -> Policy:
    if model_id == "sim-bounded":
        return SimPolicy()
    if model_id == "stub-oracle":
        from ceilings.adapters.stub import StubOracleAdapter

        return LLMPolicy(StubOracleAdapter())
    if model_id.startswith("claude-"):
        from ceilings.adapters.anthropic_claude import ClaudeAdapter

        return LLMPolicy(ClaudeAdapter(model_id=model_id))
    raise ValueError(
        f"no adapter for model {model_id!r} (families B and C are chosen at the pilot)"
    )


__all__ = ["View", "Policy", "SimPolicy", "LLMPolicy", "make_policy", "STEP_BUDGET"]
