"""Adapter contract: shape, determinism, refusals, no silent model substitution. No network."""

import json
from types import SimpleNamespace

from ceilings.adapters.anthropic_claude import MODEL_ID, ClaudeAdapter
from ceilings.adapters.base import answer_schema, normalize_probabilities
from ceilings.adapters.stub import StubOracleAdapter

SPACE = ["A", "B", "C", "D"]


def test_stub_shape_and_determinism():
    stub = StubOracleAdapter()
    msgs = [{"role": "user", "content": "question"}]
    r1, r2 = stub.complete(msgs, "low", SPACE, 7), stub.complete(msgs, "low", SPACE, 7)
    assert r1.final_answer == r2.final_answer and set(r1.probabilities) == set(SPACE)
    assert r1.usage.input_tokens > 0 and r1.refusal is None


def test_stub_uses_oracle_marker():
    r = StubOracleAdapter().complete(
        [{"role": "user", "content": "x __oracle_answer__=C"}], "low", SPACE, 1
    )
    assert r.final_answer == "C" and r.probabilities["C"] == 0.99


def test_normalize_probabilities():
    p = normalize_probabilities({"A": 2.0, "B": 0.0, "Z": 5.0}, SPACE)
    assert set(p) == set(SPACE) and abs(sum(p.values()) - 1) < 1e-9
    assert normalize_probabilities(None, SPACE) == {a: 0.25 for a in SPACE}


def test_answer_schema_probabilities_first_and_shuffled():
    s = answer_schema(SPACE)
    assert list(s["properties"]) == ["probabilities", "answer"]
    assert s["properties"]["answer"]["enum"] == SPACE
    shuffled = [answer_schema(SPACE, seed)["properties"]["answer"]["enum"] for seed in range(20)]
    assert all(sorted(o) == sorted(SPACE) for o in shuffled)
    assert len({tuple(o) for o in shuffled}) > 1
    assert answer_schema(SPACE, 3) == answer_schema(SPACE, 3)


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return self.responses.pop(0)


def _response(text="", stop="end_turn", model=MODEL_ID, details=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        stop_details=details,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        model=model,
        id="msg_1",
    )


def _claude(responses):
    client = SimpleNamespace(messages=_FakeMessages(responses))
    return ClaudeAdapter(client=client), client.messages


def test_claude_params_effort_explicit_no_sampling_no_fallbacks():
    good = json.dumps({"answer": "B", "probabilities": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}})
    adapter, fake = _claude([_response(good)])
    res = adapter.complete(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], "high", SPACE, 0
    )
    params = fake.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["effort"] == "high"
    assert params["output_config"]["format"]["type"] == "json_schema"
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "fallbacks"):
        assert forbidden not in params
    assert res.final_answer == "B" and res.valid and res.usage.reasoning_tokens is None
    assert res.usage.dollars > 0


def test_claude_refusal_recorded_and_scored_wrong():
    details = SimpleNamespace(category="cyber", explanation="declined")
    adapter, _ = _claude([_response(stop="refusal", details=details)])
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer is None and res.refusal == {
        "category": "cyber",
        "explanation": "declined",
    }


def test_claude_model_substitution_marks_invalid():
    good = json.dumps({"answer": "A", "probabilities": {a: 0.25 for a in SPACE}})
    adapter, _ = _claude([_response(good, model="claude-opus-4-8")])
    assert not adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0).valid


def test_claude_parse_failure_retried_once():
    good = json.dumps({"answer": "D", "probabilities": {a: 0.25 for a in SPACE}})
    adapter, fake = _claude([_response("not json"), _response(good)])
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert res.final_answer == "D" and len(fake.calls) == 2 and res.usage.input_tokens == 200
