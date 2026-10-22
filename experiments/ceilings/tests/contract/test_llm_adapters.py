"""Ollama / OpenAI-compatible adapters, the reasoning stage, and reply parsing. No network."""

import io
import json
import urllib.error

import pytest

from ceilings.adapters import make_adapter, safe_model_dir
from ceilings.adapters.base import REASON_PROMPT, REASONING_TOKENS, parse_answer
from ceilings.adapters.ollama import OllamaAdapter
from ceilings.adapters.openai_compat import OpenAICompatAdapter
from ceilings.agents import clean_bits, parse_probe

SPACE = ["even", "odd"]


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(replies, seen):
    def urlopen(req, timeout=None):
        seen.append((req.full_url, json.loads(req.data), dict(req.headers)))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(json.dumps(r).encode())
    return urlopen


def _ollama_reply(content, model="qwen2.5:1.5b", stop="stop", pin=20, pout=5):
    return {"model": model, "message": {"content": content}, "prompt_eval_count": pin,
            "eval_count": pout, "done_reason": stop, "created_at": "t"}


def test_model_ids():
    assert make_adapter("ollama/qwen2.5:1.5b").model_id == "ollama/qwen2.5:1.5b"
    assert safe_model_dir("ollama/qwen2.5:1.5b") == "ollama_qwen2.5_1.5b"
    with pytest.raises(ValueError):
        make_adapter("mystery-model")


def test_ollama_direct_answer_low_depth(monkeypatch):
    seen = []
    good = json.dumps({"answer": "odd", "probabilities": {"even": 0.2, "odd": 0.8}})
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([_ollama_reply(good)], seen))
    res = OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "low", SPACE, 7)
    url, body, _ = seen[0]
    assert url == "http://x:1/api/chat" and len(seen) == 1  # no reasoning stage at low
    assert sorted(body["format"]["properties"]["answer"]["enum"]) == SPACE
    assert body["options"]["temperature"] == 0.7
    assert body["options"]["seed"] == 8 and body["stream"] is False
    assert res.final_answer == "odd" and res.extra["parse"] == "json" and res.valid
    assert (res.usage.input_tokens, res.usage.output_tokens) == (20, 5)


def test_reasoning_stage_budget_and_accounting(monkeypatch):
    seen = []
    good = json.dumps({"answer": "even", "probabilities": {"even": 0.9, "odd": 0.1}})
    replies = [_ollama_reply("thinking...", stop="length", pout=1024), _ollama_reply(good)]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(replies, seen))
    res = OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "high", SPACE, 0)
    first, second = seen[0][1], seen[1][1]
    assert first["options"]["num_predict"] == REASONING_TOKENS["high"]
    assert "format" not in first and first["messages"][-1]["content"].endswith(REASON_PROMPT)
    assert second["messages"][-2] == {"role": "assistant", "content": "thinking..."}
    assert res.usage.output_tokens == 1029 and res.usage.input_tokens == 40
    assert res.extra["reasoning_stop"] == "length" and res.final_answer == "even"


def test_model_substitution_marks_invalid(monkeypatch):
    good = json.dumps({"answer": "odd", "probabilities": {"even": 0.5, "odd": 0.5}})
    monkeypatch.setattr("urllib.request.urlopen",
                        _fake_urlopen([_ollama_reply(good, model="llama3.2:3b")], []))
    res = OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert not res.valid


def test_openai_compat_schema_auth_price_and_fallback(monkeypatch):
    monkeypatch.setenv("CEILINGS_OPENAI_BASE_URL", "https://h/v1")
    monkeypatch.setenv("CEILINGS_OPENAI_API_KEY", "k")
    monkeypatch.setenv("CEILINGS_OPENAI_PRICE_IN", "1")
    monkeypatch.setenv("CEILINGS_OPENAI_PRICE_OUT", "2")
    seen = []
    err = urllib.error.HTTPError("https://h/v1/chat/completions", 400, "bad", {}, None)
    ok = {"id": "r1", "model": "m", "choices": [{"message": {"content": "I think odd."},
                                                 "finish_reason": "stop"}],
          "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([err, ok], seen))
    adapter = OpenAICompatAdapter("m")
    res = adapter.complete([{"role": "user", "content": "q"}], "low", SPACE, 0)
    assert seen[0][1]["response_format"]["type"] == "json_schema"
    assert "response_format" not in seen[1][1] and not adapter.supports_schema
    assert seen[1][2]["Authorization"] == "Bearer k"
    assert res.final_answer == "odd" and res.extra["parse"] == "text"
    assert res.usage.dollars == pytest.approx((1000 * 1 + 500 * 2) / 1e6)


def test_openai_compat_needs_base_url(monkeypatch):
    monkeypatch.delenv("CEILINGS_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatAdapter("m")


@pytest.mark.parametrize("text,expected,how", [
    ('{"answer": "B", "probabilities": {"A": 0.1, "B": 0.9}}', "B", "json"),
    ("The answer is host-03.", "host-03", "text"),
    ("I am not sure.", None, "failed"),
])
def test_parse_answer(text, expected, how):
    space = ["A", "B"] if expected in ("A", "B") else ["host-03", "host-13"]
    answer, probs, parsed = parse_answer(text, space)
    assert answer == expected and parsed == how and abs(sum(probs.values()) - 1) < 1e-9


def test_parse_answer_does_not_confuse_prefixes():
    assert parse_answer("host-13", ["host-1", "host-13"])[0] == "host-13"


def test_parse_probe_and_clean_bits():
    attrs = [["rack", "A", "B"], ["row", "front", "back"], ["power feed", "1", "2"]]
    assert parse_probe("Rack", attrs) == 0
    assert parse_probe("I'd probe the power feed.", attrs) == 2
    assert parse_probe("1", attrs) == 1
    assert parse_probe("colour", attrs) == "colour"
    assert clean_bits("```\n0 1 1\n```") == "011"
    assert clean_bits("yes") == "yes"


def test_reasoning_budget_override(monkeypatch):
    from ceilings.adapters.base import reasoning_tokens

    assert reasoning_tokens("high") == REASONING_TOKENS["high"]
    monkeypatch.setenv("CEILINGS_REASONING_TOKENS", "0,32,64,128,256")
    assert reasoning_tokens("low") == 0 and reasoning_tokens("high") == 64


def test_ollama_duty_cycle_threads_and_keep_alive(monkeypatch):
    monkeypatch.setenv("CEILINGS_OLLAMA_DUTY", "0.25")
    monkeypatch.setenv("CEILINGS_OLLAMA_NUM_THREAD", "2")
    monkeypatch.setenv("CEILINGS_OLLAMA_KEEP_ALIVE", "30s")
    seen, sleeps = [], []
    clock = iter([100.0, 102.0])  # the call takes 2 s
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    good = json.dumps({"answer": "odd", "probabilities": {"even": 0.4, "odd": 0.6}})
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen([_ollama_reply(good)], seen))
    OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "low", SPACE, 0)
    body = seen[0][1]
    assert body["options"]["num_thread"] == 2 and body["keep_alive"] == "30s"
    assert sleeps == [pytest.approx(6.0)]  # busy 25% of the time: sleep 3x the call


def test_ollama_rejects_bad_duty(monkeypatch):
    monkeypatch.setenv("CEILINGS_OLLAMA_DUTY", "0")
    with pytest.raises(ValueError):
        OllamaAdapter("m")


def test_truncated_reasoning_is_flagged_to_the_model(monkeypatch):
    seen = []
    good = json.dumps({"probabilities": {"even": 0.9, "odd": 0.1}, "answer": "even"})
    replies = [_ollama_reply("partial", stop="length"), _ollama_reply(good)]
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(replies, seen))
    OllamaAdapter("qwen2.5:1.5b", host="http://x:1").complete(
        [{"role": "user", "content": "q"}], "medium", SPACE, 0)
    assert seen[1][1]["messages"][-1]["content"].startswith("(Your reasoning was cut off")


def test_probability_diagnostics_and_argmax_fallback():
    from ceilings.adapters.base import probability_diagnostics

    text = json.dumps({"probabilities": {"even": 0.9, "odd": 0.9}, "answer": "odd"})
    answer, probs, how = parse_answer(text, SPACE)
    d = probability_diagnostics(text, answer, probs, SPACE)
    assert d["raw_prob_sum"] == pytest.approx(1.8) and not d["answer_not_argmax"]
    text = json.dumps({"probabilities": {"even": 0.8, "odd": 0.2}, "answer": "odd"})
    answer, probs, _ = parse_answer(text, SPACE)
    assert probability_diagnostics(text, answer, probs, SPACE)["answer_not_argmax"]
    answer, _, how = parse_answer(json.dumps({"probabilities": {"even": 0.3, "odd": 0.7}}), SPACE)
    assert (answer, how) == ("odd", "json_argmax")
