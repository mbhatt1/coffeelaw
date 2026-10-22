"""Adapter for any server exposing an OpenAI-style `/chat/completions` endpoint.

This covers hosted providers and self-hosted servers (vLLM, llama.cpp server, LM Studio, and so on).
Model IDs look like `openai/<model-name>`. Configuration comes from the environment:

- `CEILINGS_OPENAI_BASE_URL`: base URL including the version path, e.g. https://host/v1
- `CEILINGS_OPENAI_API_KEY`: bearer token (omit for unauthenticated local servers)
- `CEILINGS_OPENAI_PRICE_IN` / `CEILINGS_OPENAI_PRICE_OUT`: optional USD per million tokens, so
  `--max-cost` can stop a run
- `CEILINGS_TEMPERATURE`: sampling temperature, always sent explicitly (default 0.7)

Structured output uses `response_format: json_schema`. Servers that do not support it still work:
the answer is then parsed from text and the parse method is recorded in the transcript.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ceilings.adapters.base import ChatAdapter, ChatReply


class OpenAICompatAdapter(ChatAdapter):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 600.0, retries: int = 4) -> None:
        self.model_name = model_name
        self.model_id = f"openai/{model_name}"
        base = base_url or os.environ.get("CEILINGS_OPENAI_BASE_URL")
        if not base:
            raise ValueError("set CEILINGS_OPENAI_BASE_URL (for example https://host/v1)")
        self.base_url = base.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("CEILINGS_OPENAI_API_KEY")
        self.timeout = timeout
        self.retries = retries
        self.price_in = float(os.environ.get("CEILINGS_OPENAI_PRICE_IN", "0") or 0)
        self.price_out = float(os.environ.get("CEILINGS_OPENAI_PRICE_OUT", "0") or 0)
        self.supports_schema = True
        self.temperature = float(os.environ.get("CEILINGS_TEMPERATURE", "0.7"))

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 400 and "response_format" in body:
                    raise _SchemaUnsupported() from e
                if e.code in (408, 409, 429) or e.code >= 500:
                    last = e
                    time.sleep(2**attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(2**attempt)
        raise RuntimeError(f"request to {self.base_url} failed: {last}")

    def _chat(self, messages, schema, max_tokens, seed):
        body = {"model": self.model_name, "messages": messages, "max_tokens": max_tokens,
                "seed": int(seed) % (2**31), "temperature": self.temperature}
        if schema is not None and self.supports_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "schema": schema,
                                                       "strict": True}}
        try:
            d = self._post(body)
        except _SchemaUnsupported:
            self.supports_schema = False
            body.pop("response_format", None)
            d = self._post(body)
        choice = (d.get("choices") or [{}])[0]
        usage = d.get("usage") or {}
        return ChatReply(
            text=(choice.get("message") or {}).get("content") or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            request_id=d.get("id", ""),
            model_served=d.get("model", ""),
            stop=choice.get("finish_reason") or "",
        )

    def complete(self, messages, depth, answer_space, seed, max_tokens=16000):
        res = super().complete(messages, depth, answer_space, seed, max_tokens)
        res.usage.dollars = (res.usage.input_tokens * self.price_in
                             + res.usage.output_tokens * self.price_out) / 1e6
        return res


class _SchemaUnsupported(Exception):
    pass
