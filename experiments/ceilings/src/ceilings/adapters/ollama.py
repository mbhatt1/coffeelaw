"""Ollama adapter (local open-weights models) over its HTTP chat API.

Model IDs look like `ollama/qwen2.5:1.5b`. The server is `OLLAMA_HOST` (default
http://localhost:11434; from inside Docker on a Mac use http://host.docker.internal:11434).
Structured output uses Ollama's `format` JSON schema. Every call passes a seed, so a run can be
replayed exactly on the same model build.

Throttling for laptops (all optional):
- `CEILINGS_OLLAMA_DUTY`: fraction of wall time spent generating, e.g. 0.5 sleeps after each call
  for as long as the call took (0.33 sleeps twice as long). Default 1.0 (no pause).
- `CEILINGS_OLLAMA_NUM_THREAD`: CPU threads Ollama may use for this model.
- `CEILINGS_OLLAMA_KEEP_ALIVE`: how long the model stays loaded after a call (e.g. "30s").

Sampling: `CEILINGS_TEMPERATURE` (default 0.7) is always sent explicitly, so results do not depend
on each model's own default.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ceilings.adapters.base import ChatAdapter, ChatReply


class OllamaAdapter(ChatAdapter):
    def __init__(self, model_name: str, host: str | None = None, timeout: float = 600.0,
                 temperature: float | None = None, retries: int = 3) -> None:
        self.model_name = model_name
        self.model_id = f"ollama/{model_name}"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.timeout = timeout
        self.temperature = (temperature if temperature is not None
                            else float(os.environ.get("CEILINGS_TEMPERATURE", "0.7")))
        self.retries = retries
        self.duty = float(os.environ.get("CEILINGS_OLLAMA_DUTY", "1.0") or 1.0)
        if not 0 < self.duty <= 1:
            raise ValueError("CEILINGS_OLLAMA_DUTY must be in (0, 1]")
        threads = os.environ.get("CEILINGS_OLLAMA_NUM_THREAD")
        self.num_thread = int(threads) if threads else None
        self.keep_alive = os.environ.get("CEILINGS_OLLAMA_KEEP_ALIVE")

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(f"{self.host}{path}", data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama request to {self.host}{path} failed: {last}")

    def _chat(self, messages, schema, max_tokens, seed):
        options = {"num_predict": max_tokens, "seed": int(seed) % (2**31)}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_thread:
            options["num_thread"] = self.num_thread
        body = {"model": self.model_name, "messages": messages, "stream": False,
                "options": options}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        if schema is not None:
            body["format"] = schema
        start = time.monotonic()
        d = self._post("/api/chat", body)
        if self.duty < 1:
            time.sleep((time.monotonic() - start) * (1 / self.duty - 1))
        return ChatReply(
            text=d.get("message", {}).get("content", ""),
            input_tokens=int(d.get("prompt_eval_count") or 0),
            output_tokens=int(d.get("eval_count") or 0),
            request_id=f"ollama-{d.get('created_at', '')}",
            model_served=d.get("model", ""),
            stop=d.get("done_reason", ""),
        )
