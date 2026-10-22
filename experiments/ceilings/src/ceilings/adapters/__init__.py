"""Model IDs and how to build an adapter for each.

| model ID                     | adapter                                   |
|------------------------------|-------------------------------------------|
| `sim-bounded`                | simulated agent (no model; see agents.py) |
| `stub-oracle`                | stub for plumbing tests                   |
| `claude-...` / `anthropic/claude-...` | Anthropic API, native effort     |
| `ollama/<name>`              | local Ollama server                       |
| `openai/<name>`              | any OpenAI-style /chat/completions server |
"""

from __future__ import annotations

import re

from ceilings.adapters.base import ModelAdapter


def make_adapter(model_id: str) -> ModelAdapter:
    if model_id == "stub-oracle":
        from ceilings.adapters.stub import StubOracleAdapter

        return StubOracleAdapter()
    if model_id.startswith("anthropic/"):
        model_id = model_id.split("/", 1)[1]
    if model_id.startswith("claude-"):
        from ceilings.adapters.anthropic_claude import ClaudeAdapter

        return ClaudeAdapter(model_id=model_id)
    if model_id.startswith("ollama/"):
        from ceilings.adapters.ollama import OllamaAdapter

        return OllamaAdapter(model_id.split("/", 1)[1])
    if model_id.startswith("openai/"):
        from ceilings.adapters.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(model_id.split("/", 1)[1])
    raise ValueError(
        f"unknown model ID {model_id!r}; use claude-<model>, ollama/<model> or openai/<model>"
    )


def safe_model_dir(model_id: str) -> str:
    """A model ID as a single folder name (`ollama/qwen2.5:1.5b` -> `ollama_qwen2.5_1.5b`)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id)
