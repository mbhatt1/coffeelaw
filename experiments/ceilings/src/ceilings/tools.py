"""Simulated tools: the only channels from the hidden world into the team (assumption A1)."""

from __future__ import annotations

import numpy as np

from ceilings.ledger import Ledger
from ceilings.records import ToolSpec

ERROR_SYMBOL = "ERR"


class ToolSimulator:
    """A bit probe over a hidden integer answer, with independent flips at the spec's rate."""

    def __init__(
        self,
        spec: ToolSpec,
        hidden_value: int,
        n_bits: int,
        rng: np.random.Generator,
        ledger: Ledger,
    ) -> None:
        if spec.kind != "bit_probe":
            raise ValueError("only bit_probe tools are simulated; oracle evidence is injected text")
        self.spec = spec
        self.hidden_value = hidden_value
        self.n_bits = n_bits
        self.rng = rng
        self.ledger = ledger
        self.calls: list[tuple[object, str]] = []

    def probe(self, query: object) -> str:
        """Return bit `query` of the hidden value, flipped with probability `flip_rate`."""
        self.ledger.add_tool_call(self.spec.charge_tokens)
        if not isinstance(query, int) or isinstance(query, bool) or not 0 <= query < self.n_bits:
            out = ERROR_SYMBOL
        else:
            bit = (self.hidden_value >> query) & 1
            flip = bool(self.rng.random() < self.spec.flip_rate)
            out = str(bit ^ int(flip))
        self.calls.append((query, out))
        return out
