"""The only path between agents (research R8). Budgets are enforced in tokens or in bits."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ceilings.ledger import Ledger


@dataclass
class Delivery:
    sender: str
    receiver: str
    text: str
    tokens: int
    bits: int | None
    truncated: bool = False
    retried: bool = False


@dataclass
class Channel:
    """Message channel with a `tokens:B` or `bits:L` budget per direction."""

    mode: str
    budget: int
    ledger: Ledger
    count_tokens: Callable[[str], int]
    deliveries: list[Delivery] = field(default_factory=list)
    _used: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("tokens", "bits"):
            raise ValueError("mode must be 'tokens' or 'bits'")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        self._pattern = re.compile(rf"^[01]{{0,{self.budget}}}$") if self.mode == "bits" else None

    def valid_bits(self, text: str) -> bool:
        return bool(self._pattern and self._pattern.match(text))

    def send(self, sender: str, receiver: str, compose: Callable[[str | None], str]) -> Delivery:
        """Deliver `compose(None)`; in bit mode a malformed message gets one retry, then ''.

        `compose` receives the rejection reason on retry so an agent can correct itself.
        """
        text = compose(None)
        retried = False
        if self.mode == "bits":
            if not self.valid_bits(text):
                retried = True
                text = compose(f"Rejected: reply with only 0s and 1s, at most {self.budget}.")
                if not self.valid_bits(text):
                    text = ""
            delivery = Delivery(
                sender,
                receiver,
                text,
                self.count_tokens(text) if text else 0,
                bits=len(text),
                retried=retried,
            )
            self.ledger.add_message(delivery.tokens, bits=len(text))
        else:
            key = (sender, receiver)
            left = self.budget - self._used.get(key, 0)
            tokens = self.count_tokens(text)
            truncated = False
            if tokens > left:
                words = text.split()
                while words and self.count_tokens(" ".join(words)) > left:
                    words.pop()
                text = " ".join(words)
                tokens = self.count_tokens(text) if text else 0
                truncated = True
            self._used[key] = self._used.get(key, 0) + tokens
            delivery = Delivery(sender, receiver, text, tokens, bits=None, truncated=truncated)
            self.ledger.add_message(tokens)
        self.deliveries.append(delivery)
        return delivery
