"""Resource accounting in the common currency: tokens processed (research R4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ceilings.records import AgentUsage, Messages, ResourceLedger, ToolCalls


@dataclass
class _Agent:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    calls: int = 0


@dataclass
class Ledger:
    """Accumulates per-agent model usage, tool charges and message traffic for one run."""

    agents: dict[str, _Agent] = field(default_factory=dict)
    tool_count: int = 0
    tool_tokens: int = 0
    message_count: int = 0
    message_tokens: int = 0
    message_bits: int | None = None
    padding_tokens: int = 0
    dollars: float = 0.0

    def add_call(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int | None = None,
        dollars: float = 0.0,
    ) -> None:
        a = self.agents.setdefault(agent, _Agent())
        a.input_tokens += input_tokens
        a.output_tokens += output_tokens
        if reasoning_tokens is not None:
            a.reasoning_tokens = (a.reasoning_tokens or 0) + reasoning_tokens
        a.calls += 1
        self.dollars += dollars

    def add_tool_call(self, charge_tokens: int) -> None:
        self.tool_count += 1
        self.tool_tokens += charge_tokens

    def add_message(self, tokens: int, bits: int | None = None) -> None:
        self.message_count += 1
        self.message_tokens += tokens
        if bits is not None:
            self.message_bits = (self.message_bits or 0) + bits

    def add_padding(self, tokens: int) -> None:
        """Neutral top-up tokens used to match an arm's cost to its target delta."""
        self.padding_tokens += tokens

    @property
    def model_tokens(self) -> int:
        return sum(a.input_tokens + a.output_tokens for a in self.agents.values())

    @property
    def total_tokens(self) -> int:
        # Messages are counted inside the receiving agent's input tokens; tool charges and
        # padding are added explicitly because no model call reports them.
        return self.model_tokens + self.tool_tokens + self.padding_tokens

    def to_record(self, added_vs_baseline: float | None = None) -> ResourceLedger:
        return ResourceLedger(
            per_agent={
                name: AgentUsage(
                    input_tokens=a.input_tokens,
                    output_tokens=a.output_tokens,
                    reasoning_tokens=a.reasoning_tokens,
                    calls=a.calls,
                )
                for name, a in self.agents.items()
            },
            tool_calls=ToolCalls(count=self.tool_count, charged_tokens=self.tool_tokens),
            messages=Messages(
                count=self.message_count, tokens=self.message_tokens, bits=self.message_bits
            ),
            total_tokens=max(1, self.total_tokens),
            added_vs_baseline=added_vs_baseline,
            dollars=round(self.dollars, 6),
        )
