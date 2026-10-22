import numpy as np
import pytest

from ceilings.ledger import Ledger
from ceilings.records import ToolSpec
from ceilings.tools import ERROR_SYMBOL, ToolSimulator


def _tool(p=0.1, seed=0):
    spec = ToolSpec(
        kind="bit_probe",
        answer_alphabet=["0", "1", "ERR"],
        flip_rate=p,
        capacity_bits=0.531,
        charge_tokens=7,
    )
    return ToolSimulator(
        spec, hidden_value=0b101, n_bits=3, rng=np.random.default_rng(seed), ledger=Ledger()
    )


def test_flip_rate_matches_spec():
    tool = _tool(p=0.1)
    outs = [tool.probe(1) for _ in range(20_000)]  # true bit 1 of 0b101 is 0
    assert abs(np.mean([o == "1" for o in outs]) - 0.1) < 0.01


def test_outputs_in_alphabet_and_charged():
    tool = _tool()
    for q in [0, 1, 2, 3, -1, "x", True]:
        assert tool.probe(q) in tool.spec.answer_alphabet
    assert tool.ledger.tool_count == 7 and tool.ledger.tool_tokens == 49


@pytest.mark.parametrize("q", [3, -1, "1", None, True])
def test_malformed_queries_return_error_symbol(q):
    assert _tool().probe(q) == ERROR_SYMBOL
