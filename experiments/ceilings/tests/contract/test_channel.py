import pytest

from ceilings.adapters.base import approx_tokens
from ceilings.channel import Channel
from ceilings.ledger import Ledger


def _channel(mode, budget):
    return Channel(mode, budget, Ledger(), approx_tokens)


def test_bits_mode_accepts_valid_message():
    ch = _channel("bits", 4)
    d = ch.send("B", "A", lambda retry: "0101")
    assert d.text == "0101" and d.bits == 4 and not d.retried
    assert ch.ledger.message_bits == 4 and ch.ledger.message_count == 1


@pytest.mark.parametrize("bad", ["01012", "yes", "0 1"])
def test_bits_mode_rejects_then_delivers_empty(bad):
    ch = _channel("bits", 4)
    d = ch.send("B", "A", lambda retry: bad)
    assert d.text == "" and d.retried and d.bits == 0


def test_bits_mode_cuts_long_bit_strings_to_the_budget():
    ch = _channel("bits", 2)
    d = ch.send("B", "A", lambda retry: "01011")
    assert d.text == "01" and d.bits == 2 and d.truncated and not d.retried
    assert ch.ledger.message_bits == 2


def test_bits_mode_retry_can_fix_message():
    ch = _channel("bits", 4)
    d = ch.send("B", "A", lambda retry: "11" if retry else "1 1")
    assert d.text == "11" and d.retried


def test_tokens_mode_never_exceeds_budget_per_direction():
    ch = _channel("tokens", 6)
    first = ch.send("B", "A", lambda retry: "short note")
    second = ch.send("B", "A", lambda retry: "a much longer message that cannot fit at all here")
    assert first.tokens + second.tokens <= 6
    assert second.truncated
    assert ch.ledger.message_count == 2


def test_invalid_mode():
    with pytest.raises(ValueError):
        _channel("words", 3)
