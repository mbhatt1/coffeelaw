"""The theory module must reproduce every reference number in litreview-coffee-law-check.py."""

from collections import defaultdict

import pytest

from ceilings import theory

EPS = 0.1
THETAS = [(a, b) for a in (0, 1) for b in (0, 1)]


def _flip(bit: int, out: int) -> float:
    return 1 - EPS if out == bit else EPS


def _two_rounds(policy) -> dict:
    joint: dict = defaultdict(float)
    for th in THETAS:

        def rec(hist, p, depth, th=th):
            if depth == 2:
                joint[(th, tuple(hist))] += p
                return
            i = policy(hist)
            for o in (0, 1):
                rec(hist + [(i, o)], p * _flip(th[i], o), depth + 1, th)

        rec([], 0.25, 0)
    return joint


def _paper_adaptive() -> dict:
    """Table I adaptive policy (repo verify.py scenario 7)."""
    joint: dict = defaultdict(float)
    for th in THETAS:
        for x1 in (0, 1):
            for x2 in (0, 1):
                for n1 in (0, 1):
                    for n2 in (0, 1):
                        p = 0.0625 * (EPS if n1 else 1 - EPS) * (EPS if n2 else 1 - EPS)
                        o1 = th[x1] ^ n1
                        q2 = (1 - x1) if o1 == 0 else x2
                        joint[(th, (x1, o1, x2, q2, th[q2] ^ n2))] += p
    return joint


def test_capacity_and_budget():
    c = theory.bsc_capacity(0.1)
    assert c == pytest.approx(0.531, abs=1e-3)
    assert theory.budget(0.0, [c, c]) == pytest.approx(1.062, abs=1e-3)


def test_table_one_policies():
    assert theory.mutual_information(_two_rounds(lambda h: len(h))) == pytest.approx(
        1.062, abs=1e-3
    )
    assert theory.mutual_information(_two_rounds(lambda h: 0)) == pytest.approx(0.742, abs=1e-3)
    assert theory.mutual_information(_paper_adaptive()) == pytest.approx(0.982, abs=1e-3)
    assert theory.map_error(_two_rounds(lambda h: len(h))) == pytest.approx(0.19, abs=1e-3)


def test_probe_helpers_match_table_one():
    assert theory.probe_information(2, 0.1, 2) == pytest.approx(1.062, abs=1e-3)
    assert theory.probe_map_error(2, 0.1, 2) == pytest.approx(0.19, abs=1e-3)


def test_one_look_usable_information():
    joint: dict = defaultdict(float)
    for th in THETAS:
        for i in (0, 1):
            for o in (0, 1):
                joint[(th, o)] += 0.25 * 0.5 * _flip(th[i], o)
    assert theory.mutual_information(joint) == pytest.approx(0.2655, abs=1e-3)


def test_corollary_two():
    assert theory.cor2_min_rounds(10, 1.0, 0.0) == pytest.approx(9.0)
    assert theory.cor2_min_rounds(10, theory.bsc_capacity(0.1), 0.1) == pytest.approx(
        15.066, abs=1e-3
    )


def test_inner_product_values():
    assert theory.best_one_bit_message_ip(4) == pytest.approx(0.1222, abs=1e-3)
    expected = [0.0597, 0.1222, 0.2472, 0.4972, 0.9972]
    for L, value in enumerate(expected):
        assert theory.ip_prefix_information(4, L) == pytest.approx(value, abs=1e-3)
    for L, value in [(40, 0.00282), (45, 0.08746), (49, 1.0)]:
        assert theory.ip_bound(100, L) == pytest.approx(value, abs=1e-4)


def test_split_agents_hold_almost_nothing():
    assert theory.agent_alone_information(8, "parity") == pytest.approx(0.0, abs=1e-9)
    assert theory.agent_alone_information(8, "ip") < 0.01
    assert theory.agent_alone_information(16, "ip") < 0.01


def test_fano_floor_needs_three_answers():
    with pytest.raises(ValueError):
        theory.fano_floor(1.0, 0.0, 2)
    assert theory.fano_floor(10.0, 0.0, 1024) == pytest.approx(9 / theory.math.log2(1023))
