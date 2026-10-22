"""Generators: determinism, no answer leakage, exact theory, unique solutions, split evidence."""

import re

import pytest

from ceilings import tasks, theory
from ceilings.tasks import comm, info, reason


def test_info_deterministic_and_answer_never_in_evidence():
    for seed in range(200):
        a, b = info.generate(seed, k=3, p=0.1), info.generate(seed, k=3, p=0.1)
        assert a == b and a.instance_id == b.instance_id
        text = a.scenario + a.shared_evidence + "".join(a.private_evidence.values())
        hidden = a.payload["hidden"]["value"]
        # every host name appears in the scenario range only as the two endpoints
        assert text.count(a.answer) <= (
            1 if a.answer in (info.host_name(0), info.host_name(7)) else 0
        )
        assert f"{hidden:03b}" not in text


def test_info_theory_quantities():
    inst = info.generate(1, k=3, p=0.1)
    c = theory.bsc_capacity(0.1)
    assert inst.theory.H_theta == 3
    assert inst.theory.budget_by_T[4] == pytest.approx(4 * c)
    assert inst.theory.fano_floor_by_T[0] == pytest.approx(theory.fano_floor(3, 0, 8))


def _walk(public):
    child_of = {p: (c, o) for c, p, o in public["edges"]}
    person, shift = public["start"]
    steps = 0
    while person in child_of:
        person, o = child_of[person]
        shift = (shift + o) % 4
        steps += 1
    return person, shift, steps


@pytest.mark.parametrize("d", [4, 8, 16])
def test_reason_chain_unique_and_complete(d):
    for seed in range(50):
        inst = reason.generate(seed, d=d)
        pub = inst.payload["public"]
        children = [c for c, _, _ in pub["edges"]]
        assert len(children) == len(set(children)) == d  # a single path, each person once
        person, shift, steps = _walk(pub)
        assert person == pub["target"] and steps == d
        assert reason.SHIFTS[shift] == inst.answer
        assert inst.theory.I_held_team == inst.theory.H_theta
        for line in inst.shared_evidence.splitlines()[1:]:
            assert re.match(r"^\w+ works (the same shift as|\d shift\(s\) after) \w+\.$", line)


def test_reason_solver_respects_budget_and_resumes():
    inst = reason.generate(3, d=12)
    pub = inst.payload["public"]
    value, prog = reason.solve(pub, 4)
    assert value is None and prog[2] == 4
    value, prog = reason.solve(pub, 8, prog)
    assert reason.SHIFTS[value] == inst.answer and prog[2] == 12


@pytest.mark.parametrize("variant", ["parity", "ip"])
@pytest.mark.parametrize("n", [8, 16])
def test_comm_split_evidence(variant, n):
    inst = comm.generate(5, n=n, variant=variant)
    assert inst.theory.I_held_by_agent["A"] <= 0.01
    assert inst.theory.I_held_team == pytest.approx(inst.theory.H_theta)
    x = inst.payload["private"]["A"]["bits"]
    y = inst.payload["private"]["B"]["bits"]
    parsed_x = [1 if ln.endswith("yes") else 0 for ln in inst.private_evidence["A"].splitlines()]
    parsed_y = [1 if ln.endswith("yes") else 0 for ln in inst.private_evidence["B"].splitlines()]
    assert parsed_x == x and parsed_y == y  # natural-language wrapper round-trips
    f = theory.inner_product if variant == "ip" else theory.parity
    assert inst.answer == ("odd" if f(x, y) else "even")


def test_registry_and_storage(tmp_path):
    insts = tasks.generate_many("comm_ip", 5, "calibration", n=8)
    tasks.save_instances(insts, tmp_path)
    assert tasks.load_instances(tmp_path, "comm_ip", "1", "calibration") == insts
    assert {i.split for i in insts} == {"calibration"}
