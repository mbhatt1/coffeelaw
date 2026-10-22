"""Information-bound family (FR-001): a fresh hidden host ID reachable only through a noisy probe.

Formal mapping: Theta is uniform on {0..2^k-1}; the probe returns bit q of Theta through a binary
symmetric channel with flip rate p, so each call carries at most 1 - h(p) bits (Theorem 1).
"""

from __future__ import annotations

import numpy as np

from ceilings import theory
from ceilings.records import TaskInstance, TheoryQuantities, ToolSpec, content_hash

FAMILY_ID = "info"
VERSION = "1"
T_MAX = 16
TOOL_RESULT_TEMPLATE = "Probe result: bit {q} of the failing host's ID reads {v}."
FORMAL_MAPPING = (
    "Theta = failing host index, uniform on 2^k values; probe(q) = bit q of Theta XOR Bernoulli(p); "
    "D_0 carries no information about Theta (the index is generated per instance from the seed)."
)


def host_name(i: int) -> str:
    return f"host-{i:02d}"


def generate(seed: int, k: int = 3, p: float = 0.1, split: str = "main") -> TaskInstance:
    rng = np.random.default_rng(seed)
    hidden = int(rng.integers(2**k))
    hosts = [host_name(i) for i in range(2**k)]
    scenario = (
        f"Incident triage. Exactly one of {2**k} hosts ({hosts[0]} to {hosts[-1]}) is failing. "
        "Which host is it?"
    )
    shared = (
        f"Each host's ID is its number written as a {k}-bit binary number (bit 0 is the least "
        f"significant bit). You can use a probe: give it a bit position from 0 to {k - 1} and it "
        f"reports that bit of the failing host's ID. The probe is wrong with probability {p}, "
        "independently on every call. Nothing else you know says which host is failing."
    )
    c = theory.bsc_capacity(p)
    m = 2**k
    budgets = [theory.budget(0.0, [c] * T) for T in range(T_MAX + 1)]
    floors = [theory.fano_floor(float(k), b, m) if m >= 3 else 0.0 for b in budgets]
    tool = ToolSpec(
        kind="bit_probe",
        answer_alphabet=["0", "1", "ERR"],
        flip_rate=p,
        capacity_bits=c,
        charge_tokens=len(TOOL_RESULT_TEMPLATE.format(q=0, v=0)) // 4,
    )
    body = {"family": FAMILY_ID, "version": VERSION, "seed": seed, "k": k, "p": p, "split": split}
    return TaskInstance(
        instance_id=content_hash(body),
        family_id=FAMILY_ID,
        family_version=VERSION,
        seed=seed,
        difficulty={"k": k, "p": p},
        answer=host_name(hidden),
        answer_space=hosts,
        scenario=scenario,
        shared_evidence=shared,
        private_evidence={"A": "", "B": ""},
        tool=tool,
        theory=TheoryQuantities(
            H_theta=float(k),
            I_held_by_agent={"A": 0.0, "B": 0.0},
            I_held_team=0.0,
            budget_by_T=budgets,
            fano_floor_by_T=floors,
        ),
        split=split,
        payload={
            "public": {"k": k, "p": p},
            "private": {"A": {}, "B": {}},
            "hidden": {"value": hidden},
        },
    )


def oracle_evidence(instance: TaskInstance, amount: int) -> str:
    """Arm E on this family is delivered as extra probe calls, not as text (see arms.py)."""
    return ""
