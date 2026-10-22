"""Information-bound family (FR-001): a fresh hidden host reachable only through a noisy probe.

Formal mapping: Theta is uniform on {0..2^k-1}. Host i has k two-valued attributes; attribute q
takes its second value exactly when bit q of i is 1. The probe reports one attribute of the failing
host through a binary symmetric channel with flip rate p, so each call carries at most 1 - h(p) bits
(Theorem 1). Agents only look values up in the host table; no binary arithmetic is needed.
"""

from __future__ import annotations

import numpy as np

from ceilings import theory
from ceilings.records import TaskInstance, TheoryQuantities, ToolSpec, content_hash

FAMILY_ID = "info"
VERSION = "3"
T_MAX = 16
ATTRIBUTES = [
    ("rack", "A", "B"),
    ("row", "front", "back"),
    ("zone", "east", "west"),
    ("power feed", "1", "2"),
    ("switch", "sw-1", "sw-2"),
    ("tier", "gold", "silver"),
]
FORMAL_MAPPING = __doc__


def host_name(i: int) -> str:
    return f"host-{i:02d}"


def attribute_value(host: int, q: int) -> str:
    name, v0, v1 = ATTRIBUTES[q]
    return v1 if (host >> q) & 1 else v0


def probe_text(attributes: list, query: object, result: str) -> str:
    """How a probe result reads to an agent."""
    if result not in ("0", "1") or not isinstance(query, int):
        return f"Probe result: invalid request {query!r} (no such attribute); nothing learned."
    name, v0, v1 = attributes[query]
    return f"Probe result: the failing host's {name} is {v1 if result == '1' else v0}."


def generate(seed: int, k: int = 3, p: float = 0.1, split: str = "main") -> TaskInstance:
    if not 1 <= k <= len(ATTRIBUTES):
        raise ValueError(f"k must be between 1 and {len(ATTRIBUTES)}")
    rng = np.random.default_rng(seed)
    hidden = int(rng.integers(2**k))
    hosts = [host_name(i) for i in range(2**k)]
    attrs = [list(a) for a in ATTRIBUTES[:k]]
    names = ", ".join(a[0] for a in attrs)
    scenario = (
        f"Incident triage. Exactly one of {2**k} hosts ({hosts[0]} to {hosts[-1]}) is failing. "
        "Which host is it?"
    )
    table = "\n".join(
        f"{host_name(i)}: " + ", ".join(f"{a[0]} {attribute_value(i, q)}" for q, a in
                                        enumerate(attrs))
        for i in range(2**k)
    )
    reliability = (
        "Probe reports are always correct." if p == 0 else
        f"Each probe report is wrong with probability {p}, independently of the others; a wrong "
        "report shows the attribute's other value."
    )
    shared = (
        f"Host table (each host differs in at least one attribute):\n{table}\n\n"
        f"The team may receive probe reports. A probe report states one attribute ({names}) of "
        f"the failing host. {reliability} Apart from probe reports, nothing you know says which "
        "host is failing."
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
        charge_tokens=len(probe_text(attrs, 0, "0")) // 4,
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
            "public": {"k": k, "p": p, "attributes": attrs},
            "private": {"A": {}, "B": {}},
            "hidden": {"value": hidden},
        },
    )
