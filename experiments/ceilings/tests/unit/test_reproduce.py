"""Same seeds, same instances and same outcomes (FR-019, SC-006)."""

from ceilings import tasks
from ceilings.agents import SimPolicy
from ceilings.arms import build
from ceilings.orchestrator import run_team


def test_instances_regenerate_identically():
    for fam, diff in [("info", {"k": 3, "p": 0.1}), ("reason", {"d": 8}), ("comm_ip", {"n": 8})]:
        a = tasks.generate_many(fam, 20, "main", **diff)
        b = tasks.generate_many(fam, 20, "main", **diff)
        assert [i.instance_id for i in a] == [i.instance_id for i in b]


def test_team_runs_are_deterministic_per_seed():
    inst = tasks.generate_many("info", 1, "main", k=3, p=0.1)[0]
    cfg = build("E", "info", 5)
    r1, r2 = run_team(inst, cfg, SimPolicy(), 42), run_team(inst, cfg, SimPolicy(), 42)
    assert r1.final_answer == r2.final_answer and r1.transcript == r2.transcript
    assert r1.ledger.total_tokens == r2.ledger.total_tokens
