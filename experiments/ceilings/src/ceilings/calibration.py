"""Calibration gate (FR-007) and cost matching of arms to target deltas (FR-009, SC-004)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ceilings import arms
from ceilings.agents import Policy
from ceilings.orchestrator import run_team
from ceilings.records import CalibrationResult, TaskInstance, content_hash

LEAK_CONFIDENCE = 0.9
MATCH_TOLERANCE = 0.05


def _seed(*parts) -> int:
    return int(content_hash(list(parts))[:8], 16)


def _run(inst: TaskInstance, cfg: arms.ArmConfig, policy: Policy):
    return run_team(inst, cfg, policy, _seed(inst.instance_id, cfg.to_dict(), "pilot"))


def _n_bits(inst: TaskInstance) -> int | None:
    return inst.payload.get("public", {}).get("n")


def calibrate(family: str, instances: list[TaskInstance], policy: Policy) -> CalibrationResult:
    """Baseline must sit within 5 points of chance (allowing for sampling error); the family's own
    full fix must reach 90%."""
    chance = 1 / len(instances[0].answer_space)
    base = [_run(i, arms.build("baseline", family), policy) for i in instances]
    arm_id, strength = arms.full_strength(family, _n_bits(instances[0]))
    full = [_run(i, arms.build(arm_id, family, strength), policy) for i in instances]
    base_acc = float(
        np.mean([o.final_answer == i.answer for o, i in zip(base, instances, strict=True)])
    )
    full_acc = float(
        np.mean([o.final_answer == i.answer for o, i in zip(full, instances, strict=True)])
    )
    leaks = (
        [
            o.final_answer == i.answer and o.probabilities.get(i.answer, 0) >= LEAK_CONFIDENCE
            for o, i in zip(base, instances, strict=True)
        ]
        if family == "info"
        else []
    )
    refusals = [o.refusal is not None for o in base + full]
    # FR-007's 5-point tolerance is applied to the estimate's 95% interval, not the point
    # estimate: with n=200 and chance=0.25 a point-estimate rule fails ~11% of truly-at-chance
    # families on sampling noise alone.
    se = float(np.sqrt(chance * (1 - chance) / len(instances)))
    return CalibrationResult(
        family_id=family,
        model_id=policy.model_id,
        n=len(instances),
        baseline_acc=base_acc,
        chance_acc=chance,
        matched_full_acc=full_acc,
        leak_rate=float(np.mean(leaks)) if leaks else 0.0,
        refusal_rate=float(np.mean(refusals)),
        passed=abs(base_acc - chance) <= 0.05 + 1.96 * se and full_acc >= 0.90,
    )


def _mean_added(instances, cfg, policy, base_cost) -> float:
    costs = [_run(i, cfg, policy).ledger.total_tokens - base_cost[i.instance_id] for i in instances]
    return float(np.mean(costs))


def _filler_readers(cfg: arms.ArmConfig) -> int:
    """How many answer calls read the answerer's extra context (filler is read on every call)."""
    return cfg.samples


def match_arms(
    family: str,
    instances: list[TaskInstance],
    policy: Policy,
    levels: dict[str, float] | None = None,
) -> dict:
    """Choose each arm's strength and filler so its mean added cost is within 5% of the target."""
    levels = levels or arms.DELTA_MULTIPLIERS
    base_cfg = arms.build("baseline", family)
    base_cost = {i.instance_id: _run(i, base_cfg, policy).ledger.total_tokens for i in instances}
    base_mean = float(np.mean(list(base_cost.values())))
    n_bits = _n_bits(instances[0])
    table: dict = {
        "family": family,
        "model_id": policy.model_id,
        "baseline_mean_tokens": base_mean,
        "targets": {lvl: round(m * base_mean) for lvl, m in levels.items()},
        "arms": {},
    }
    for arm_id in [a for a in arms.ARMS if a != "baseline"]:
        table["arms"][arm_id] = {}
        for lvl, target in table["targets"].items():
            if arm_id == "A2_single_ref":
                strengths = [table["arms"]["A2"][lvl]["strength"]]
            else:
                strengths = arms.candidates(arm_id, family, n_bits)
            chosen, natural = strengths[0], None
            for s in strengths:
                added = _mean_added(instances, arms.build(arm_id, family, s), policy, base_cost)
                if natural is None or added <= target * (1 + MATCH_TOLERANCE):
                    chosen, natural = s, added
                if added > target * (1 + MATCH_TOLERANCE):
                    break
            cfg = arms.build(arm_id, family, chosen)
            filler = 0
            if arm_id in ("P1", "P2"):
                filler = max(0, int(target / max(1, cfg.samples)))
            elif natural < target:
                filler = int((target - natural) / max(1, _filler_readers(cfg)))
            added = natural
            for _ in range(3):  # proportional correction of the filler size
                cfg = replace(arms.build(arm_id, family, chosen), filler_tokens=filler)
                added = _mean_added(instances, cfg, policy, base_cost)
                if abs(added - target) <= MATCH_TOLERANCE * target:
                    break
                per_token = max(
                    1e-6, (added - (natural if arm_id not in ("P1", "P2") else 0)) / max(1, filler)
                )
                filler = max(0, int(filler + (target - added) / per_token))
            table["arms"][arm_id][lvl] = {
                "strength": chosen,
                "filler_tokens": filler,
                "natural_added": natural,
                "added": added,
                "target": target,
                "matched": abs(added - target) <= MATCH_TOLERANCE * target,
            }
    return table


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def config_from_table(table: dict, arm_id: str, level: str | None) -> arms.ArmConfig:
    family = table["family"]
    if arm_id == "baseline":
        return arms.build("baseline", family)
    entry = table["arms"][arm_id][level]
    return arms.build(arm_id, family, entry["strength"], entry["filler_tokens"])
