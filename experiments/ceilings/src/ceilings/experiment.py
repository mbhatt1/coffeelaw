"""Building blocks shared by the CLI and the notebook: prepare families, build cells, sweeps."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ceilings import arms, calibration, tasks, theory
from ceilings.agents import Policy
from ceilings.records import TaskInstance
from ceilings.runner import Cell

DEFAULT_DIFFICULTY = {
    "info": {"k": 3, "p": 0.1},
    "reason": {"d": 12},
    "comm_parity": {"n": 8},
    "comm_ip": {"n": 8},
}


def prepare(
    family: str,
    policy: Policy,
    root: Path | str,
    n_main: int,
    n_pilot: int,
    difficulty: dict | None = None,
    seed0: int = 0,
) -> dict:
    """Generate instances, run the calibration gate and cost matching; save all three."""
    root = Path(root)
    diff = difficulty or DEFAULT_DIFFICULTY[family]
    pilot = tasks.generate_many(family, n_pilot, "calibration", seed0, **diff)
    main = tasks.generate_many(family, n_main, "main", seed0, **diff)
    tasks.save_instances(pilot, root)
    tasks.save_instances(main, root)
    cal = calibration.calibrate(family, pilot, policy)
    calibration.save_json(
        cal.model_dump(), root / "calibration" / f"{family}__{policy.model_id}.json"
    )
    table = calibration.match_arms(family, pilot, policy)
    calibration.save_json(table, root / "arms" / family / f"{policy.model_id}.json")
    return {"calibration": cal, "table": table, "main": main, "pilot": pilot}


def load_table(root: Path | str, family: str, model_id: str) -> dict:
    return json.loads((Path(root) / "arms" / family / f"{model_id}.json").read_text())


def main_cells(
    instances: list[TaskInstance],
    table: dict,
    deltas=("small", "medium", "large"),
    arm_ids=tuple(arms.ARMS),
    replicates: int = 1,
) -> list[Cell]:
    cells = []
    for inst in instances:
        for rep in range(replicates):
            for arm_id in arm_ids:
                if arm_id == "baseline":
                    cells.append(
                        Cell(
                            inst, calibration.config_from_table(table, "baseline", None), None, rep
                        )
                    )
                    continue
                for lvl in deltas:
                    cfg = calibration.config_from_table(table, arm_id, lvl)
                    cells.append(Cell(inst, cfg, lvl, rep))
    return cells


def info_sweep_cells(
    k: int, Ts: list[int], ps: list[float], n: int, seed0: int = 500_000
) -> tuple[list[Cell], dict]:
    """H6 sweep: T probes at flip rate p, no filler. Returns cells and the Fano floor per cell."""
    cells, floors = [], {}
    for p in ps:
        insts = tasks.generate_many("info", n, "main", seed0 + int(p * 1000) * 10_000, k=k, p=p)
        c = theory.bsc_capacity(p)
        for T in Ts:
            floors[(T, p)] = theory.fano_floor(float(k), T * c, 2**k)
            cfg = replace(arms.build("E", "info", T), arm_id="E")
            for inst in insts:
                cells.append(Cell(inst, cfg, None, 0, {"T": T, "flip_rate": p}))
    return cells, floors


def comm_sweep_cells(
    variant: str, n_bits: int, Ls: list[int], n: int, seed0: int = 700_000
) -> list[Cell]:
    """H7 sweep: message budget of exactly L bits from B to A."""
    family = f"comm_{variant}"
    insts = tasks.generate_many(family, n, "main", seed0, n=n_bits)
    cells = []
    for L in Ls:
        cfg = arms.build("M", family, L)
        for inst in insts:
            cells.append(Cell(inst, cfg, None, 0, {"L_bits": L}))
    return cells
