"""Grid execution with content-hash run IDs, resume, transcripts and a spending cap."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ceilings import prereg
from ceilings.agents import Policy
from ceilings.arms import ArmConfig
from ceilings.orchestrator import run_team
from ceilings.records import Refusal, RunRecord, Sweep, TaskInstance, content_hash, now


class BudgetExceeded(RuntimeError):
    """Accumulated spend passed --max-cost (CLI exit code 4)."""


@dataclass
class Cell:
    instance: TaskInstance
    cfg: ArmConfig
    delta_level: str | None = None
    replicate: int = 0
    sweep: dict | None = None


def run_id_for(
    instance_id: str,
    arm_id: str,
    delta_level: str | None,
    model_id: str,
    replicate: int,
    sweep: dict | None = None,
) -> str:
    return content_hash([instance_id, arm_id, delta_level, model_id, replicate, sweep])


def runs_path(root: Path, model_id: str, family: str) -> Path:
    return Path(root) / "runs" / model_id / family / "runs.jsonl"


def existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["run_id"] for line in path.read_text().splitlines() if line}


def run_cell(
    cell: Cell, policy: Policy, prereg_hash: str, transcript_dir: Path | None
) -> RunRecord:
    inst, cfg = cell.instance, cell.cfg
    rid = run_id_for(
        inst.instance_id, cfg.arm_id, cell.delta_level, policy.model_id, cell.replicate, cell.sweep
    )
    started = now()
    outcome = run_team(inst, cfg, policy, seed=int(rid[:8], 16))
    ref = ""
    if transcript_dir is not None:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        ref = str(transcript_dir / f"{rid}.json")
        Path(ref).write_text(
            json.dumps({"config": cfg.to_dict(), "steps": outcome.transcript}, default=str)
        )
    return RunRecord(
        run_id=rid,
        instance_id=inst.instance_id,
        family_id=inst.family_id,
        arm_id=cfg.arm_id,
        delta_level=cell.delta_level,
        model_id=policy.model_id,
        replicate=cell.replicate,
        sweep=Sweep(**cell.sweep) if cell.sweep else None,
        prereg_hash=prereg_hash,
        final_answer=outcome.final_answer,
        probabilities=outcome.probabilities,
        sample_answers=outcome.sample_answers,
        correct=outcome.final_answer == inst.answer,
        valid=outcome.valid,
        refusal=Refusal(**outcome.refusal) if outcome.refusal else None,
        ledger=outcome.ledger.to_record(),
        transcript_ref=ref,
        provider_request_ids=outcome.request_ids,
        started_at=started,
        finished_at=now(),
    )


def run_grid(
    cells: Iterable[Cell],
    policy: Policy,
    root: Path | str = "data",
    max_cost: float | None = None,
    require_frozen: bool = False,
    save_transcripts: bool = True,
) -> list[RunRecord]:
    """Run every cell not already on disk; append records as JSONL; return the new records."""
    root = Path(root)
    prereg_hash = prereg.check_hash() if require_frozen else prereg.current_hash()
    done: dict[Path, set[str]] = {}
    new: list[RunRecord] = []
    spent = 0.0
    for cell in cells:
        path = runs_path(root, policy.model_id, cell.instance.family_id)
        if path not in done:
            done[path] = existing_run_ids(path)
        rid = run_id_for(
            cell.instance.instance_id,
            cell.cfg.arm_id,
            cell.delta_level,
            policy.model_id,
            cell.replicate,
            cell.sweep,
        )
        if rid in done[path]:
            continue
        tdir = path.parent / "transcripts" if save_transcripts else None
        record = run_cell(cell, policy, prereg_hash, tdir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(record.model_dump_json() + "\n")
        done[path].add(rid)
        new.append(record)
        spent += record.ledger.dollars or 0.0
        if max_cost is not None and spent > max_cost:
            raise BudgetExceeded(f"spent ${spent:.2f} > --max-cost ${max_cost:.2f}")
    return new


def load_runs(root: Path | str = "data", model_id: str | None = None) -> list[dict]:
    root = Path(root) / "runs"
    if not root.exists():
        return []
    files = sorted(root.glob(f"{model_id or '*'}/*/runs.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line]
