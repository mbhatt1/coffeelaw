"""Task-family registry and instance storage."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ceilings.records import TaskInstance
from ceilings.tasks import comm, info, reason

BINDING = {
    "info": "information",
    "reason": "reasoning",
    "comm_parity": "communication",
    "comm_ip": "communication",
}

GENERATORS: dict[str, Callable[..., TaskInstance]] = {
    "info": lambda seed, split="main", **d: info.generate(seed, split=split, **d),
    "reason": lambda seed, split="main", **d: reason.generate(seed, split=split, **d),
    "comm_parity": lambda seed, split="main", **d: comm.generate(
        seed, variant="parity", split=split, **d
    ),
    "comm_ip": lambda seed, split="main", **d: comm.generate(seed, variant="ip", split=split, **d),
}


def evidence_for_arm_e(instance: TaskInstance, amount: int) -> str:
    """Text injected by arm E (the information family uses extra probes instead)."""
    if instance.family_id == "reason":
        return reason.restated_evidence(instance, amount)
    if instance.family_id.startswith("comm_"):
        return comm.same_kind_private_evidence(instance, amount)
    return ""


def generate_many(
    family: str, count: int, split: str, seed0: int = 0, **difficulty
) -> list[TaskInstance]:
    offset = {"calibration": 1_000_000, "main": 0, "heldout_blind": 2_000_000}[split]
    return [GENERATORS[family](seed0 + offset + i, split=split, **difficulty) for i in range(count)]


def instance_path(root: Path, family: str, version: str, split: str) -> Path:
    return Path(root) / "instances" / family / version / f"{split}.jsonl"


def save_instances(instances: list[TaskInstance], root: Path) -> Path:
    first = instances[0]
    path = instance_path(root, first.family_id, first.family_version, first.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in instances))
    return path


def load_instances(root: Path, family: str, version: str, split: str) -> list[TaskInstance]:
    path = instance_path(root, family, version, split)
    return [TaskInstance.model_validate(json.loads(line)) for line in path.read_text().splitlines()]
