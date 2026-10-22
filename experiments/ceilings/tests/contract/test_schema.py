import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from ceilings.records import (
    AgentUsage,
    Messages,
    ResourceLedger,
    RunRecord,
    ToolCalls,
    content_hash,
    now,
)

SCHEMA = json.loads(
    (
        Path(__file__).parents[2] / "specs/001-ceiling-discrimination-experiments/"
        "contracts/run-record.schema.json"
    ).read_text()
)


def _record(**over) -> dict:
    base = dict(
        run_id=content_hash("r"),
        instance_id=content_hash("i"),
        family_id="info",
        arm_id="E",
        delta_level="small",
        model_id="sim-bounded",
        replicate=0,
        sweep=None,
        prereg_hash=content_hash("p"),
        final_answer="host-01",
        probabilities={"host-00": 0.25, "host-01": 0.75},
        sample_answers=["host-01"],
        correct=True,
        refusal=None,
        ledger=ResourceLedger(
            per_agent={"A": AgentUsage(input_tokens=10, output_tokens=5, calls=1)},
            tool_calls=ToolCalls(count=1, charged_tokens=3),
            messages=Messages(count=0, tokens=0),
            total_tokens=18,
        ),
        transcript_ref="",
        provider_request_ids=["x"],
        started_at=now(),
        finished_at=now(),
    )
    base.update(over)
    return base


def test_valid_record_matches_schema():
    rec = RunRecord(**_record())
    jsonschema.validate(json.loads(rec.model_dump_json()), SCHEMA)


def test_sweep_record_matches_schema():
    rec = RunRecord(**_record(sweep={"T": 3, "flip_rate": 0.1}))
    jsonschema.validate(json.loads(rec.model_dump_json()), SCHEMA)


@pytest.mark.parametrize(
    "bad",
    [
        {"probabilities": {"a": 0.5, "b": 0.4}},
        {"arm_id": "Z"},
    ],
)
def test_invalid_records_rejected(bad):
    with pytest.raises(ValidationError):
        RunRecord(**_record(**bad))


def test_schema_rejects_missing_total_tokens():
    data = json.loads(RunRecord(**_record()).model_dump_json())
    del data["ledger"]["total_tokens"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, SCHEMA)
