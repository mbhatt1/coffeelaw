# Quickstart: validating the ceiling-discrimination suite end to end

This is a run guide, not the implementation. It lists the scenarios that show the suite works, in the order you'd run them. Commands follow [contracts/cli.md](contracts/cli.md). Record shapes are in [data-model.md](data-model.md) and [contracts/run-record.schema.json](contracts/run-record.schema.json).

## Prerequisites

- Python 3.12 and the package installed in editable mode (`pip install -e .[dev]`).
- Claude credentials resolvable by the Anthropic SDK (an `ANTHROPIC_API_KEY`, or an `ant auth login` profile).
- Credentials or a local server for the other two model families (chosen at the pilot; see research R2).
- A spending cap for the session, passed as `--max-cost` to every command that calls a model.

## Scenario 0: theory checks, no model calls

```bash
pytest tests/unit -q
```

**Expected:** the theory module reproduces every number in `litreview-coffee-law-check.py`:
- C = 0.531, budgets 1.062 and 0.982, MAP error 0.19;
- the IP n=4 best 1-bit message 0.122;
- the Fano floors 9 and 15.07.

Generator tests confirm two properties: info-family answers never appear in evidence, and comm-family single-agent information is ≤ 0.01 bits.

## Scenario 1: dry run with a stub model

```bash
ceilings generate --family info --difficulty k=4,p=0.1 --n 50 --split calibration --seed 1
ceilings run --family info --arms baseline,E,K,M,A1,P1,P2 --delta small --model stub-oracle --split calibration
ceilings audit
```

**Expected:**
- with `stub-oracle` (always correct when evidence is present, otherwise chance), E is correct and baseline sits at chance;
- every record validates against the schema;
- `audit` passes ledger totals and bit-mode checks.

This scenario proves the plumbing with no spend.

## Scenario 2: calibration gate (pilot, ~10% scale)

```bash
ceilings calibrate --family info --model claude-opus-5-5 --max-cost 50
ceilings calibrate --family reason --model claude-opus-5-5 --max-cost 50
ceilings calibrate --family comm_ip --model claude-opus-5-5 --max-cost 50
```

**Expected:** each family reports `passed: true` (baseline within 5 points of chance; matched full intervention ≥ 90%). On a failure, retune the difficulty and rerun. Repeat for the other two model families.

## Scenario 3: cost matching

```bash
ceilings match --family info --model claude-opus-5-5 --delta small,medium,large
```

**Expected:** every arm is `matched: true`, with mean added tokens within ±5% of target (SC-004).

## Scenario 4: freeze the pre-registration

```bash
ceilings prereg freeze --config prereg/config.json
```

**Expected:** `prereg/HASH` is written, and every later command checks it. Publish the frozen file before Scenario 5.

## Scenario 5: main run and analysis (Story 1)

```bash
ceilings run --family info,reason,comm_parity,comm_ip --arms baseline,E,K,M,A1,A2,A2_single_ref,P1,P2 \
  --delta small,medium,large --model all --split main --batch --max-cost <budget>
ceilings analyze --out reports/
```

**Expected:** `reports/gain_table.csv` holds the family × arm table, and `reports/predictions.json` holds a verdict for each of H1–H5. The diagonal pattern is the prediction under test, not a requirement for success; SC-001 asks only for decisive verdicts.

## Scenario 6: quantitative sweeps (Story 2)

```bash
ceilings sweep --family info --T 0,1,2,4,8,12,16 --flip-rate 0,0.1,0.2
ceilings sweep --family comm_parity,comm_ip --L 0,1,2,4,8,16,32
ceilings analyze --out reports/
```

**Expected:**
- **H6:** no error rate significantly below `fano_floor_by_T`.
- **H7:** the change-point vs linear comparison is reported for IP, and parity shows its jump at L=1.

## Scenario 7: mixed tasks and the blind diagnostic (Stories 3–4)

```bash
ceilings run --family mix_info_reason,mix_comm_reason --arms baseline,E,K,M,E+K,M+K --model all
ceilings mask --split heldout_blind
# analyst fills blind/calls.json from the masked tables only
ceilings unmask --calls blind/calls.json
ceilings report
```

**Expected:**
- **H8:** the interaction estimate.
- **H9:** blind accuracy with a confidence interval against chance (1 in 3), plus naturalistic agreement and κ.
- `reports/final.md` lists every deviation from the pre-registration (SC-007).
