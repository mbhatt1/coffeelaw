# Implementation Plan: Ceiling-Discrimination Experiment Suite

**Branch**: `001-ceiling-discrimination-experiments` (no git repository; spec directory only) | **Date**: 2026-10-21 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-ceiling-discrimination-experiments/spec.md`

## Summary

Build a pre-registered experiment harness that tests the Coffee Law's three ceilings (information, reasoning, communication) against each other at matched cost.

The harness:
- generates natural-language task families whose binding ceiling is known by construction, with exactly computable information-theoretic quantities;
- runs each family under baseline, three matched-Δ interventions (E, K, M), two agent-count controls (A1, A2) and two placebos (P1, P2), on three model families;
- logs every token in a common currency;
- analyzes the results with paired bootstrap, equivalence tests and Holm correction, producing a verdict for each of the nine predictions H1–H9.

Quantitative sweeps (Fano floor, communication threshold), mixed-ceiling tasks and a masked blind diagnostic extend the core test. Technical approach: a Python package with deterministic seeded generators, a shared exact-theory module, provider adapters behind one interface, a channel object that enforces message budgets in tokens or bits, and a hash-locked pre-registration ([research.md](research.md)).

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**:
- `anthropic` (official SDK, for Claude Opus 5.5);
- adapters for two further model families (chosen at the pilot by the criteria in research R2);
- numpy, scipy, pandas, pyarrow, statsmodels, pydantic v2.

**Storage**: Files only:
- JSONL run records and transcripts under `data/`;
- Parquet analysis tables;
- a hash-locked pre-registration JSON under `prereg/`.

**Testing**:
- pytest + hypothesis: unit tests for theory and generators, contract tests for the adapter, channel and schema, integration tests with a stub model;
- a cheap live smoke test per adapter.

**Target Platform**: A local workstation or a single Linux VM. The open-weights family needs a GPU host for local serving.

**Project Type**: A research library with a CLI (`ceilings`).

**Performance Goals**:
- The main grid (~190k team-runs before the K/A2 multiplier) completes within the provider rate limits in about 2 weeks of wall-clock time.
- Analysis of the full dataset runs in under 30 minutes.

**Constraints**:
- Arms matched to ±5% of Δ tokens (SC-004).
- Equivalence margin ±5 points with ≥80% power (FR-017).
- No model fallbacks, and no shared state between agents.
- Spending is capped per command (`--max-cost`).
- The pre-registration is frozen before the main run.

**Scale/Scope**:
- 4 single-ceiling sub-families + 2 mixed + naturalistic;
- 9 arms, 3 Δ levels, 3 model families;
- ~850 paired instances per cell (planning figure, corrected during implementation; confirmed by the pilot power analysis).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

This project has no `.specify/memory/constitution.md`. In its place, the spec's integrity requirements are used as gates:

| Gate | Source | Status (pre-research) | Status (post-design) |
|---|---|---|---|
| Pre-registration frozen and hash-checked before main data | FR-015, SC-007 | Planned | **Pass**: `prereg freeze`, hash check on every command (contracts/cli.md) |
| Binding ceiling established, not assumed | FR-007, SC-002 | Planned | **Pass**: calibration gate, family `ready` state (data-model.md) |
| Matched cost, verifiable from ledgers | FR-008/009, SC-004 | Planned | **Pass**: ResourceLedger, `match` command, `audit` |
| No contamination across model families | FR-014 | Risk: default refusal fallbacks would swap models | **Pass**: fallbacks disabled (research R6), with a model-identity check in the adapter contract |
| Message budget is the only inter-agent path | FR-011 | Planned | **Pass**: Channel contract, bit-mode validation in `audit` |
| Every result reproducible from released materials | FR-019, SC-006 | Planned | **Pass**: seeded generators, content-hash IDs, an open-weights family |

No violations, so Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-ceiling-discrimination-experiments/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0: decisions R1–R12
├── data-model.md        # Phase 1: entities and validation
├── quickstart.md        # Phase 1: validation scenarios 0–7
├── contracts/
│   ├── cli.md
│   ├── interfaces.md
│   └── run-record.schema.json
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks; not created here)
```

### Source Code (project root `experiments/ceilings/`)

```text
src/ceilings/
├── theory.py            # entropy, MI, capacity, budget, Fano floors, IP bound (shared with tests)
├── tasks/
│   ├── info.py          # hidden k-bit value + bit-probe tool, incident-triage wrapper
│   ├── reason.py        # chained constraint puzzles, depth d
│   ├── comm.py          # parity-like and inner-product-like splits, hiring wrapper
│   ├── mixed.py         # info+reason, comm+reason compositions
│   └── natural.py       # loaders for naturalistic benchmark items (Story 4)
├── tools.py             # ToolSimulator (bit_probe, oracle)
├── channel.py           # Channel: token and bit budgets
├── adapters/
│   ├── base.py          # ModelAdapter interface
│   ├── anthropic_claude.py
│   ├── family_b.py      # second commercial family (named at pilot)
│   ├── open_weights.py  # locally served family (named at pilot)
│   └── stub.py          # stub-oracle for dry runs
├── arms.py              # arm configs and Δ matching
├── orchestrator.py      # TeamOrchestrator protocols per arm
├── ledger.py            # ResourceLedger
├── records.py           # pydantic models (mirrors contracts/run-record.schema.json)
├── runner.py            # grid execution, batching, resume, cost caps
├── prereg.py            # freeze, hash check, deviations
├── analysis/
│   ├── gains.py         # paired bootstrap, gain table, cost curves
│   ├── equivalence.py   # TOST, Holm
│   ├── interaction.py   # mixed-model interaction (H8)
│   ├── changepoint.py   # threshold vs linear (H7)
│   ├── usable_info.py   # calibration + H − log-loss
│   ├── diagnostic.py    # mask / unmask, accuracy, κ
│   └── report.py
└── cli.py

tests/
├── unit/                # theory numbers, generators, ledger arithmetic
├── contract/            # adapter, channel, schema, prereg hash
└── integration/         # stub-model end-to-end (quickstart Scenario 1)

prereg/                  # config.json → config_frozen.json, HASH, DEVIATIONS.md
data/                    # instances/, calibration/, arms/, runs/ (gitignored, released separately)
reports/
```

**Structure Decision**: a single Python project, with a `src/` layout and one module per ceiling mechanism. Keeping the tool (information), the orchestrator's compute settings (reasoning) and the channel (communication) in separate modules means each arm changes exactly one component. That separation is what makes the design valid.

`litreview-coffee-law-check.py` stays at the root as the reference for the theory numbers. `tests/unit` asserts that `theory.py` matches it.

## Complexity Tracking

No violations to justify.
