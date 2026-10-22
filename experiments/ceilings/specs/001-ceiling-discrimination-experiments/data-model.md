# Data Model: Ceiling-Discrimination Experiment Suite

Entities follow the spec's Key Entities. Every record is immutable once written: corrections create a new version and a deviation entry, never an edit in place. IDs are content hashes, so reruns are idempotent.

## TaskFamily

| Field | Type | Notes |
|---|---|---|
| `family_id` | enum | `info`, `reason`, `comm_parity`, `comm_ip`, `mix_info_reason`, `mix_comm_reason`, `natural_*` |
| `binding` | set of {`information`, `reasoning`, `communication`} | One element for single families, two for mixed ones, unknown (`null`) for naturalistic tasks until annotated |
| `scenario` | string | Natural-language wrapper (incident triage, hiring, scheduling); FR-005 |
| `formal_mapping` | text | Documents how the scenario maps to the formal task (FR-005) |
| `difficulty` | object | `k` (hidden bits), `p` (tool flip rate), `d` (reasoning depth), `n` (split size) as applicable |
| `calibration` | CalibrationResult or null | Set by the gate; the family is `ready` only when the gate passes |

**Validation**: a family enters the main run only if `calibration.passed` is true for every model family (FR-007, SC-002).

**States**: `draft` → `calibrating` → `ready` | `rejected` (difficulty retuned, which creates a new family version).

## CalibrationResult

| Field | Type | Notes |
|---|---|---|
| `model_id` | string | |
| `baseline_acc`, `chance_acc` | float | Gate: `|baseline_acc − chance_acc| ≤ 0.05` |
| `matched_full_acc` | float | Gate: `≥ 0.90` |
| `leak_rate` | float | Fraction of info-family instances solved at baseline beyond chance (flagged and replaced) |
| `refusal_rate` | float | Over 0.05 triggers a prompt review |
| `passed` | bool | |

## TaskInstance

| Field | Type | Notes |
|---|---|---|
| `instance_id` | hash | sha256 of (family version, seed) |
| `family_id`, `family_version` | | |
| `seed` | int | Fully reproduces the instance |
| `answer` | value | Hidden truth Θ |
| `answer_space` | list | All possible answers (needed for probabilities and chance level) |
| `private_evidence` | map agent → text | Empty for non-split families; each agent's share for comm families |
| `shared_evidence` | text | Everything in all agents' context |
| `tool` | ToolSpec or null | Present for info and mixed-info families |
| `theory` | TheoryQuantities | Exact values (FR-006) |
| `split` | `calibration` / `main` / `heldout_blind` | Held-out instances feed Story 4 |
| `excluded` | bool + reason | Leakage or malformed instance |

## ToolSpec

| Field | Type | Notes |
|---|---|---|
| `kind` | enum | `bit_probe` (returns the requested bit), `oracle` (returns relevant evidence, used for arm E) |
| `answer_alphabet` | list | Fixed output alphabet, so capacity is known exactly |
| `flip_rate` | float | p |
| `capacity_bits` | float | `1 − h(p)` for `bit_probe` |
| `charge_tokens` | int | Fixed per-call charge in the resource currency (R4) |

## TheoryQuantities

| Field | Notes |
|---|---|
| `H_theta` | Answer entropy in bits |
| `I_held_by_agent` | Map agent → I(Θ; that agent's evidence) |
| `I_held_team` | I(Θ; all evidence) |
| `budget_by_T` | B_T for T = 0..T_max (info families) |
| `fano_floor_by_T` | Error floor from Corollary 2 / Theorem 1 |
| `comm_bound_by_L` | `log2(1 + 2^{L+1−n/2})` for IP splits; exact single-message optimum where computable |

## Arm

| Field | Type | Notes |
|---|---|---|
| `arm_id` | enum | `baseline`, `E`, `K`, `M`, `A1`, `A2`, `A2_single_ref`, `P1`, `P2`, or a pair such as `E+K` for mixed families |
| `delta_level` | enum | `small`, `medium`, `large` (none for baseline) |
| `target_delta_tokens` | int | Set by the pilot |
| `config` | object | E: oracle evidence amount; K: (effort, N samples); M: message budget (tokens or L bits); A1/A2: agent count and per-agent shares; P1: irrelevant-evidence length; P2: padding length |
| `matched` | bool | Pilot mean added cost within ±5% of target (SC-004) |

## RunRecord (one execution of one instance × arm × Δ × model)

| Field | Type | Notes |
|---|---|---|
| `run_id` | hash | sha256 of (instance_id, arm_id, delta_level, model_id, replicate) |
| `instance_id`, `family_id`, `arm_id`, `delta_level`, `model_id`, `replicate` | | `family_id` is denormalized from the instance so analysis can group runs without a join |
| `valid` | bool | false when the adapter detects model substitution or another integrity failure |
| `prereg_hash` | string | Must equal `prereg/HASH` at write time |
| `final_answer` | value or null | null on refusal or parse failure |
| `probabilities` | map answer → float | Elicited; must sum to 1 ± 1e-6 after normalization |
| `sample_answers` | list | Arm K: every sample's answer |
| `correct` | bool | |
| `refusal` | {category, explanation} or null | |
| `ledger` | ResourceLedger | |
| `transcript_ref` | path | Full messages and tool calls, stored separately |
| `provider_request_ids` | list | For audit |
| `started_at`, `finished_at` | timestamp | |

**Validation**: `probabilities` keys = `answer_space`; `ledger.total_tokens` > 0; `prereg_hash` matches; in bit mode, every message in the transcript matches `[01]{0,L}`.

## ResourceLedger

| Field | Notes |
|---|---|
| `per_agent` | Map agent → {input_tokens, output_tokens, reasoning_tokens (where reported), calls} |
| `tool_calls` | Count and charged tokens |
| `messages` | Count, total tokens, total bits (bit mode) |
| `total_tokens` | The common currency (R4) |
| `added_vs_baseline` | `total_tokens` minus the mean baseline cost for the same instance and model |
| `dollars` | Secondary currency (sensitivity analysis only) |

## Prediction

| Field | Notes |
|---|---|
| `prediction_id` | H1–H9 |
| `statement`, `paper_source` | From the spec's prediction table |
| `test` | `superiority`, `TOST(±0.05)`, `interaction`, `changepoint_vs_linear`, `floor_check`, `classification_accuracy` |
| `cells` | The family × arm × Δ cells it uses |
| `verdict` | `supported` / `contradicted` / `inconclusive` (FR-020) |
| `estimate`, `ci`, `p_adj` | Results after Holm correction |

## DiagnosticCall (Story 4)

| Field | Notes |
|---|---|
| `masked_task_id` | Random ID shown to the analyst |
| `analyst_label` | `information` / `reasoning` / `communication` / `none_single` |
| `evidence_seen` | The masked gain table used |
| `truth_label` | Construction label, or the annotators' consensus for naturalistic tasks (revealed only after all calls are locked) |
| `annotator_labels` | Naturalistic tasks: two labels plus κ |

## Relationships

```text
TaskFamily 1──* TaskInstance 1──* RunRecord *──1 Arm
TaskFamily 1──* CalibrationResult (one per model)
RunRecord 1──1 ResourceLedger
Prediction *──* (TaskFamily × Arm × Δ) cells
TaskInstance(split=heldout_blind) 1──1 DiagnosticCall
```
