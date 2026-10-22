---

description: "Task list for the Ceiling-Discrimination Experiment Suite"
---

# Tasks: Ceiling-Discrimination Experiment Suite

**Input**: Design documents from `specs/001-ceiling-discrimination-experiments/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ (cli.md, interfaces.md, run-record.schema.json), quickstart.md

**Tests**: Included. The plan's Testing section and quickstart Scenarios 0–1 require unit tests for the theory module and generators, contract tests for the adapter, channel, schema and pre-registration, and a stub-model integration test. These guard the experiment's validity (wrong theory numbers or leaky generators would invalidate every result), so they are written before the code they cover.

**Organization**: Tasks are grouped by user story (US1–US4 = spec Stories 1–4).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story the task belongs to (US1–US4)
- Paths are relative to the project root `experiments/ceilings/`

## Conventions for every task

- Python 3.12, `src/` layout, package `ceilings`. Public functions get type hints and short docstrings.
- All randomness flows from explicit seeds (`numpy.random.Generator`). Nothing reads global RNG state.
- All IDs are SHA-256 content hashes (helper `content_hash` in `src/ceilings/records.py`).
- No model calls in unit or contract tests; they use `stub-oracle` (`src/ceilings/adapters/stub.py`).
- Reference numbers for the theory come from `litreview-coffee-law-check.py` at the repository root.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create `pyproject.toml`:
  - package `ceilings`, src layout, `requires-python = ">=3.12"`;
  - dependencies: `anthropic`, `numpy`, `scipy`, `pandas`, `pyarrow`, `statsmodels`, `pydantic>=2`, `jsonschema`, `cryptography`;
  - extras `dev = [pytest, hypothesis, ruff]`;
  - console script `ceilings = "ceilings.cli:main"`.
- [X] T002 Create the package and test skeleton from plan.md (empty `__init__.py` files):
  - `src/ceilings/`, `src/ceilings/tasks/`, `src/ceilings/adapters/`, `src/ceilings/analysis/`;
  - `tests/unit/`, `tests/contract/`, `tests/integration/`;
  - plus `prereg/`, `data/` and `reports/` directories.
- [X] T003 [P] Add ruff configuration (line length 100, rules E,F,I,B,UP) to `pyproject.toml` and a `.gitignore` excluding `data/`, `reports/`, `blind/key*`, `.venv/`, `__pycache__/` at `.gitignore`.
- [X] T004 [P] Create `prereg/config.json` with the pre-registration skeleton:
  - top-level keys `predictions` (H1–H9 with `statement`, `paper_source`, `test`, `cells`, `margin`), `families`, `arms`, `delta_levels`, `models`, `sample_sizes`, `exclusion_rules`, `analysis`;
  - prediction statements and falsification conditions copied verbatim from the spec's prediction table;
  - numeric fields set to `null`, to be filled after the pilot.
- [X] T005 [P] Create `prereg/DEVIATIONS.md` with a header and a table template (`date | section | planned | actual | reason | impact`).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Theory, records, ledger, tool, channel, adapter interface, pre-registration and runner core that every story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for the foundation (write first; they must fail before implementation)

- [X] T006 [P] Write `tests/unit/test_theory.py` asserting that `ceilings.theory` reproduces, to 1e-3:
  - `bsc_capacity(0.1) = 0.531`;
  - two-round budget 1.062;
  - mutual information 1.062 (bit 0 then bit 1), 0.742 (bit 0 twice) and 0.982 (the paper's adaptive policy, verify.py scenario 7);
  - MAP error 0.19;
  - one-look usable information 0.2655 (first bit random, index unseen);
  - Corollary 2 bounds 9.0 (k=10, C=1, ε=0) and 15.066 (k=10, C=0.531, ε=0.1);
  - IP n=4 best one-bit message 0.1222 and raw-prefix values 0.0597/0.1222/0.2472/0.4972/0.9972;
  - `ip_bound(n=100, L)` = 0.00282 / 0.08746 / 1.0 at L = 40 / 45 / 49.
- [X] T007 [P] Write `tests/contract/test_schema.py`: build a valid `RunRecord` via `ceilings.records`, serialize it, and validate it with `jsonschema` against `specs/001-ceiling-discrimination-experiments/contracts/run-record.schema.json`. Also check that records missing `ledger.total_tokens`, with probabilities not summing to 1 ± 1e-6, or with an unknown `arm_id` are rejected.
- [X] T008 [P] Write `tests/contract/test_channel.py`:
  - in `bits:L` mode, `"0101"` is accepted for L=4, while `"01012"` and `"01010"` (too long) are rejected;
  - one retry is allowed, then the empty message is delivered;
  - in `tokens:B` mode, cumulative tokens per direction never exceed B and truncation is logged;
  - every delivery appears in the ledger.
- [X] T009 [P] Write `tests/contract/test_tools.py`:
  - over 20,000 seeded calls, `bit_probe` flip frequency is within 0.01 of `flip_rate`;
  - outputs always lie in `answer_alphabet`;
  - malformed queries return the fixed error symbol;
  - each call adds `charge_tokens` to the ledger.
- [X] T010 [P] Write `tests/contract/test_prereg.py`: `freeze` writes `prereg/config_frozen.json` and `prereg/HASH`; `check_hash()` passes when unchanged and raises `PreregHashMismatch` (CLI exit code 3) after any byte change.
- [X] T011 [P] Write `tests/contract/test_adapter_stub.py`: `StubOracleAdapter` returns the contract shape `{final_answer, probabilities, refusal, usage{input,output,reasoning}, request_id}`, has probabilities over exactly `answer_space`, is deterministic per seed, and reports usage greater than 0.

### Implementation for the foundation

- [X] T012 [P] Implement `src/ceilings/theory.py` with:
  - `entropy(p)`, `binary_entropy(p)`, `mutual_information(joint)` (from a dict of (θ, obs) → prob);
  - `bsc_capacity(p)`, `budget(I0, capacities)`, `fano_floor(H, budget, M)` (Theorem 1 form);
  - `cor2_min_rounds(k, C, eps)`, `ip_bound(n, L)`;
  - `best_one_bit_message_ip(n)` (exhaustive, n ≤ 4);
  - `map_error(joint)`.

  Make T006 pass.
- [X] T013 [P] Implement `src/ceilings/records.py`: pydantic v2 models `TaskFamily`, `CalibrationResult`, `TaskInstance`, `ToolSpec`, `TheoryQuantities`, `Arm`, `ResourceLedger`, `RunRecord`, `Prediction`, `DiagnosticCall`, with the fields and validation rules in `data-model.md`, plus `content_hash(obj) -> str` (SHA-256 of canonical sorted-key JSON). Make T007 pass.
- [X] T014 [P] Implement `src/ceilings/ledger.py`: a `Ledger` class that accumulates per-agent `{input_tokens, output_tokens, reasoning_tokens, calls}`, tool calls with `charged_tokens`, and messages with tokens/bits, and exposes `total_tokens` and `to_record() -> ResourceLedger`.
- [X] T015 [P] Implement `src/ceilings/tools.py`: a `ToolSimulator` for `bit_probe` and `oracle` per `contracts/interfaces.md`, using a seeded RNG stream per (instance, run) and charging the ledger. Make T009 pass.
- [X] T016 [P] Implement `src/ceilings/channel.py`: the `Channel` class with `tokens:B` and `bits:L` modes, regex `^[01]{0,L}$`, a one-retry policy and ledger/transcript recording. Token counting goes through an injected `count_tokens(text, model_id)` callable. Make T008 pass.
- [X] T017 [P] Implement `src/ceilings/prereg.py`: `freeze(config_path)`, `check_hash()`, `load_frozen()`, and `log_deviation(section, planned, actual, reason, impact)` appending to `prereg/DEVIATIONS.md`. Make T010 pass.
- [X] T018 Implement `src/ceilings/adapters/base.py`, depending on T013:
  - abstract `ModelAdapter` with `async complete(messages, depth, answer_space, seed) -> AdapterResult`;
  - `count_tokens(text) -> int`;
  - `supports_batch: bool`;
  - an `AdapterResult` dataclass;
  - a shared helper `answer_schema(answer_space)` that builds the structured-output JSON schema `{answer: enum, probabilities: {each answer: number}}`;
  - `normalize_probabilities()`, which renormalizes and clips at 1e-6.
- [X] T019 Implement `src/ceilings/adapters/stub.py`: `StubOracleAdapter`, which returns the correct answer with probability 0.99 when the answer is derivable from the evidence it is given (it reads a `__oracle_answer__` marker that test fixtures inject) and a uniform guess otherwise, with fixed synthetic usage. Make T011 pass.
- [X] T020 Implement `src/ceilings/adapters/anthropic_claude.py`: `ClaudeAdapter(ModelAdapter)` using the official `anthropic` SDK. Model `claude-opus-5-5`.
  - Pass `output_config={"effort": depth, "format": <answer_schema>}` with the effort always set explicitly.
  - Send no `thinking` budget and no sampling parameters (the model rejects them).
  - **Do not** pass `fallbacks` (research R6).
  - Stream when `max_tokens > 16000`.
  - On `stop_reason == "refusal"`, record `stop_details.category`/`explanation` and return `final_answer=None`.
  - Read usage from `response.usage` (`input_tokens`, `output_tokens`) and set `reasoning_tokens=None`.
  - Mark the run invalid if `response.model` differs from the requested model.
  - Implement `count_tokens` via `client.messages.count_tokens`.
  - Add a batch helper using `client.messages.batches.create` with `custom_id=run_id`, polling until `processing_status == "ended"` and keying results by `custom_id`.
- [X] T021 Implement `src/ceilings/orchestrator.py`, the base `TeamOrchestrator`:
  - protocol steps: (1) build per-agent context from `shared_evidence` + `private_evidence`; (2) tool rounds through `ToolSimulator`; (3) message rounds through `Channel`; (4) a final answer from the designated answerer;
  - a single-agent and a two-agent configuration;
  - every adapter call recorded in the `Ledger` and transcript.
- [X] T022 Implement `src/ceilings/runner.py`:
  - `run_grid(instances, arms, deltas, models, split, max_cost, concurrency, batch)`;
  - `run_id` = `content_hash(instance_id, arm_id, delta_level, model_id, replicate)`;
  - skip runs whose record already exists in `data/runs/<model>/<family>/<shard>.jsonl` (resume);
  - stop with exit code 4 when the accumulated `ledger.dollars` estimate exceeds `max_cost`;
  - write transcripts to `data/runs/<model>/<family>/transcripts/<run_id>.json`;
  - call `prereg.check_hash()` before any main-split run.
- [X] T023 Implement the skeleton of `src/ceilings/cli.py`: an `argparse` entry point `main()` with subcommands `generate, theory, calibrate, match, prereg, run, sweep, analyze, mask, unmask, audit, report` as stubs that raise `NotImplementedError`, and the global exit-code mapping from `contracts/cli.md` (0, 2, 3, 4). `prereg freeze` is wired to `prereg.freeze`.

**Checkpoint**: Foundation ready. `pytest tests/unit/test_theory.py tests/contract -q` passes (quickstart Scenario 0, theory part).

---

## Phase 3: User Story 1 - Core diagonal test on constructed tasks (Priority: P1) 🎯 MVP

**Goal**: Three families whose binding ceiling is known by construction (info, reason, comm_parity/comm_ip), run under baseline + E, K, M, A1, A2 (+ single-agent reference), P1, P2 at matched Δ on three model families. Output: a gain table and verdicts for H1–H5.

**Independent Test**: Quickstart Scenarios 1–5. A stub dry run passes `audit`; families pass the calibration gate; arms are matched to ±5%; `analyze` writes `reports/gain_table.csv` and verdicts H1–H5.

### Tests for User Story 1 (write first)

- [X] T024 [P] [US1] Write `tests/unit/test_tasks_info.py`:
  - same seed gives an identical instance and `instance_id`;
  - the hidden answer's string forms never occur in `shared_evidence` or `private_evidence`;
  - `theory.H_theta == k`, `budget_by_T[T] == T*(1-h(p))`;
  - `fano_floor_by_T` matches `theory.fano_floor`.
- [X] T025 [P] [US1] Write `tests/unit/test_tasks_reason.py`: every generated puzzle has exactly one solution (checked by a brute-force solver in the test), its depth equals `d`, and all needed evidence is in `shared_evidence` (`I_held_team == H_theta`).
- [X] T026 [P] [US1] Write `tests/unit/test_tasks_comm.py`: for `comm_parity` and `comm_ip` with n ∈ {8, 16}, exact enumeration over a 2,000-instance sample gives `I_held_by_agent ≤ 0.01` bits for each agent and `I_held_team == H_theta`. The natural-language wrapper round-trips to the same bit vectors through `formal_mapping`.
- [X] T027 [P] [US1] Write `tests/unit/test_arms.py`: for each arm, the orchestrator config differs from baseline in exactly one resource:
  - E: evidence;
  - K: depth/samples;
  - M: message budget;
  - A1: agent count with equal totals;
  - A2: agent count + compute;
  - P1: irrelevant evidence;
  - P2: padding.

  Also check that P1 text shares no answer-bearing tokens with the instance.
- [X] T028 [P] [US1] Write `tests/unit/test_analysis_core.py` on synthetic paired data:
  - the paired bootstrap CI covers a known gain of 0.10 in at least 93 of 100 seeded repetitions;
  - TOST at ±0.05 declares equivalence for a true gain of 0 with n=850 paired and ρ=0.5 in at least 75% of repetitions, and rejects equivalence for a true gain of 0.10;
  - the Holm adjustment matches `statsmodels`.
- [X] T029 [US1] Write `tests/integration/test_story1_stub.py` (quickstart Scenario 1): generate 50 info instances (k=4, p=0.1), run baseline,E,K,M,A1,P1,P2 at `small` Δ with `stub-oracle`, then assert that E accuracy > 0.9, baseline is within 0.15 of chance, every record validates against the schema, and `audit` exits 0.

### Implementation for User Story 1

- [X] T030 [P] [US1] Implement `src/ceilings/tasks/info.py`:
  - `generate(seed, k, p, n_rounds_max, split)`: a hidden k-bit answer from the seeded RNG, an incident-triage scenario ("one of 2^k hosts is failing; a probe reports one bit of its ID, wrong with probability p"), and a `ToolSpec(kind="bit_probe", answer_alphabet=["0","1","ERR"], flip_rate=p, capacity_bits=1-h(p), charge_tokens=<len of tool-result template>)`;
  - an `oracle_evidence(instance, amount)` helper for arm E that returns the passage containing the host ID;
  - `theory` filled from `ceilings.theory`.

  Make T024 pass.
- [X] T031 [P] [US1] Implement `src/ceilings/tasks/reason.py`: `generate(seed, d, split)` builds a scheduling/assignment constraint puzzle (meeting slots and people) with a unique solution reachable only through a chain of d dependent deductions. All constraints go in `shared_evidence`, the answer space is the candidate assignments for the queried item, and `restated_evidence(instance, amount)` (used by arm E on this family) paraphrases constraints already given. Make T025 pass.
- [X] T032 [P] [US1] Implement `src/ceilings/tasks/comm.py`: `generate(seed, n, variant, split)` with `variant ∈ {"parity","ip"}`.
  - Hiring scenario: agent 1 holds the candidate's n yes/no skills (x), and agent 2 holds the role's n yes/no requirements (y).
  - parity: "is the total number of yes answers across both lists even?"
  - ip: "is the number of requirements the candidate meets odd?" (⟨x,y⟩ mod 2).
  - Each list is presented as n numbered lines.
  - `same_kind_private_evidence(instance, amount)` (arm E on this family) adds unrelated attributes of the same format.
  - Fill `theory` (`comm_bound_by_L` from `theory.ip_bound`).

  Make T026 pass.
- [X] T033 [US1] Implement `src/ceilings/tasks/__init__.py`: a `FAMILIES` registry mapping `family_id` → generator plus the arm-E evidence function, and `load_instances(family, version, split)` reading `data/instances/<family>/<version>/<split>.jsonl`.
- [X] T034 [US1] Implement `src/ceilings/arms.py`:
  - `ArmConfig` per arm, following `data-model.md` `Arm.config`;
  - `build_arm(arm_id, delta_level, family, pilot_table)` returning the orchestrator config;
  - P1 irrelevant-evidence generator: seeded filler drawn from a fixed public-domain corpus file `src/ceilings/data/filler.txt`, with any answer-bearing tokens removed;
  - P2 padding: repeated neutral separator lines.

  Make T027 pass.
- [X] T035 [US1] Extend `src/ceilings/orchestrator.py` with the per-arm protocols:
  - **K**: depth = effort level, plus N independent samples aggregated by majority vote for `final_answer` and by mean probability for `probabilities`, with `sample_answers` stored;
  - **M**: message rounds with the arm's budget;
  - **A1**: m agents sharing the baseline's total tool calls, depth and message budget, split evenly;
  - **A2**: m agents with baseline + Δ compute in total;
  - **A2_single_ref**: one agent with the same Δ compute;
  - **E, P1, P2**: context injection.
- [X] T036 [US1] Implement `src/ceilings/calibration.py`: `calibrate(family, model, n=200, max_cost)` runs baseline and the family's full matched intervention on the `calibration` split and computes `CalibrationResult`:
  - gate: |baseline − chance| ≤ 0.05 and matched_full ≥ 0.90;
  - `leak_rate` for info families, with leaked instances flagged `excluded`;
  - `refusal_rate`.

  It writes `data/calibration/<family>__<model>.json` and sets the family state to `ready` or `rejected`.
- [X] T037 [US1] Add `match_arms(family, model, deltas, n=100)` to `src/ceilings/arms.py`: run pilot instances for each arm config candidate (K: the grid of effort × N; E/P1/P2: amount; M: budget; A1/A2: agent count), choose the config whose mean `added_vs_baseline` is closest to each target Δ, set `matched` when within ±5%, and write `data/arms/<family>/<model>.json`.
- [ ] T038 [P] [US1] Implement `src/ceilings/adapters/family_b.py`, the second commercial family. **Needs a researcher decision first:** record the chosen family and model ID in `prereg/config.json` under `models`, using research R2's criteria (controllable reasoning depth, independent sampling, usage reporting, a different developer). Implement the same `ModelAdapter` contract with no silent model substitution, and record usage from the provider response.
- [ ] T039 [P] [US1] Implement `src/ceilings/adapters/open_weights.py`, the locally served open-weights family. **Needs a researcher decision first:** record the model and the server endpoint in `prereg/config.json`. Implement the contract using constrained decoding to the answer schema, local token counting, a fixed seed per replicate, and token log-probabilities for the answer tokens stored in the transcript.
- [X] T040 [US1] Implement `src/ceilings/analysis/gains.py`:
  - load run records into a pandas DataFrame;
  - paired cluster bootstrap (10,000 resamples, instance clusters) for gain vs baseline per family × arm × Δ × model;
  - gain-versus-realized-cost curves with linear interpolation to nominal Δ;
  - `gain_table()` → `reports/gain_table.csv`.
- [X] T041 [US1] Implement `src/ceilings/analysis/equivalence.py`: `tost(diff_samples, margin=0.05)` on paired differences, `superiority(diff_samples)`, and `holm(pvals)` via `statsmodels.stats.multitest`. Make T028 pass.
- [X] T042 [US1] Implement `src/ceilings/analysis/report.py`, part 1: `assign_verdicts()` for H1–H5, using the tests named in the frozen pre-registration and the margin from `prereg/config_frozen.json`. A verdict is `supported` / `contradicted` / `inconclusive` (FR-020), and it requires agreement in at least 2 of 3 model families (SC-005). Output goes to `reports/predictions.json`.
- [X] T043 [US1] Implement `src/ceilings/audit.py`, which checks:
  - every record's `prereg_hash` matches;
  - ledger totals equal the sum of components;
  - bit-mode transcripts match `^[01]{0,L}$`;
  - each arm's realized mean added cost is within ±5% of target (SC-004);
  - refusal rate is ≤ 5% per arm.

  Results go to `reports/audit.json`, with exit code 2 on failure.
- [X] T044 [US1] Wire the CLI subcommands `generate`, `theory`, `calibrate`, `match`, `run`, `analyze`, `audit` in `src/ceilings/cli.py` to the modules above, with the options listed in `contracts/cli.md`. Make T029 pass.
- [ ] T045 [US1] Fill the H1–H5 entries in `prereg/config.json`: family versions, arms, Δ targets from `data/arms/`, margins (0.05), sample sizes from the pilot power analysis (variance from the calibration and match runs; target ≥80% power for TOST), and exclusion rules (leakage, malformed). Then run `ceilings prereg freeze`.

**Checkpoint**: US1 is fully runnable. The stub dry run passes, and with real models, quickstart Scenarios 2–5 produce the gain table and H1–H5 verdicts.

---

## Phase 4: User Story 2 - Quantitative ceiling checks (Priority: P2)

**Goal**: Fano-floor sweeps on info tasks (H6), the communication threshold signature (H7), and usable ≤ available ≤ budget from calibrated probabilities

**Independent Test**: Quickstart Scenario 6. Sweeps over T × p (info) and L in bit mode (comm) produce `reports/predictions.json` entries for H6 and H7 and a usable-information figure.

### Tests for User Story 2 (write first)

- [X] T046 [P] [US2] Write `tests/unit/test_usable_info.py`:
  - temperature scaling fitted on a calibration split lowers held-out log-loss on synthetic overconfident predictions;
  - `usable_information = H − logloss` equals the exact mutual information within 0.02 bits for a synthetic Bayes-optimal predictor on the 4-server example;
  - it is ≤ 0.2655 + 0.02 for a predictor that sees only one answer (first bit random, index unseen).
- [X] T047 [P] [US2] Write `tests/unit/test_changepoint.py`: on synthetic accuracy-vs-L data generated from a flat-then-rising curve (change at L=8, n=16), the change-point model wins on held-out log-likelihood. On linear data, the linear model wins.

### Implementation for User Story 2

- [X] T048 [US2] Add sweep mode to `src/ceilings/runner.py` and the `sweep` subcommand to `src/ceilings/cli.py`:
  - info: grid T ∈ {0,1,2,4,8,12,16} × flip_rate ∈ {0,0.1,0.2}, with the number of tool rounds fixed to T;
  - comm: L ∈ {0,1,2,4,8,n/2,n} with `Channel` in `bits:L` mode.

  Records are tagged with `sweep{T, flip_rate, L_bits}`.
- [X] T049 [P] [US2] Implement `src/ceilings/analysis/usable_info.py`: per model × family, fit temperature scaling on the `calibration` split; compute held-out log-loss and `usable = H_theta − logloss` with a bootstrap CI; compare against the instance's `I_held_team` and `budget_by_T`; flag any `usable > available + CI`.
- [X] T050 [P] [US2] Implement `src/ceilings/analysis/changepoint.py`: fit (a) a logistic-linear model of accuracy vs L and (b) a two-segment model (flat at chance until L*, then logistic rise) by maximum likelihood, and compare them by 5-fold held-out log-likelihood. Return the winning model, L* with a CI, and the parity-variant jump at L=1.
- [X] T051 [US2] Extend `src/ceilings/analysis/report.py` with verdicts:
  - **H6:** error vs T stays above `fano_floor_by_T` (one-sided test per T and p). Any significant violation is marked `audit_required`, with a leakage check rerun before the verdict.
  - **H7:** change-point beats linear for `comm_ip`, and parity reaches ≥ 0.9 of its ceiling at L=1.
  - **Figures:** `reports/fig_fano.png` and `reports/fig_comm_threshold.png` via matplotlib (add `matplotlib` to the dev extras in `pyproject.toml`).
- [ ] T052 [US2] Fill the H6 and H7 entries (sweep grids, tests, sample sizes) in `prereg/config.json` before freezing. If the freeze already happened in T045, log the addition with `prereg.log_deviation` and refreeze with a new hash, recording both hashes.

**Checkpoint**: US2 adds quantitative verdicts using US1's generators. It is independently testable through its own sweeps.

---

## Phase 5: User Story 3 - Two ceilings at once (Priority: P3)

**Goal**: Mixed families (information+reasoning, communication+reasoning) with 2×2 interventions, testing for a positive interaction (H8)

**Independent Test**: Run `mix_info_reason` and `mix_comm_reason` under baseline, E, K, M, E+K, M+K, then check that `reports/predictions.json` has an H8 interaction estimate with a CI. With the single interventions only, the diagnostic reports `none_single`.

### Tests for User Story 3 (write first)

- [ ] T053 [P] [US3] Write `tests/unit/test_tasks_mixed.py`: `mix_info_reason` needs both the hidden value (I_held_team at baseline = 0) and depth-d reasoning; `mix_comm_reason` agents each hold ≤ 0.01 bits alone; the theory quantities compose correctly.
- [ ] T054 [P] [US3] Write `tests/unit/test_interaction.py`: on synthetic 2×2 data with a known positive interaction, the mixed-model interaction CI excludes 0. On additive data, the CI covers 0.

### Implementation for User Story 3

- [ ] T055 [US3] Implement `src/ceilings/tasks/mixed.py`:
  - `mix_info_reason`: a host ID probed via `bit_probe` parameterizes a depth-d scheduling puzzle, and the answer is the puzzle's solution;
  - `mix_comm_reason`: split hiring lists whose combination feeds a depth-d deduction.

  Register both in `src/ceilings/tasks/__init__.py`.
- [ ] T056 [US3] Add the paired arms `E+K` and `M+K` to `src/ceilings/arms.py` (each component at the same per-component Δ as its single arm, total cost 2Δ), and handle them in `src/ceilings/orchestrator.py`.
- [ ] T057 [US3] Implement `src/ceilings/analysis/interaction.py`: a logistic mixed model `correct ~ a*b + (1|instance)` per mixed family × model (statsmodels `BinomialBayesMixedGLM`, or a GEE with exchangeable correlation if convergence fails, decided in the pre-registration), reporting the interaction estimate and CI.
- [ ] T058 [US3] Extend `src/ceilings/analysis/report.py` with the H8 verdict, and add `diagnose_single(gain_row)`, which returns `none_single` when no single intervention reaches the pre-registered effect threshold while a pair does. Fill H8 in `prereg/config.json` (or log a deviation and refreeze, as in T052).

**Checkpoint**: US3 adds the interaction test, reusing US1's generators and arms.

---

## Phase 6: User Story 4 - Blind diagnostic on held-out and naturalistic tasks (Priority: P4)

**Goal**: An analyst who cannot see family labels classifies the binding ceiling from masked gain tables, first on held-out constructed tasks (H9), then on naturalistic tasks labelled in advance by two annotators

**Independent Test**: Quickstart Scenario 7. `mask` → the analyst fills `blind/calls.json` → `unmask` → `reports/diagnostic.json` holds accuracy with a CI against 1/3, agreement on naturalistic tasks, and κ.

### Tests for User Story 4 (write first)

- [ ] T059 [P] [US4] Write `tests/unit/test_diagnostic.py`: the masked tables contain no family IDs, scenario text or arm-to-family hints; `unmask` refuses to run until every masked task has a locked call; accuracy and Cohen's κ match hand-computed values on a toy example.

### Implementation for User Story 4

- [ ] T060 [US4] Implement `src/ceilings/tasks/natural.py`: a loader for naturalistic items from `data/natural/<benchmark>.jsonl` (fields: `item_id`, `prompt`, `answer`, `answer_space`, `evidence_by_agent?`, `retrieval_corpus?`) for three benchmark kinds:
  - closed-book knowledge questions with a retrieval tool;
  - multi-step math;
  - a distributed-information collaboration task.

  **Needs a researcher decision first:** record the specific benchmarks and licences in `prereg/config.json`. Map arm E to retrieval, K to compute, and M to message budget.
- [ ] T061 [US4] Implement `scripts/annotate_export.py` and `scripts/annotate_import.py`: export naturalistic items to `blind/annotation/<annotator>.csv` (items only, no results), import the two label sets, and compute Cohen's κ. Labels are locked (hashed into `prereg/DEVIATIONS.md`) before any naturalistic runs.
- [ ] T062 [US4] Implement `src/ceilings/analysis/diagnostic.py`:
  - `mask(split)` writes per-task gain tables with random IDs to `blind/masked/*.csv` and the ID→truth key to `blind/key.enc`, encrypted with Fernet (`cryptography`) using a passphrase read from the `CEILINGS_BLIND_KEY` environment variable;
  - `lock_calls(path)` hashes the analyst's `blind/calls.json`;
  - `unmask(calls)` decrypts the key and computes accuracy vs 1/3 (binomial CI), agreement with the annotators, and κ.

  Make T059 pass.
- [ ] T063 [US4] Wire the `mask` and `unmask` subcommands in `src/ceilings/cli.py`. Add the H9 verdict to `src/ceilings/analysis/report.py`, and fill H9 in `prereg/config.json` (or log a deviation and refreeze).

**Checkpoint**: All four stories are runnable end to end.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final report, reproducibility release, documentation

- [ ] T064 Implement the `report` subcommand in `src/ceilings/cli.py` and `final_report()` in `src/ceilings/analysis/report.py`, writing `reports/final.md`:
  - a verdict per H1–H9 with estimates and CIs;
  - per-model breakdowns;
  - the gain table and figures;
  - the status of the paper's §IX predictions covered (SC-008);
  - every entry from `prereg/DEVIATIONS.md` (SC-007);
  - audit results.
- [ ] T065 [P] Implement `scripts/release_bundle.py`: package the generators' seeds and versions, all instances, the frozen pre-registration and hash, run records, transcripts, prompts and analysis outputs into `release/ceilings-<date>.tar.gz` with a `MANIFEST.json` of SHA-256 sums (FR-019, SC-006).
- [X] T066 [P] Write `README.md` at the repository root: purpose (testing the Coffee Law ceilings), install, the quickstart scenario order, cost warnings and `--max-cost`, the blind-diagnostic protocol, and links to `specs/001-ceiling-discrimination-experiments/` and `litreview-coffee-law.md`.
- [X] T067 [P] Add `tests/unit/test_reproduce.py`: regenerate 20 instances per family from stored seeds and assert byte-identical `instance_id`s. Rerun the stub integration twice and assert identical gain tables.
- [ ] T068 Run `ruff check src tests` and `pytest -q`, then quickstart Scenarios 0 and 1 (no spend), and record the results in `reports/smoke.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup and blocks every story.
- **US1 (Phase 3)**: depends on Foundational. It is the MVP.
- **US2 (Phase 4)**: depends on Foundational plus the US1 generators T030 and T032, and on T048's runner sweep mode. It does not need US1's analysis.
- **US3 (Phase 5)**: depends on the US1 generators T030–T032 and arms T034–T035.
- **US4 (Phase 6)**: depends on the US1 analysis T040–T042 (masked tables are built from gain tables). The naturalistic part also depends on the researcher decision in T060.
- **Polish (Phase 7)**: depends on whichever stories are in scope.

### Researcher decisions that block tasks

- T038 and T039: naming model families B and C (FR-014). US1's main run cannot start without them, but stub and Claude-only pilots can.
- T060: choosing the naturalistic benchmarks.
- T045 / T052 / T058 / T063: freezing the pre-registration. **The main run must not start before T045's freeze.** Later additions are logged deviations.

### Within each story

Tests come before implementation and must fail first. Generators come before arms, arms before the orchestrator protocols, then calibration, then matching, then analysis, then CLI wiring.

### Parallel Opportunities

- **Setup:** T003, T004 and T005 in parallel.
- **Foundational tests:** T006–T011 in parallel.
- **Foundational implementation:** T012–T017 in parallel (separate files), then T018 → T019/T020 → T021 → T022 → T023.
- **US1:** tests T024–T028 in parallel; generators T030, T031 and T032 in parallel; adapters T038 and T039 in parallel with the generators once the decisions are made.
- **US2:** T049 and T050 in parallel.
- **US3:** T053 and T054 in parallel.
- **Polish:** T065, T066 and T067 in parallel.

---

## Parallel Example: User Story 1

```bash
# Tests first, together:
Task: "Write tests/unit/test_tasks_info.py"
Task: "Write tests/unit/test_tasks_reason.py"
Task: "Write tests/unit/test_tasks_comm.py"
Task: "Write tests/unit/test_arms.py"
Task: "Write tests/unit/test_analysis_core.py"

# Then the three generators, together:
Task: "Implement src/ceilings/tasks/info.py"
Task: "Implement src/ceilings/tasks/reason.py"
Task: "Implement src/ceilings/tasks/comm.py"
```

---

## Implementation Strategy

### MVP first (User Story 1 only)

1. Phase 1 Setup, then Phase 2 Foundational (Scenario 0 passes).
2. Phase 3 through T037 with Claude + stub only, then run the pilot (calibration and matching) to get real costs and variances.
3. Decide families B and C (T038, T039), complete the pilot on all three, then freeze the pre-registration (T045).
4. Main run, then `analyze` and `audit`. **Stop and validate**: H1–H5 verdicts. This is the reviewer's requested experiment.

### Incremental delivery

1. US1, the diagonal test (reviewer ask).
2. US2, quantitative checks: the strongest evidence beyond direction.
3. US3, interactions (pre-empts the "real tasks have several ceilings" objection).
4. US4, the blind diagnostic (practitioner value).

### Cost control

Every model-calling command takes `--max-cost`. Run the pilot at about 10% scale before any main-run spend. Batch every single-call round on Claude (Message Batches, 50% cost).

---

## Notes

- [P] = different files, no dependency on incomplete tasks.
- Story labels map to spec Stories 1–4.
- Never change `prereg/config_frozen.json` by hand. Every change goes through `prereg.log_deviation` and a refreeze.
- Model fallbacks stay disabled (research R6). If a provider adds automatic substitution, the adapter must detect it and mark the run invalid.

## Implementation notes (2026-10-21, /speckit-implement)

Done: 50 of 68 tasks. All 68 tests pass in Docker on Python 3.12 and 3.10.

**Deviations from the task text:**

- T023/T044: `calibrate` and `match` are merged into one `ceilings prepare` command. The `mask`, `unmask` and `report` subcommands (US4/Polish) are not implemented yet.
- T029: the integration test uses the stub model for the dry run, and the simulated agent `sim-bounded` (`src/ceilings/agents.py`) for the end-to-end diagonal check.
- T034: P1 filler comes from fixed in-code lines, not `src/ceilings/data/filler.txt`.
- T036: the calibration gate applies FR-007's 5-point tolerance with a 95% sampling-error allowance (see the docstring in `calibration.py`).
- T037: `match_arms` lives in `src/ceilings/calibration.py`, to avoid a circular import with the orchestrator. Arms whose natural cost is below the target are topped up with neutral filler; arm P2 measures what filler alone does.
- T051: the figures are produced by `notebooks/ceilings_small_run.ipynb`, not by `report.py`.
- T028: the TOST power check uses n = 850. The planning figure was corrected from ~600 to ~850 (research R10, spec FR-017).
- Added (not in the task list):
  - `src/ceilings/agents.py`: the policy layer and the simulated bounded agent;
  - `src/ceilings/experiment.py`: shared helpers for the CLI and notebook;
  - `Dockerfile`, `.dockerignore`, `README.md` (T066);
  - `notebooks/ceilings_small_run.ipynb`: written but **not executed** here. It is to be run on GCP.
  - The supported Python version is lowered to 3.10 for GCP images.

**Not done:**

- Blocked on researcher decisions: T038 and T039 (families B and C), T045 and T052 (pre-registration fill and freeze after a real pilot).
- Not started: US3 (T053–T058), US4 (T059–T063), and T064, T065, T068.
