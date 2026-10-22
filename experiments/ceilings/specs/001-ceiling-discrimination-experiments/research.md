# Phase 0 Research: Ceiling-Discrimination Experiment Suite

Each entry records a decision, why it was made, and what else was considered. Together they resolve every unknown in the plan's Technical Context.

## R1. Harness language and stack

- **Decision**: Python 3.12, a single installable package (`ceilings`) with a command-line entry point. Core libraries:
  - numpy and scipy for the exact information-theoretic quantities;
  - pandas and pyarrow for run tables;
  - statsmodels for equivalence tests and multiple-comparison correction;
  - pydantic v2 for record schemas;
  - pytest and hypothesis for tests;
  - the official `anthropic` Python SDK for Claude.
- **Rationale**: The paper's own checks are Python (`formal_proofs/verify*.py`), the machine already has Python 3.12, and every library needed is mature. pydantic gives schema-validated run records, which is how FR-012 (complete records) gets enforced.
- **Alternatives considered**: TypeScript (weaker scientific stack); R for the analysis (would split the codebase in two, and Python covers the tests needed).

## R2. Model families (FR-014)

- **Decision**: Three families from three different developers:
  1. **Claude Opus 5.5** (`claude-opus-5-5`, Anthropic's current default model).
  2. **A second commercial family**, to be named at the pilot.
  3. **An open-weights family** served locally, to be named at the pilot.

  A family is eligible only if it meets three criteria:
  - a controllable reasoning-depth setting;
  - independent sampling;
  - a usage report (or local count) of input, output and reasoning tokens per call.
- **Rationale**: Families 2 and 3 are chosen by those criteria at pilot time, so the plan doesn't depend on model names that may change before the run. The open-weights family serves SC-006 (third-party reproduction): it can be rerun with fixed weights and seeds, and it can expose token log-probabilities, which give a direct cross-check on the elicited probabilities.
- **Alternatives considered**: Claude only (fails FR-014). Three commercial APIs (no family that can be exactly reproduced, so SC-006 rests on providers keeping old model versions available).

## R3. The compute knob for arm K (and A2)

- **Decision**: Compute is increased in two ways, both logged in tokens:
  - **Depth**: the reasoning-effort setting. On Claude Opus 5.5 this is `output_config.effort` ∈ {low, medium, high, xhigh, max}. Thinking is always on for this model, its default effort is `medium`, and it rejects explicit thinking budgets. Baseline is pinned to `low`, set explicitly.
  - **Breadth**: N independent samples combined by majority vote, or by averaging probabilities for log-loss.

  Each Δ level is hit by choosing the (effort, N) pair whose pilot-measured mean token cost is closest to the target Δ.
- **Rationale**: Effort comes in discrete steps, so depth alone can't hit a target cost precisely. Adding samples fills the gaps in steps of one baseline-call cost. Because effort is the only reasoning knob on Opus 5.5, depth must be a categorical setting, not a token budget.
- **Alternatives considered**: Explicit thinking-token budgets (rejected with a 400 error on Opus 5.5). Samples only (would leave the depth form of compute untested, though the paper's Theorem 3 says the two should be interchangeable at the same total; we test that as a side result).

## R4. Matching cost across arms (FR-008, FR-009, SC-004)

- **Decision**: The currency is **tokens processed**: all input + output + reasoning tokens, summed over all agents, plus a fixed charge per tool call (default: the token length of the tool's response format, so a call costs what reading its answer costs).
  - Matching is done in the **pilot**: each arm's configuration is tuned until its mean added cost is within ±5% of Δ.
  - In the **main run**, the realized cost of every run is recorded. The analysis reports gains both at nominal Δ and as gain-versus-realized-cost curves, interpolating where an arm missed its target.
- **Rationale**: Exact matching per call is impossible with stochastic reasoning length, but matching on average plus a curve-based analysis makes "unequal spending" testable and correctable. Tokens are counted *processed*, not billed, so prompt caching and batch discounts don't distort the currency.
- **Alternatives considered**: Dollars (vary by provider and over time; kept as a secondary sensitivity currency). Wall-clock time (dominated by infrastructure noise).

## R5. Eliciting probabilities for log-loss and usable information (FR-012, FR-013)

- **Decision**: Every final answer is requested through **structured output**: a probability for each possible answer, plus the single chosen answer.
  - Primary log-loss uses the elicited probabilities after calibration (temperature scaling fitted on a held-out calibration split per model × family).
  - A secondary estimate uses sample frequencies from the K arm.
  - On the open-weights family, token log-probabilities give a third estimate.
- **Rationale**: Claude Opus 5.5 does not expose token log-probabilities and rejects sampling-parameter overrides, so elicited probabilities are the common denominator across families. Calibration matters because the paper's estimator of usable information (§VII) assumes calibrated predictions; uncalibrated overconfidence would show up as negative usable information.
- **Alternatives considered**: Correctness only, with no probabilities (would rule out the usable-information checks in Story 2).

## R6. Refusals and model fallbacks

- **Decision**: Model fallbacks are **disabled** for all experiment calls. A refusal is recorded with its category and scored as wrong (spec edge case), and refusal rates are reported per arm.
- **Rationale**: Anthropic's guidance is to enable server-side refusal fallbacks by default for `claude-opus-5-5`. We depart from it deliberately: a fallback silently re-runs the request on a different model, which would mix two models inside one "model family" cell and break FR-014's per-family attribution. The scenarios are benign (incident triage, hiring decisions with synthetic data), so refusals are expected to be rare. The spec's >5% refusal trigger catches it if they aren't.
- **Alternatives considered**: Enable fallbacks and discard any run where one fired (adds complexity and biases the sample toward non-refused items).

## R7. Execution at scale and cost control

- **Decision**:
  - Single-agent and fixed-protocol arms run through each provider's **batch interface** where available. On Claude that is Message Batches, which costs 50% and has results keyed by `custom_id` = run ID.
  - Interactive multi-agent arms (M, A1, A2, mixed) run through the asynchronous client with bounded concurrency.
  - All runs are **resumable**: a run is skipped if its record already exists, and its ID is a hash of (instance, arm, Δ, model, seed).
- **Rationale**: Multi-agent protocols need each agent to see the others' messages, so they can't be one batch request. Each round of a protocol, however, *can* be batched across instances. The runner batches round by round for the cheap arms and falls back to async calls for the rest.
- **Planning cost**: 5 families × 7 arms × 3 Δ × 3 models × ~850 instances ≈ 270k team-runs, before the multi-call cost of K and A2. The 10% pilot sets the real per-run cost and gates the main budget.

## R8. Enforcing the message budget (FR-011)

- **Decision**: Agents never share state. All inter-agent traffic goes through a `Channel` object that:
  - counts tokens in normal mode;
  - in **bit mode** (Story 2 sweeps), accepts only messages matching `[01]{0,L}`, rejecting the rest and asking the agent to retry once (a second failure is sent as the empty message).

  Each agent's prompt names its budget.
- **Rationale**: Theorem 5 is about bits crossing a cut. Bit mode makes the measured budget exactly L bits, so the inner-product threshold test (H7) compares against the paper's bound with no conversion between tokens and bits.
- **Alternatives considered**: Token caps only (an L-token message can carry far more than L bits, which would blur the threshold).

## R9. Task construction and the calibration gate (FR-001–FR-007)

- **Decision**:
  - **Information family**: a hidden k-bit value (k ∈ {3, 4, 6}) is generated per instance by a seeded random generator, so it can't be in training data. A simulated probe tool returns one requested bit, flipped with probability p ∈ {0, 0.1, 0.2}. The scenario text is incident triage ("which of 2^k hosts is failing").
  - **Reasoning family**: chained constraint puzzles (scheduling or assignment) with a unique solution and depth d ∈ {4, 8, 16}, all evidence in context.
  - **Communication family**: two agents, n-bit private vectors (n ∈ {8, 16, 32}).
    - Parity-like variant: is the total count of flagged items even?
    - Inner-product-like variant: is the number of matching flagged requirements odd? This is a hiring scenario where one agent holds a candidate's skills and the other holds the job's requirements.
    - Deterministic computation aids (a calculator tool that runs only on data the agent holds) are allowed and logged as reasoning resources.
  - **Mixed families** are compositions: probe a hidden value, then solve a depth-d puzzle parameterized by it; or a split-evidence task whose combination step needs depth d.
  - The **calibration gate** runs on the pilot. Each family's difficulty (k, d, n) is tuned so that baseline is within 5 points of chance and the matching full intervention is ≥90%.
- **Rationale**: Every family has exactly computable H(Θ), held information and budget (FR-006), using the same formulas as `litreview-coffee-law-check.py`. The natural-language wrappers satisfy FR-005.

## R10. Statistics (FR-016, FR-017)

- **Decision**:
  - Paired cluster bootstrap over instances (10,000 resamples) for confidence intervals on gains.
  - Two one-sided tests (TOST) at ±5 percentage points for every "no effect" prediction.
  - Holm correction across the prediction set H1–H9.
  - **Revision after the local test runs (see `TESTING.md`):**
    - Binary paired outcomes use exact or score methods. Superiority uses a one-sided exact McNemar test. TOST inverts the Newcombe (1998, method 10) paired-difference interval. The earlier t-based TOST returned p = 0 whenever all differences were identical.
    - There are two Holm families per model: support tests and falsification tests. Each comparison is counted once.
    - A verdict is "not evaluable" when the gate failed, n is below the pre-registered size, or an arm is missing.
  - Interaction term for Story 3 from a logistic mixed model, with instance as a random effect.
  - For H7, a change-point model fit against a linear model, compared by held-out log-likelihood.
  - The power analysis uses pilot variances. The planning figure: about 850 paired instances at a within-instance correlation of 0.5 (outcome SD 0.5, ±5-point margin, α = 0.05, 80% power).
  - **Correction (implementation):** the first draft gave ~620, using z₀.₈ = 0.84. For TOST with a true difference of zero, both one-sided tests must pass, so the right term is z₀.₉ = 1.28: n = ((1.645 + 1.282) × σ_d / 0.05)² ≈ 820–850 with σ_d ≈ 0.49. `tests/unit/test_analysis.py::test_tost_power_and_rejection` checks the corrected figure by simulation.
- **Rationale**: These are standard and pre-registerable, and they fit paired designs.

## R11. Pre-registration and integrity (FR-015, FR-019, SC-007)

- **Decision**:
  - `prereg/config_frozen.json` holds every prediction, margin, sample size, exclusion rule and analysis choice.
  - Its SHA-256 is recorded in `prereg/HASH` and checked by the analysis code at load time.
  - Every departure is logged in `prereg/DEVIATIONS.md`.
  - The frozen file is published (e.g. on OSF) before the main run.
  - All generators are seeded and every instance is stored, so the dataset can be released exactly.
- **Rationale**: A hash checked at load time makes SC-007 checkable by machine: any edit to the frozen file after the freeze is detected, not just discouraged.

## R12. Blind diagnostic (Story 4, FR-018)

- **Decision**:
  - A script produces *masked* intervention tables (family names replaced by random IDs), and the analyst classifies from those alone.
  - Naturalistic tasks come from three kinds of public benchmark task, each matched to the ceiling annotators are expected to assign:
    - closed-book knowledge questions with retrieval available (expected: information-bound);
    - multi-step math problems (expected: reasoning-bound);
    - a distributed-information collaboration task like the paper's reference [23] (expected: communication-bound).
  - Two annotators label each item's expected ceiling before any runs; agreement is reported as Cohen's κ.
- **Rationale**: Masking keeps the analyst blind without needing a second team. The benchmark kinds were picked because each has a strong a-priori expected ceiling, which makes disagreements informative.
