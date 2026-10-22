# Feature Specification: Ceiling-Discrimination Experiment Suite

**Feature Branch**: `001-ceiling-discrimination-experiments` (spec directory only; this folder is not a git repository, so no branch was created)

**Created**: 2026-10-21

**Status**: Draft

**Input**: User description: "Experiment suite that tests the Coffee Law paper's three ceilings (information, reasoning, communication) independently on the same set of tasks, under matched resource budgets. Conditions: (1) Information-limited: add relevant external evidence -> performance improves; (2) Reasoning-limited: increase compute with evidence fixed -> performance improves; (3) Communication-limited: increase message budget with private evidence fixed -> performance improves; (4) Negative control: add agents without changing evidence, compute, or communication -> no systematic gain beyond the matched resource budget. Goal: discriminate which ceiling binds per task, addressing the reviewer request for at least one experiment that discriminates the three ceilings under matched resource budgets."

## Background

The Coffee Law paper (Bhatt, 2026) claims that a team of agents can be limited by three different resources, and that each is lifted by a different intervention:

- **Information:** the answer is not in anything the team has observed. Only new evidence (tools, retrieval, an oracle) helps.
- **Reasoning:** the answer is determined by what the team holds, but the team cannot compute it. Only more computation helps.
- **Communication:** the evidence is split across agents. Only more exchange between them helps.

The paper reports no experiments of its own. Its diagnostic (§IX.1) and several predictions (§IX.2–6) are untested. A reviewer asked for at least one experiment that tells the three ceilings apart under matched resource budgets. This suite is that experiment.

**The core design idea:** build task families where the binding ceiling is known *by construction*. Then apply every intervention to every family at the same added cost, and check whether the gains form the predicted "diagonal" pattern: each intervention helps only its own family. Finally, check whether the diagnostic can recover the ceiling *blind*, on tasks whose ceiling the analyst is not told.

## User Scenarios & Testing *(mandatory)*

The primary user is the **researcher** running the study (the paper's author or a replicator). Secondary users are **reviewers**, who judge whether the evidence supports the theory, and **practitioners**, who want to know whether the diagnostic works on their own agent systems.

### User Story 1 - Core diagonal test on constructed tasks (Priority: P1)

The researcher runs three task families whose binding ceiling is known by construction: information-bound, reasoning-bound and communication-bound. Each family is run under a baseline and under each of the three interventions, all at the same added resource budget, plus the agent-count negative control. The researcher then reads off a 3 × 3 table of gains (task family × intervention) and the control results.

**Why this priority**: This is exactly what the reviewer asked for, and the minimum result that turns the decomposition from a conceptual claim into a tested one. Every other story builds on these task families and this budget accounting.

**Independent Test**: Can be fully tested by running the three families × (baseline + 3 interventions + 2 agent controls + 2 placebo controls) on at least three model families and producing the gain table with confidence intervals. It delivers a pass/fail verdict on the paper's central prediction.

**Acceptance Scenarios**:

1. **Given** an information-bound task whose answer exists only behind a tool, **When** the team receives relevant evidence at budget Δ, **Then** accuracy rises above baseline. **When** it instead receives Δ of extra compute, or Δ of extra message budget, **Then** accuracy is statistically equivalent to baseline.
2. **Given** a reasoning-bound task whose answer is fully determined by evidence already in context, **When** the team receives Δ of extra compute, **Then** accuracy rises. **When** it instead receives Δ of extra relevant-looking evidence (restating what it already has) or Δ of message budget, **Then** accuracy is equivalent to baseline.
3. **Given** a communication-bound task whose evidence is split so that no single agent holds any information about the answer alone, **When** the message budget is raised by Δ, **Then** accuracy rises. **When** each agent instead receives Δ of extra compute or Δ of extra private evidence of the same kind it already has, **Then** accuracy is equivalent to baseline.
4. **Given** any family, **When** agents are added while total evidence, total compute and total message budget stay fixed (control A1), **Then** accuracy is equivalent to baseline.
5. **Given** any family, **When** agents are added together with Δ extra total compute (control A2), **Then** the gain is equivalent to giving one agent the same Δ compute. This tests the paper's claim that only total compute counts (Theorem 3).
6. **Given** any family, **When** the team receives Δ of *irrelevant* evidence (placebo P1) or Δ of non-functional extra tokens (placebo P2), **Then** accuracy is equivalent to baseline. This shows that gains come from the content of an intervention, not from its length or cost.

---

### User Story 2 - Quantitative ceiling checks (Priority: P2)

On the same constructed families, the researcher checks the paper's *quantitative* predictions, not just their direction:

- the Fano error floor for information-bound tasks (Corollary 2);
- the communication "signature": a threshold jump for inner-product-like splits versus a rise from the first message bit for parity-like splits (Theorem 5, §IX.5);
- the chain usable ≤ available ≤ budget, with usable information estimated from the team's predicted probabilities.

**Why this priority**: The direction-only test in Story 1 could be passed by many theories. The quantitative checks are where the decomposition says something a single "capability" axis does not, so they carry most of the scientific weight after Story 1.

**Independent Test**: Can be tested by sweeping (a) the number of informative tool calls and the tool noise rate on information-bound tasks, and (b) the message budget on two communication-bound sub-families (parity-like and inner-product-like), then plotting the observed accuracy and usable information against the theoretical floors and ceilings.

**Acceptance Scenarios**:

1. **Given** an information-bound task with a uniform hidden answer of k bits and a yes/no tool with known error rate p, **When** the team makes T tool calls, **Then** its error rate never falls significantly below the Fano floor implied by the budget T·(1 − h(p)), and its estimated usable information never significantly exceeds that budget.
2. **Given** a parity-like split, **When** the message budget rises from 0 to 1 bit, **Then** accuracy jumps from chance to near the single-agent-with-everything ceiling.
3. **Given** an inner-product-like split of size n, **When** the message budget rises from 0 toward n bits, **Then** accuracy stays near chance until the budget approaches about n/2 bits and only then rises. The pre-registered prediction is "flat then rising", tested against the alternative "steadily rising".
4. **Given** any run, **When** usable information is estimated as answer entropy minus the team's held-out log-loss, **Then** the estimate is at most the computed available information, up to estimation error.

---

### User Story 3 - Two ceilings at once (Priority: P3)

The researcher runs mixed task families where two ceilings bind together (information + reasoning, and communication + reasoning) and applies each intervention alone and in pairs.

**Why this priority**: A reviewer can reasonably object that real tasks rarely have a single binding ceiling, and that a single-intervention diagnostic would then mislead. This story tests whether the decomposition correctly predicts *interaction*: neither intervention alone helps much, and both together help a lot.

**Independent Test**: Can be tested with a 2 × 2 design (each intervention off/on) on each mixed family, estimating the interaction term.

**Acceptance Scenarios**:

1. **Given** a task that needs a hidden fact *and* a long computation on it, **When** only evidence or only compute is added, **Then** the gain is small. **When** both are added, **Then** the gain is significantly larger than the sum of the two single gains (positive interaction).
2. **Given** a mixed task, **When** the diagnostic is applied using single interventions only, **Then** it reports "no single binding ceiling" rather than naming the wrong one.

---

### User Story 4 - Blind diagnostic on held-out and naturalistic tasks (Priority: P4)

An analyst who is not told the task family runs the three-intervention diagnostic and names the binding ceiling. They do this first on held-out constructed tasks, then on naturalistic tasks drawn from published benchmarks. For the naturalistic tasks, independent annotators label the expected ceiling before any results are seen.

**Why this priority**: This is the practitioner's question: does the diagnostic work when you don't already know the answer? It depends on Stories 1–3 to establish that the interventions are valid.

**Independent Test**: Can be tested by comparing the analyst's blind classifications with the construction labels (held-out constructed tasks) and with the pre-registered annotator labels (naturalistic tasks).

**Acceptance Scenarios**:

1. **Given** held-out constructed tasks with hidden family labels, **When** the analyst applies the diagnostic, **Then** the binding ceiling is named correctly well above chance (chance = 1 in 3).
2. **Given** naturalistic tasks with annotator labels fixed in advance, **When** the analyst applies the diagnostic, **Then** agreement with the annotators is reported with a confidence interval, together with a breakdown of disagreements by type.

---

### Edge Cases

- **Leakage of hidden facts.** An information-bound task scores above chance at baseline, which means the facts reached the model through training or the prompt. Such instances are flagged, excluded, and replaced. The baseline at-chance check is a gate, not a result.
- **Reasoning binds inside an information or communication task.** For example, a model cannot compute a parity even when it holds every bit. Each family has a calibration gate: with its own intervention fully applied, accuracy must reach at least 90%. Otherwise its difficulty is reduced, or deterministic computation aids (a calculator, or code run only on data the agent already holds) are allowed. Computation aids enlarge what agents can compute, not what they know, so they are part of the reasoning resource and are logged as such.
- **Agents send evidence through a side channel.** For example, they encode extra bits in message formatting, or a shared scratchpad leaks. The message budget is counted on the full content that crosses between agents, and the shared state is restricted to the message channel.
- **The tool returns more than intended.** For example, error messages reveal the answer. Tool outputs are restricted to a fixed answer alphabet in the quantitative sweeps, so that tool capacity is known exactly.
- **Extra compute degrades performance** (overthinking, hallucination). This is recorded as a negative gain, not truncated. The paper predicts zero gain on information-bound tasks; a significant *negative* gain is reported as a separate finding.
- **A model refuses or fails to produce a parseable answer.** The instance is scored as wrong, and refusal rates are reported per arm. A refusal rate above 5% in any arm triggers a prompt review before the main run.
- **Ceiling or floor effects.** Baseline accuracy that is too close to 100% or to chance leaves no room to detect a change. Difficulty is calibrated so that every family's baseline is near chance and its matched intervention can reach at least 90%.
- **One model family behaves differently from the others.** Results are reported per model family as well as pooled, and the verdict for each prediction requires it to hold in the majority of model families.

## Requirements *(mandatory)*

### Functional Requirements

**Task families**

- **FR-001**: The suite MUST provide an **information-bound family**. The answer is a uniformly random hidden value generated fresh for each instance, so that no model can know it in advance. It is reachable only through a tool whose answer alphabet and error rate are fixed and known.
- **FR-002**: The suite MUST provide a **reasoning-bound family**. The answer is fully determined by evidence placed in every agent's context, and difficulty is tunable by the amount of computation required (for example, the number of dependent steps).
- **FR-003**: The suite MUST provide a **communication-bound family** with two sub-families:
  - a *parity-like* split, where a one-bit summary from each agent suffices;
  - an *inner-product-like* split, where about n/2 bits must be exchanged.

  In both, each agent's private evidence alone carries essentially no information about the answer, while the agents' combined evidence determines it.
- **FR-004**: The suite MUST provide **mixed families**: information + reasoning, and communication + reasoning.
- **FR-005**: Every family MUST be presented in natural-language form (a realistic scenario such as incident triage or a hiring decision), so that results speak to agent systems rather than raw bit puzzles. The mapping to the underlying formal task MUST be documented.
- **FR-006**: For every family the information-theoretic quantities MUST be computable exactly: the answer entropy, what each agent holds, the tool capacity, and the budget after T calls.
- **FR-007**: Every family MUST pass a calibration gate before the main run:
  - baseline accuracy is within 5 percentage points of chance;
  - the family's own intervention, fully applied, reaches at least 90% accuracy;
  - leaked instances (FR-001 edge case) are removed.

**Interventions and controls**

- **FR-008**: The suite MUST define a single **resource currency**, with every arm's added cost measured in it. The default is total tokens processed across all agents (inputs, outputs, tool results and messages), with each tool call charged a fixed token-equivalent cost.
- **FR-009**: Every intervention arm MUST add the same increment Δ of resource over baseline. At least three Δ levels MUST be used (small, medium, large), so that gains are curves rather than single points.
- **FR-010**: The suite MUST implement these arms:
  - **E** (relevant evidence or tool access);
  - **K** (extra computation: longer reasoning and/or more samples, with evidence fixed);
  - **M** (extra message budget, with private evidence fixed);
  - **A1** (more agents, with total evidence, compute and messages fixed);
  - **A2** (more agents plus Δ compute, compared against one agent plus Δ compute);
  - **P1** (Δ of irrelevant evidence);
  - **P2** (Δ of non-functional tokens).
- **FR-011**: The message budget MUST be measured on everything that crosses between agents. In the quantitative sweeps it MUST be enforced in bits, by restricting messages to a fixed alphabet and length.

**Measurement**

- **FR-012**: For every run the suite MUST record:
  - the final answer and correctness;
  - the team's probability for each possible answer, where elicitable, or else a sample-frequency estimate;
  - the resources used per agent;
  - every tool call and message;
  - refusals.
- **FR-013**: The suite MUST estimate usable information as answer entropy minus held-out log-loss of calibrated predictions, and report it alongside accuracy and the exact available information and budget.
- **FR-014**: The suite MUST run on at least three language-model families from different developers, so that any result can be distinguished from a quirk of one model.

**Analysis and integrity**

- **FR-015**: All predictions, equivalence margins, sample sizes, exclusion rules and analysis plans MUST be pre-registered and time-stamped before the main data collection.
- **FR-016**: The analysis MUST use paired comparisons (the same instances across arms), confidence intervals, and equivalence tests for every "no effect" prediction, with correction for multiple comparisons across the prediction set.
- **FR-017**: Sample sizes MUST come from a pre-registered power analysis that can confirm a ±5-percentage-point equivalence margin with at least 80% power. The planning figure is about 850 paired instances per family per arm (corrected from ~600 during implementation; see research R10).
- **FR-018**: The blind diagnostic in Story 4 MUST be performed by an analyst who has no access to family labels. Naturalistic-task labels MUST be fixed by at least two independent annotators before any results are produced, and inter-annotator agreement MUST be reported.
- **FR-019**: All task generators, instances, prompts, run records and analysis outputs MUST be released, so that a third party can reproduce every table and figure.
- **FR-020**: Every outcome MUST be reported against the pre-registered prediction as **supported**, **contradicted**, or **inconclusive** (confidence interval too wide), including predictions that fail.

### Pre-registered Predictions and Falsification Conditions

| ID | Prediction (paper source) | Would count as contradicting the theory |
|---|---|---|
| H1 | On information-bound tasks, E helps; K, M, A1, P1 and P2 do not (Thm 1, Cor 1) | K gives a significant gain on leakage-free information-bound instances |
| H2 | On reasoning-bound tasks, K helps; E (restating held evidence), M and A1 do not (Thm 2, Cor 3) | Restated evidence gives a gain as large as K at matched Δ, and the gain is not explained by easier presentation (checked with P1) |
| H3 | On communication-bound tasks, M helps; K and same-kind private E do not (Thm 5) | K alone lifts inner-product-like accuracy significantly above chance at fixed message budget |
| H4 | A1 gives no gain on any family (Cor 1, Thm 3) | A significant A1 gain on any family |
| H5 | The A2 gain equals the gain from the same Δ compute given to one agent (Thm 3) | Multi-agent A2 significantly beats single-agent + Δ compute, beyond what better use of tool calls explains |
| H6 | Information-bound error never falls below the Fano floor (Cor 2) | Error significantly below the floor (indicates leakage, a bug, or a flaw in the theory; audited before reporting) |
| H7 | Inner-product-like splits stay near chance until about n/2 bits; parity-like splits jump at 1 bit (Thm 5, §IX.5) | Inner-product-like accuracy rises steadily from the first bits |
| H8 | Mixed tasks show a positive interaction between the two matching interventions | No interaction: each single intervention gives its full share of the combined gain |
| H9 | The blind diagnostic names the binding ceiling well above chance on held-out constructed tasks | Accuracy not significantly above 1 in 3 |

### Key Entities

- **Task family**: a generator of instances with a known binding ceiling (information, reasoning, communication, or a mixed pair), a natural-language scenario, its formal mapping, and its difficulty settings.
- **Task instance**: one generated problem, with its hidden answer, each agent's private evidence, the tool definition, and its exact information-theoretic quantities (entropy, held information, budget).
- **Arm**: a condition applied to an instance (baseline, E, K, M, A1, A2, P1, P2, or a pair for mixed tasks) at an increment Δ.
- **Resource ledger**: per-run accounting of resources used per agent, per tool call and per message, in the common currency.
- **Run record**: one execution of one instance under one arm and one model family, holding the answer, probabilities, correctness, ledger and transcript.
- **Prediction**: a pre-registered hypothesis (H1–H9) with its test, margin, and verdict.
- **Diagnostic call**: a blind analyst's classification of a task (one ceiling, or "no single ceiling"), with the evidence used.

## Success Criteria *(mandatory)*

These criteria judge whether the *experiment* succeeded, meaning it produced decisive, trustworthy evidence. They do not require the theory to be confirmed. A clean refutation also counts as success.

### Measurable Outcomes

- **SC-001**: Every pre-registered prediction H1–H9 receives a verdict of supported or contradicted. At most 2 of the 9 end as inconclusive.
- **SC-002**: All three single-ceiling families pass the calibration gate (FR-007) on every model family before the main run, so that each family's binding ceiling is established, not assumed.
- **SC-003**: The interval for each "no effect" comparison is narrow enough to confirm or reject the ±5-percentage-point equivalence margin.
- **SC-004**: Every arm's added cost is within ±5% of the target Δ, as verified from the resource ledgers, so that no comparison can be explained by unequal spending.
- **SC-005**: The headline results (the gain table for Story 1 and the communication-signature curves for Story 2) hold in at least two of the three model families.
- **SC-006**: An independent party using only the released materials reproduces the headline gain table to within 2 percentage points per cell.
- **SC-007**: The pre-registration is time-stamped before any main-run data exists. Every deviation from it is listed in the final report.
- **SC-008**: The final report states plainly, for each of the paper's untested §IX predictions it covers (the diagnostic, the communication signature, the error floor), whether the prediction is now supported, contradicted, or still open.

## Assumptions

- **Resource currency**: total tokens processed is the default (FR-008). The paper states that its compute comparison holds in any fixed cost model (RAM steps, FLOPs, tokens), so the choice should not change the direction of the results. A sensitivity analysis with a second currency (monetary cost) is a reasonable addition.
- **Agents are language-model instances**, not people. Human or mixed teams are out of scope, although the theory claims to cover them.
- **Constructed tasks carry the main test.** Naturalistic tasks (Story 4) test transfer, not the theory itself, because their binding ceiling cannot be known exactly.
- **Thresholds are planning values.** The ±5-point margin, 90% calibration ceiling, 5% refusal limit and about 850 instances per cell are reasonable defaults for a first study. The pre-registration may revise them after a pilot, but not after the main run.
- **Scale**: the suite targets 3 single-ceiling families + 2 mixed families, 7 arms, 3 Δ levels, 3+ model families and about 850 instances per cell. Total cost is dominated by the K and A2 arms. A pilot at about 10% scale precedes the main run.
- **Out of scope** for this suite, though natural follow-ups:
  - the model-diversity vs prompt-diversity split (§IX.2);
  - debate vs voting at matched compute (§IX.3);
  - RL post-training effects (§IX.4);
  - channel-seeking and reward hacking under sandboxing (§IX.7);
  - changing-world (drift) settings.
- **Dependencies**:
  - access to at least three model families with controllable reasoning length and sampling;
  - a deterministic tool simulator for the hidden-answer families;
  - independent annotators and a blind analyst for Story 4.
