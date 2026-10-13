# From any-agent to LLMs: owning the generality, then modeling the model

The results in `formal_proofs.md` were criticized, correctly, for carrying an
"LLM" label on math that never used anything specific to language models. This
document fixes that in two layers. **Part A** states the single assumption the
general theory actually needs and declares its scope: any communicating agent.
**Part B** adds the pieces that are genuinely LLM-specific — a miscalibrated
update operator and a lossy/persuasive language channel — and proves how much
each one costs, closing the error terms the earlier bound left sealed.

================================================================================
# Part A — The general theory (any agent)

**Assumption A (the only one).** An agent is a *stochastic policy*: a conditional
distribution `π(a | h)` of action given history. Nothing else is assumed — no
rationality, no calibration, no architecture.

Under Assumption A, every result in `formal_proofs.md` holds verbatim:

- **Conservation.** A closed collective's transcript is a function of initial
  data and truth-independent noise, so `I(Θ; transcript) ≤ I(Θ; D₀)` by the
  data-processing inequality.
- **Synergy.** `S ↦ I(Θ; S)` can be supermodular; pooling realizes joint
  information no member holds (parity: 0 + 0 → 1 bit).
- **Growth.** Only a world channel whose output depends on `Θ` raises total
  information.
- **Tools / routing / consensus-is-not-knowledge** — all as proved.

These are the **agent-agnostic floor**. They hold for sensors, people, Kalman
filters, or LLMs identically, because the data-processing and Jensen
inequalities do not inspect what produced a message. We now *stop* calling this
a theory of LLMs. It is a theory of information flow in any coupled policy
collective. That is the honest scope, and it is genuinely useful as a set of
hard constraints.

What it does **not** capture — and what makes LLMs LLMs — is that a real agent
(i) does not update like a Bayesian and (ii) communicates through natural
language that is lossy and persuadable. Part B models exactly those two things.

================================================================================
# Part B — The LLM specialization (modeling the model)

We keep Assumption A and add two named, measurable deviation operators. Each
comes with a theorem quantifying its cost, and each cost instantiates one of the
abstract error terms `ε_t` (estimation) and `δ_t` (decision) from the regret
bound `R_T ≤ 2 Σ ε_t + Σ δ_t`.

Performance is measured by **log-loss of the truth**, `ℓ(b) = −log b(θ*)`, whose
expectation under a reference law `q` decomposes as
`E_{θ~q}[−log b(θ)] = H(q) + KL(q ‖ b)` (cross-entropy identity). This is the
right yardstick for LLMs because it is what calibration actually grades.

--------------------------------------------------------------------------------
## B1. The miscalibrated update operator (the "bad Bayesian" cost)

**Model.** Where an ideal agent would report the Bayesian posterior
`b* = p(Θ | evidence)`, an LLM reports `b̂ = T(b, O)`, a possibly non-Bayesian
map. Named special cases, all observed in LLMs:

    conservatism / under-reaction:   b̂ = (1−γ) b_prior + γ b*      (0<γ<1)
    overconfidence / sharpening:      b̂ ∝ (b*)^{1/τ}               (τ<1)
    board anchoring / imitation:      b̂ = (1−α) b_own + α b_board

**Theorem B1 (miscalibration gap, exact).** Grading `b̂` against the calibrated
target `b*`, the excess expected log-loss is exactly the divergence from the
Bayesian posterior to the reported belief:

        E_{Θ~b*}[−log b̂(Θ)] − E_{Θ~b*}[−log b*(Θ)] = KL(b* ‖ b̂) ≥ 0.

*Proof.* Cross-entropy identity: `E_{Θ~b*}[−log b̂(Θ)] = H(b*) + KL(b* ‖ b̂)` and
`E_{Θ~b*}[−log b*(Θ)] = H(b*)`. Subtract; `≥ 0` by Gibbs. ∎

**Consequence.** The price of being a bad Bayesian is a single KL number,
computable from the agent's own outputs without ground truth beyond `b*`. It is
precisely the estimation error term: `ε_t ⊇ KL(b*_t ‖ b̂_t)`. *Verified*
(`verify_llm.py`, B1): conservatism with `γ=½` on `b*=(0.9,0.1)` yields
`b̂=(0.7,0.3)` and gap `= 0.1678` bits, matching KL exactly.

--------------------------------------------------------------------------------
## B2. The natural-language channel (the compression cost)

**Model.** Agent `i`'s evidence `D` reaches agent `j` only through a message
`M = C(D, ξ)` — a finite token string, hence a channel of bounded capacity.

**Theorem B2 (capacity floor).** For any encoder `C`,
`I(Θ ; M) ≤ I(Θ ; D)`, and the receiver's achievable information is capped by
`I(Θ ; M)`. The unavoidable loss `I(Θ ; D) − I(Θ ; M) ≥ 0` lower-bounds receiver
suboptimality; it is strictly positive whenever the message cannot carry the
decisive distinction.
*Proof.* DPI (`M` is a function of `D` and `Θ`-independent `ξ`). ∎

**Consequence.** `ε_t ⊇ I(Θ;D_t) − I(Θ;M_t)`. The right design target is not
message volume but preservation of the task-relevant distinction. *Verified*
(B2): a 1-bit message about a 2-bit truth loses exactly 1 bit — `I(Θ;D)=2`,
`I(Θ;M)=1`.

--------------------------------------------------------------------------------
## B3. The persuasion channel (the LLM-critical failure)

This is the behavior the general theory cannot even express: fluent text moving
a belief **without carrying any information about the truth**.

**Model.** A message `M` with `I(Θ ; M) = 0` (independent of the truth) is fed to
a receiver whose update anchors toward the message's assertion:
`b̂ = (1−α) b + α·e_claim`, where `e_claim` is the point mass on the asserted
value.

**Theorem B3 (persuasion moves belief at zero information).** There exist
`M ⟂ Θ` and `α > 0` such that the receiver's MAP estimate flips from the true
value to a false one and its log-loss of the truth strictly increases, while the
information gained about `Θ` is exactly `0`.
*Proof / witness (verified, B3).* `Θ` a fair bit, truth `0`; receiver prior
`b=(0.6,0.4)` (MAP `=0`, correct). Persuasive constant message "claim `=1`" has
`I(Θ;M)=0`. Anchoring gives `b̂=((1−α)0.6, (1−α)0.4+α)`; the MAP flips to `1`
whenever `α > 1/6`. At `α=0.2`: `b̂=(0.48,0.52)`, MAP `=1` (wrong), log-loss of
truth rises from `0.737` to `1.059` bits. Information about `Θ` gained: `0`. ∎

**Consequence.** Persuasion is a pure `δ_t` term: it degrades the *decision*
with no estimation content. It is the exact mechanism of "consensus is not
knowledge" and "copy a confident wrong agent," now with a threshold: harm
occurs once the anchoring weight `α` exceeds the receiver's prior margin for
the truth.

--------------------------------------------------------------------------------
## B4. Putting it together: the sealed box, opened

The earlier abstract bound was `R_T ≤ 2 Σ_t ε_t + Σ_t δ_t` with `ε,δ` undefined.
Part B gives them LLM content:

        ε_t  =  [ I(Θ;D_t) − I(Θ;M_t) ]        (B2, channel loss)
              +  KL( b*_t ‖ b̂_t )               (B1, miscalibration)
        δ_t  =  persuasion / anchoring-induced suboptimal action   (B3)

and the per-round **collective-intelligence condition** (from the Lyapunov drift
in the information-geometry notes, not yet included in this directory) becomes concrete and checkable:

        Bayesian information gain from the world      (Theorem 3, > 0)
            >
        channel loss + miscalibration KL + persuasion-induced excess loss
                                 (B2            + B1              + B3)

When the left side wins each round, the collective's belief trajectory contracts
to the true vertex (collective intelligence). When persuasion and miscalibration
dominate, it contracts to a confident wrong vertex (groupthink). This is exactly
the regime boundary the earlier geometry located, now expressed in quantities
you can measure from an LLM run: estimated-vs-Bayes KL, message capacity about
the task variable, and anchoring weight.

================================================================================
# What is now LLM-specific, and what remains assumed

**Genuinely about LLMs now:**
- B1 models the actual (non-Bayesian) update and prices it as a KL you can
  estimate from model outputs.
- B2 models the natural-language bottleneck as a capacity constraint.
- B3 models persuadability — the thing the general theory structurally cannot
  represent — and gives the harm a threshold in the anchoring weight.

**Still assumed, not derived (honest limits):**
- The *values* of `γ, τ, α`, the capacity of `C`, and the calibrated target `b*`
  must be **measured** per model and task; Part B says how they enter, not what
  they are for GPT-class models.
- B1's grading uses `b*` as reference, which presumes a well-defined calibrated
  posterior; for open-ended generation that target is itself nontrivial.
- The anchoring/conservatism forms are tractable caricatures of real update
  behavior, chosen because they are measurable, not because they are exact.

So the division of labor is clean: **Part A is a theory of any agent and is
proved. Part B is a theory of LLMs and is proved *conditional on* three measured
parameters.** The LLM content lives entirely in those parameters, which is
exactly where it should live — and where an experiment can now go measure it.
