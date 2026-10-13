# From any-agent to LLMs: owning the generality, then modeling the model

The results in [`formal_proofs.md`](formal_proofs.md) were criticized, correctly, for carrying an
"LLM" label on math that never used anything specific to language models. This
document fixes that in two layers. **Part A** states the single assumption the
general theory actually needs and declares its scope: any communicating agent.
**Part B** adds the pieces that are genuinely LLM-specific — a miscalibrated
update operator and a lossy/persuasive language channel — and proves how much
each one costs, closing the error terms the earlier bound left sealed.

================================================================================
<a id="part-a"></a>

# Part A — The general theory (any agent)

**Assumption A (the only one).** An agent is a *stochastic policy*: a conditional
distribution `π(a | h)` of action given history. Nothing else is assumed — no
rationality, no calibration, no architecture.

Under Assumption A, every result in [`formal_proofs.md`](formal_proofs.md) holds verbatim:

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
<a id="part-b"></a>

# Part B — The LLM specialization (modeling the model)

We keep Assumption A and add two named deviation operators: a non-Bayesian
belief update, and a lossy, persuadable message channel. B0 proves the regret
bound that gives these deviations a price; B1–B3 price each one; B4 assembles
them into a single per-round bound.

Throughout, `U` is the range of the utility (`u(a,θ) ∈ [0, U]`),
`TV(p,q) = ½ Σ_x |p(x) − q(x)|` is total variation, and `c := ln 2 / 2`.
Logs are base 2.

--------------------------------------------------------------------------------
<a id="b0"></a>

## B0. The plug-in regret bound

An agent should act on a reference belief `b*` but instead reports `b̂` and
picks an action `â` that is `δ`-optimal for `b̂`:
`Σ_θ b̂(θ)u(â,θ) ≥ max_a Σ_θ b̂(θ)u(a,θ) − δ`.
Its regret against `b*` is
`r := max_a Σ_θ b*(θ)u(a,θ) − Σ_θ b*(θ)u(â,θ)`.

**Lemma B0.** `r ≤ 2U · TV(b*, b̂) + δ ≤ 2U · sqrt( c · KL(b* ‖ b̂) ) + δ`.

*Proof.* For any fixed action `a`,
`|Σ_θ (b*(θ) − b̂(θ)) u(a,θ)| = |Σ_θ (b*(θ) − b̂(θ)) (u(a,θ) − U/2)|
≤ (U/2) Σ_θ |b*(θ) − b̂(θ)| = U · TV(b*, b̂)` (the shift is allowed because
`Σ_θ (b* − b̂) = 0`). Let `a*` be optimal for `b*`. Then
`r = [E_{b*}u(a*) − E_{b̂}u(a*)] + [E_{b̂}u(a*) − E_{b̂}u(â)] + [E_{b̂}u(â) − E_{b*}u(â)]`,
and the three brackets are at most `U·TV`, `δ`, and `U·TV`. The second
inequality is Pinsker's, `TV(p,q) ≤ sqrt((ln 2 / 2) · KL(p‖q))` with KL in bits. ∎

Summing over rounds gives the regret bound

        R_T  ≤  2 Σ_t ε_t + Σ_t δ_t,      ε_t := U · TV(b*_t, b̂_t).

So the **estimation error** `ε_t` is a belief distance and the **decision error**
`δ_t` is suboptimality given the agent's own belief. B1–B3 compute `ε_t` for the
LLM-specific deviations.

--------------------------------------------------------------------------------
<a id="b1"></a>

## B1. The miscalibrated update operator (the "bad Bayesian" cost)

**Model.** Where an ideal agent would report the Bayesian posterior
`b* = p(Θ | evidence)`, an LLM reports `b̂ = T(b, O)`, a possibly non-Bayesian
map. Three tractable special cases, each a commonly hypothesized failure mode
(we cite no measurements here):

    conservatism / under-reaction:   b̂ = (1−γ) b_prior + γ b*      (0<γ<1)
    overconfidence / sharpening:      b̂ ∝ (b*)^{1/τ}               (τ<1)
    board anchoring / imitation:      b̂ = (1−α) b_own + α b_board

**Theorem B1 (miscalibration gap, exact).** Grading `b̂` against the calibrated
target `b*`, the excess expected log-loss is exactly the divergence from the
Bayesian posterior to the reported belief:

        E_{Θ~b*}[−log b̂(Θ)] − E_{Θ~b*}[−log b*(Θ)] = KL(b* ‖ b̂) ≥ 0.

*Proof.* Cross-entropy identity: `E_{Θ~b*}[−log b̂(Θ)] = H(b*) + KL(b* ‖ b̂)` and
`E_{Θ~b*}[−log b*(Θ)] = H(b*)`. Subtract; `≥ 0` by Gibbs. ∎

**Consequence.** For decisions, by Lemma B0, miscalibration costs at most
`2U·sqrt(c·KL(b* ‖ b̂))` per round.

**Measuring it.** `KL(b* ‖ b̂)` needs the reference posterior `b*`, which is
available only when the data-generating process is known (synthetic tasks,
controlled experiments) or approximated by a trusted reference model. Without
`b*`, held-out ground truth still identifies the *difference* in expected
log-loss between two predictors, but not the gap to Bayes itself.

*Verified* ([`verify_llm.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_llm.py), B1): conservatism with `γ=½` on `b*=(0.9,0.1)` and a
uniform prior yields `b̂=(0.7,0.3)` and gap `= 0.1678` bits, matching KL exactly.

--------------------------------------------------------------------------------
<a id="b2"></a>

## B2. The natural-language channel (the compression cost)

**Model.** Agent `i`'s evidence `D` reaches agent `j` only through a message
`M = C(D, ξ)` with `ξ ⟂ (Θ, D)` — a finite token string, hence a channel of
bounded capacity. Write `b^D = p(Θ | D)` and `b^M = p(Θ | M)` for the Bayesian
posteriors of a receiver who sees the data and one who sees only the message.

**Theorem B2 (channel loss is an expected KL).**

        E[ KL( b^D ‖ b^M ) ]  =  I(Θ ; D) − I(Θ ; M)  ≥  0.

*Proof.* Since `M` is a function of `D` and `ξ ⟂ (Θ,D)`, `Θ — D — M`, so
`p(θ | D, M) = p(θ | D)` and `I(Θ ; M | D) = 0`. Then
`E[KL(p(Θ|D,M) ‖ p(Θ|M))] = I(Θ ; D | M)` (Lemma 1.2), and by the chain rule two
ways, `I(Θ;D) = I(Θ;D,M) = I(Θ;M) + I(Θ;D|M)`. Nonnegativity is DPI. ∎

**Consequence.** By Lemma B0 and Jensen (`sqrt` is concave), the extra regret of
acting on the message instead of the data is at most
`2U · sqrt( c · (I(Θ;D) − I(Θ;M)) )`. The design target is not message volume
but preserving the decision-relevant distinction. *Verified* (B2): a 1-bit
message about a 2-bit truth gives `I(Θ;D)=2`, `I(Θ;M)=1`, and
`E KL(b^D ‖ b^M) = 1` bit.

--------------------------------------------------------------------------------
<a id="b3"></a>

## B3. The persuasion channel (the LLM-critical failure)

This is the behavior the general theory cannot express: text moving a belief
**without carrying any information about the truth**.

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
iff `(1−α)·0.2 < α`, i.e. `α > 1/6`. At `α=0.2`: `b̂=(0.48,0.52)`, MAP `=1`
(wrong), log-loss of truth rises from `0.737` to `1.059` bits. Information about
`Θ` gained: `0`. ∎

**Where it sits in the bound.** Since `I(Θ;M) = 0`, the Bayesian posterior given
the message is the prior: `b^M = b`. Persuasion therefore adds nothing to the
channel-loss term of B2 and everything to the miscalibration term of B1:
`KL(b^M ‖ b̂) = KL(b ‖ (1−α)b + α e_claim) > 0`. It is the case of B1 in which the
non-Bayesian update is *triggered by a message that carries no information*.
This is the formal content of "consensus is not knowledge" and "copying a
confident wrong agent," with a threshold: the decision flips once `α` exceeds
the receiver's prior margin for the truth.

--------------------------------------------------------------------------------
<a id="b4"></a>

## B4. Putting it together

Consider one round in which receiver `j` gets message `M_t` built from the
collective's data `D_t`, reports `b̂_t`, and acts `δ_t`-optimally on `b̂_t`.
Compare with an oracle that knows the realized `Θ`.

**Theorem B4 (per-round decomposition).** With `b^D_t = p(Θ|D_t)` and
`b^M_t = p(Θ|M_t)`,

        E[ r_t^oracle ]  ≤  2U · (  E[ 1 − b^D_t(Θ) ]                         (information the collective lacks)
                                  + sqrt( c · (I(Θ;D_t) − I(Θ;M_t)) )       (B2: channel loss)
                                  + E[ sqrt( c · KL(b^M_t ‖ b̂_t) ) ] )      (B1 + B3: miscalibration, persuasion)
                         + δ_t                                                (decision error)

*Proof.* Apply Lemma B0 with reference `b* = e_Θ` (the oracle's belief), so
`r_t^oracle ≤ 2U·TV(e_Θ, b̂_t) + δ_t`. By the triangle inequality for TV,
`TV(e_Θ, b̂_t) ≤ TV(e_Θ, b^D_t) + TV(b^D_t, b^M_t) + TV(b^M_t, b̂_t)`. The first term
equals `1 − b^D_t(Θ)`. Take expectations; bound the second by Pinsker, Jensen and
Theorem B2, and the third by Pinsker. ∎

**Reading.** Each term is a separate failure, and each has its own fix:
- `E[1 − b^D_t(Θ)]` is the probability that a draw from the collective's best
  posterior is wrong. It is at least the MAP error, so by [Theorem 7](formal_proofs.md#theorem-7) (Fano) it
  stays bounded away from zero until the tool budget `Σ_t C_t` approaches
  `H(Θ) − I(Θ;D₀)`. Only the world channel lowers it.
- The channel-loss term falls only if messages preserve the decision-relevant
  distinction.
- The miscalibration term falls only if agents update closer to Bayes and resist
  zero-information persuasion.
- `δ_t` falls only with better action selection given the agent's own belief.

**Collective-intelligence condition (proved form).** If the tool budget grows so
that `E[1 − b^D_t(Θ)] → 0`, and the channel-loss, miscalibration and decision
terms each tend to `0`, then the per-round oracle regret tends to `0`. If any
term stays bounded away from zero, Theorem B4 no longer forces the regret to
vanish. The earlier informal condition "world information gain > channel loss +
miscalibration + persuasion" is a heuristic summary of this and is not a theorem:
its terms are in different units. The drift version of the evidence term is
proved in [`proof.md`, Lemma G4](proof.md#g4).

*Verified* ([`verify_llm.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_llm.py), B0 and B4): Lemma B0 on 2000 random belief pairs and
utilities, and every term of Theorem B4 on a 2-bit example with a 1-bit message
and an anchoring receiver.

================================================================================
<a id="limits"></a>

# What is now LLM-specific, and what remains assumed

**Genuinely about LLMs now:**
- B1 models the actual (non-Bayesian) update and prices it as a KL.
- B2 models the natural-language bottleneck as an exact information loss.
- B3 models persuadability — the thing the general theory structurally cannot
  represent — and gives the harm a threshold in the anchoring weight.
- B0 and B4 turn all three into a proved regret bound.

**Still assumed, not derived (honest limits):**
- The *values* of `γ, τ, α`, the capacity of `C`, and the calibrated target `b*`
  must be **measured** per model and task; Part B says how they enter, not what
  they are for any particular model.
- B1's grading uses `b*` as reference, which presumes a well-defined calibrated
  posterior; for open-ended generation that target is itself nontrivial.
- The anchoring/conservatism forms are tractable caricatures of real update
  behavior, chosen because they are measurable, not because they are exact.
- Theorem B4 is an upper bound. It says which terms must vanish for regret to
  vanish; it does not say any term is large for a given model.

So the division of labor is: **Part A is a theory of any agent and is proved.
Part B is a theory of LLMs and is proved *conditional on* measured parameters.**
The LLM content lives entirely in those parameters, which is exactly where an
experiment can now go measure it.
