# Collective information acquisition as the geometry of one convex functional

`formal_proofs.md` proves conservation, synergy and growth. This document shows
they are three readings of a single object: the negative-entropy potential on
the belief simplex, its Bregman divergence, and the two flows it induces
(Bayesian evidence and consensus averaging).

Throughout, `Θ` is the task-relevant truth on a finite alphabet `𝒳 = {θ_1,…,θ_K}`
with prior `p(θ)`. `D₀` is everything jointly available to the whole system at
`t=0`: every agent's private context, the initial board `G₀`, and the frozen
weights as a fixed prior. `ξ` denotes exogenous randomness (sampling noise,
scheduling, board rules), independent of `(Θ, D₀)`. Logs are base 2 and
information is in bits unless written `ln`.

Statements marked **Lemma** or **Theorem** are proved. Statements marked
**Model** are modeling assumptions, not theorems.

--------------------------------------------------------------------------------
## Theorems 1–3 in brief

Proofs are in `formal_proofs.md`; here we only fix notation.

- **Conservation (Theorem 1).** If after `t=0` the agents only communicate and
  compute, the whole trajectory `W` is a function `ψ(D₀, ξ)`, so
  `I(Θ ; W) ≤ I(Θ ; D₀, ξ) = I(Θ ; D₀) + I(Θ ; ξ | D₀) = I(Θ ; D₀)`.
- **Synergy (Theorem 2).** `I(Θ ; S ∪ {Y}) − I(Θ ; S) = I(Θ ; Y | S) ≥ 0`, but
  `S ↦ I(Θ;S)` need not be submodular. With `Θ = B₁ ⊕ B₂`:
  `I(Θ;B₁) = I(Θ;B₂) = 0`, `I(Θ;B₁,B₂) = 1`. The bit was present jointly in
  `D₀`; only its accessibility changed.
- **Growth (Theorem 3).** A world channel `O ∼ p(· | θ, a)` adds
  `I(Θ ; O | H, a) = E_{θ∼p(·|H)}[ KL( p(O|θ,a) ‖ p(O|H,a) ) ] ≥ 0`, strictly
  positive iff the likelihood varies with `θ` on a set of positive posterior
  probability. Theorem 7 bounds the total by the tools' channel capacity.

================================================================================
# The functional

Let `Δ = { b ∈ ℝ^K_{≥0} : Σ_k b_k = 1 }` be the belief simplex. Define the
**negative-entropy potential**

        F(b) = Σ_k b_k log b_k = −H(b),     strictly convex on Δ.

In bits, `∇F(b)_k = log b_k + 1/ln 2`, and on the tangent space
`{δ : Σ_k δ_k = 0}` the Hessian is `∇²F(b) = (1/ln 2) · diag(1/b_k) ≻ 0`.
(In nats the familiar forms `ln b_k + 1` and `diag(1/b_k)` hold; the factor
`1/ln 2` is only the change of units.)

--------------------------------------------------------------------------------
## G1. KL is the Bregman divergence of F; mutual information is its mean

**Lemma G1.** The Bregman divergence of `F`,
`D_F(p, q) = F(p) − F(q) − ⟨∇F(q), p − q⟩`, equals `KL(p ‖ q)` (in bits).
*Proof.* `⟨∇F(q), p − q⟩ = Σ_k (log q_k + 1/ln 2)(p_k − q_k)
= Σ_k p_k log q_k − Σ_k q_k log q_k`, since `Σ_k (p_k − q_k) = 0` removes the
constant. So `D_F(p,q) = Σ p_k log p_k − Σ q_k log q_k − Σ p_k log q_k +
Σ q_k log q_k = Σ_k p_k log(p_k/q_k)`. ∎

**Lemma G2.** For any signal `M`, with prior `b` and posterior `b^M = p(·|M)`,

        I(Θ ; M) = E_M[ KL(b^M ‖ b) ] = E_M[ D_F(b^M, b) ] = H(b) − E_M[ H(b^M) ].

*Proof.* `I(Θ;M) = Σ_{θ,m} p(θ,m) log[p(θ|m)/p(θ)] = Σ_m p(m) KL(b^m ‖ b)`;
the middle equality is Lemma G1; the last is `I(Θ;M) = H(Θ) − H(Θ|M)`. ∎

So every information quantity in Theorems 1–3 is an **expected Bregman
divergence of `F = −H`**: information gain is expected statistical distance
travelled from the prior. In this language, synergy is the failure of
submodularity of `S ↦ E[D_F(b^S, b)]`.

--------------------------------------------------------------------------------
## G2. Value of information and information gain are the same Jensen gap

Let `V(b) = max_a Σ_θ b(θ) u(a,θ)`. It is convex (a maximum of linear functions),
and Bayesian posteriors are a martingale, `E_M[b^M] = b`, so Jensen gives
`VoI(M) = E_M[V(b^M)] − V(b) ≥ 0` (`formal_proofs.md`, Lemmas 6.1–6.2), with
equality iff an action optimal at the prior stays optimal at almost every
posterior.

**Lemma G3 (information gain is the value of information under log loss).**
Take actions `q ∈ Δ` (report a distribution) and utility `u(q,θ) = log q(θ)`.
Then `V(b) = max_q Σ_θ b(θ) log q(θ) = −H(b)`, and `VoI(M) = I(Θ ; M)`.
*Proof.* `Σ_θ b(θ) log q(θ) = −H(b) − KL(b ‖ q) ≤ −H(b)`, with equality at
`q = b` (Gibbs), so `V = −H = F`. Then
`VoI(M) = E_M[−H(b^M)] + H(b) = I(Θ;M)` by Lemma G2. ∎

So "uncertainty goes down" (`I ≥ 0`) and "payoff goes up" (`VoI ≥ 0`) are one
theorem: Jensen's inequality for a convex value function along a martingale.
Information gain is the special case where the decision is "report a belief" and
the payoff is log loss.

--------------------------------------------------------------------------------
## G3. The Fisher–Rao metric: small information gain is squared length

`F`'s Hessian is the Fisher information metric of the categorical family. For a
tangent perturbation `δ` (`Σ_k δ_k = 0`), second-order Taylor expansion of
Lemma G1 gives, in bits,

        KL(b + δ ‖ b) = (1 / (2 ln 2)) · Σ_k δ_k² / b_k  +  O(‖δ‖³).

(In nats the constant is `½`.) A small information gain is therefore a squared
Fisher–Rao length, and a belief trajectory `b₀ → b₁ → … → b_T` is a path on this
Riemannian manifold.

`(Δ, F)` is **dually flat** (Amari–Nagaoka) with two affine coordinate systems:
- **mixture (m) coordinates** `b_k` — averaging is linear here;
- **exponential (e) coordinates** `η_k = log b_k` (up to an additive constant) —
  Bayesian updating is a **translation** here:
  `log b⁺_k = log b_k + log p(o | θ_k) − log Z`.

--------------------------------------------------------------------------------
## G4. Two flows on the simplex

A real collective does not apply the exact Bayesian map. **Model:** its belief
dynamics combine two flows.

| Flow | Coordinates in which it is affine | Effect |
|------|-----------------------------------|--------|
| **Evidence** (Bayes, Theorem 3) | e-coordinates (translation by log-likelihood) | moves toward the true vertex in expectation (Lemma G4) |
| **Consensus** `bᵢ ← Σⱼ Wᵢⱼ bⱼ` (linear pooling, row-stochastic `W`) | m-coordinates (convex combination) | contracts beliefs toward each other (Lemma G5) |

Linear pooling is **not** Bayesian: pooling agents who saw overlapping evidence
neither double-counts nor combines their likelihoods; it averages beliefs.

Use the log-loss of the truth as a Lyapunov function. Fix the true value `θ*` and
let

        L(b) = −log b(θ*) = KL(e_{θ*} ‖ b).

**Lemma G4 (evidence drift).** If observations are drawn from the true
likelihood, `o ∼ p(· | θ*, a)`, and the agent updates by Bayes' rule, then

        E_o[ L(b) − L(b⁺) ] = KL( p(· | θ*, a) ‖ p_b(· | a) ) ≥ 0,

where `p_b(o|a) = Σ_θ b(θ) p(o|θ,a)` is the agent's predictive distribution.
The decrease is strictly positive iff that predictive differs from the true
likelihood.
*Proof.* `b⁺(θ*) = b(θ*) p(o|θ*,a) / p_b(o|a)`, so
`L(b) − L(b⁺) = log[ p(o|θ*,a) / p_b(o|a) ]`. Take the expectation under
`o ∼ p(·|θ*,a)`; nonnegativity and the equality case are Gibbs. ∎

So with a correctly specified likelihood, evidence lowers the log-loss of the
truth **in expectation** every round. (Almost-sure convergence `b_t(θ*) → 1`
additionally needs identifiability; this is Doob's consistency theorem.)

**Lemma G5 (consensus).** For agent `i` after one pooling step
`b'_i = Σⱼ Wᵢⱼ bⱼ`:

1. `L(b'_i) ≤ Σⱼ Wᵢⱼ L(bⱼ)` — the pooled log-loss is at most the weighted
   average of the neighbours' log-losses (convexity of `−log`);
2. `H(b'_i) ≥ Σⱼ Wᵢⱼ H(bⱼ)` — the pooled belief is at least as uncertain as the
   neighbours' **average** uncertainty (concavity of `H`); it need not be more
   uncertain than every neighbour;
3. `L(b'_i)` can be larger than `L(b_i)`: pooling hurts an agent whose
   neighbours are further from the truth than it is.

*Proof.* (1) and (2) are Jensen. (3) by example: `b_1 = (0.9, 0.1)`,
`b_2 = (0.1, 0.9)`, `θ* = θ_1`, `W_{1·} = (½, ½)` gives `b'_1 = (0.5, 0.5)` and
`L` rises from `0.152` to `1` bit. ∎

**Regimes (model, not theorem).** One round's change in `L` for an agent is the
sum of an evidence term, which is `≤ 0` in expectation (Lemma G4), and a
consensus term, which can have either sign (Lemma G5(3)). The geometry says the
collective converges to the truth when the evidence flow dominates and the
consensus flow averages toward agents closer to `θ*`; it can lock onto a wrong
vertex when consensus repeatedly pulls toward confident wrong agents. Which
regime occurs depends on `W`, the environment and the policy. No universal
threshold follows from these lemmas.

================================================================================
# Tools (the external channel), placed in the geometry

Split each action into a board post and a tool query, `A_t^i = (M_t^i, Q_t^i)`.
With world state `E_t` and `D_t = (G_t, E_t, X_t^{1:N})`:

        (M_t^i, Q_t^i) ∼ π(· | X_t^i, G_t, O_t^i)
        G_{t+1}        = F_φ(G_t, M_t^{1:N})
        O_{t+1}        ∼ K(· | E_t, Θ, Q_t^{1:N})      # measurement: depends on Θ
        E_{t+1}        ∼ T(· | E_t, Q_t^{1:N})          # intervention: changes the world

**Theorem 4 (tools carry the only new information).** If `Q_t = g(D_t, ξ)` with
`ξ ⟂ (Θ, D_t)`, then `I(Θ ; Q_t | D_t) = 0` and
`I(Θ ; D_{t+1}) − I(Θ ; D_t) = I(Θ ; O_{t+1} | D_t, Q_t)`
(`formal_proofs.md`, Theorem 4). By Lemma G2 this is the only term that adds
expected Bregman distance `E[D_F(b⁺, b)]` about `Θ`; Theorem 7 caps it per round
by the tool's channel capacity.

**Theorem 5 (the board's routing value is a value of information).** With
pooled state, the group can pick the query with the highest information yield
for the actual state of the world; an isolated agent must pick blind. The
difference is the value of information of the pooled signal in the decision
problem "which query to run," with utility equal to information yield
(`formal_proofs.md`, Theorem 5). In the minimal witness — agent 1 knows which
query is informative (`K`), agent 2 owns the tool — the gain is `½` bit, and it
is not information about `V` created by talking but the value of knowing which
experiment to run. In geometric terms, pooling lets the group choose the query
whose expected Bregman step `E[D_F(b⁺, b)]` is largest.

--------------------------------------------------------------------------------
## Beyond the manifold: access control and the disclosure edge

Two parts of the tool loop sit **outside** this information geometry:

- **Access-control state** `S_t = (identity, scope, authorization, attestation,
  audit)`, with `O_{t+1} ∼ K(· | E_t, Q_t, S_t)`, governs *who may use the
  channel*, not the curvature of `F`. Text on the board (claimed intent,
  reputation) is copyable; only hard-to-forge, non-transferable `S_t` should gate
  the channel.
- **The disclosure edge** `O_{t+1} → G_{t+1}` is a capability-amplification
  channel. The mechanism that gives the collective its power — making
  distributed information jointly accessible (Theorem 2) — is the same one that
  turns a scoped private measurement into broadly reusable capability. A safe
  design keeps raw results scoped, publishes only a vetted summary to `G`, and
  logs the action.

Theorems 1–7 describe what the collective can know; `S_t` and the disclosure
edge describe what it is permitted to do with a channel.

================================================================================
# What this unifies

One potential `F = −H` and the martingale property of Bayesian posteriors give:

1. **Conservation:** post-processing cannot raise `E[D_F(b⁺, b)]` above the
   information already in `D₀` (DPI).
2. **Synergy:** `S ↦ E[D_F(b^S, b)]` can fail to be submodular.
3. **Growth:** only a channel that samples from the true `θ` adds Bregman
   distance (Theorems 3, 4), and Theorem 7 caps how much.
4. **Value of information:** VoI is the Jensen gap of a convex `V`; information
   gain is the log-loss case (Lemma G3).
5. **Regret:** decision regret is controlled by belief distance — first-order in
   total variation, hence by `sqrt(KL)` through Pinsker
   (`llm_specialization.md`, Lemma B0), and locally by the Fisher metric (G3).

================================================================================
# What is NOT proven (honest limits)

- **Lemma G4 needs a correctly specified likelihood.** If an agent's model of
  `p(o|θ,a)` is wrong, or it does not update by Bayes' rule, the evidence drift
  can be positive. Real LLMs are not calibrated Bayesians
  (`llm_specialization.md`, B1).
- **The two-flow decomposition is a model.** Real message passing mixes the two
  flows nonlinearly; whether a given system is well described by "Bayes step +
  linear pooling" must be measured.
- **No universal emergence threshold** follows. The sign of the net drift
  depends on `W`, the environment and the policy.
- **Strict growth in Theorem 3** requires the likelihood to depend on `θ`; a tool
  whose output is independent of `θ` adds nothing.

Exact numerical checks of Lemmas G1–G5 and the G3 constant are in
`verify_geometry.py`.
