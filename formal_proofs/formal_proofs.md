# Formal proofs: collective information acquisition

All random variables are discrete with finite support on a common finite
probability space `(Ω, 2^Ω, ℙ)`. Logarithms are base 2; `0 log 0 := 0`.

<a id="definitions"></a>

## 0. Definitions

For random variables with pmfs `p`:

    H(X)        := − Σ_x p(x) log p(x)
    H(X|Y)      := − Σ_{x,y} p(x,y) log p(x|y)
    I(X;Y)      := Σ_{x,y} p(x,y) log [ p(x,y) / (p(x) p(y)) ]
    I(X;Y|Z)    := Σ_{x,y,z} p(x,y,z) log [ p(x,y|z) / (p(x|z) p(y|z)) ]
    KL(p ‖ q)   := Σ_x p(x) log [ p(x)/q(x) ]      (q(x)=0 ⇒ p(x)=0 assumed)

`X — Y — Z` ("Markov chain") means `p(z | y, x) = p(z | y)` whenever `p(x,y)>0`,
equivalently `X ⟂ Z | Y`.

--------------------------------------------------------------------------------
<a id="lemmas"></a>

## 1. Standard lemmas (proved for self-containedness)

**Lemma 1.1 (Gibbs / nonnegativity of KL).** `KL(p ‖ q) ≥ 0`, with equality iff
`p = q`.
*Proof.* `−KL(p‖q) = Σ_x p(x) log(q(x)/p(x))`. The function `log` is strictly
concave, so by Jensen `Σ_x p(x) log(q(x)/p(x)) ≤ log Σ_x p(x)·q(x)/p(x)
= log Σ_{x: p(x)>0} q(x) ≤ log 1 = 0`. Strict concavity gives equality iff
`q(x)/p(x)` is ℙ-a.s. constant and `Σ q = 1`, i.e. `p=q`. ∎

**Lemma 1.2 (MI as KL; nonnegativity).** `I(X;Y) = KL( p_{XY} ‖ p_X⊗p_Y ) ≥ 0`,
and `I(X;Y|Z) = Σ_z p(z) KL( p_{XY|z} ‖ p_{X|z}⊗p_{Y|z} ) ≥ 0`.
*Proof.* Immediate from the definitions and [Lemma 1.1](#lemmas) applied to the joint vs.
product pmf: for each `z`, apply it to `p_{XY|z}` vs. `p_{X|z}⊗p_{Y|z}`, then
average with the nonnegative weights `p(z)`. ∎

**Lemma 1.3 (Chain rule for MI).**
`I(X ; Y, Z) = I(X ; Z) + I(X ; Y | Z)`.
*Proof.* Expand both sides with `p(y,z|x)=p(z|x)p(y|x,z)` and
`p(y,z)=p(z)p(y|z)`:
`I(X;Y,Z) = Σ p(x,y,z) log [p(x,y,z)/(p(x)p(y,z))]`
`= Σ p(x,y,z) log [p(x,z)/(p(x)p(z))] + Σ p(x,y,z) log [p(x,y,z)p(z)/(p(x,z)p(y,z))]`
`= I(X;Z) + I(X;Y|Z)`. ∎

**Lemma 1.4 (Functions carry no extra information).** If `R ⟂ (Θ, A)` then
`I(Θ ; R | A) = 0`; and if `B = f(A, R)` with `R ⟂ (Θ, A)` then `Θ — A — B`,
hence (with [Lemma 1.5](#lemmas)) `I(Θ;B) ≤ I(Θ;A)`.
*Proof.* `I(Θ;R|A) = Σ_a p(a) KL(p_{ΘR|a} ‖ p_{Θ|a}⊗p_{R|a})`. Since
`R ⟂ (Θ,A)`, `p(θ,r|a)=p(θ|a)p(r)` and `p(r|a)=p(r)`, so each conditional KL is
`KL(p_{Θ|a}⊗p_R ‖ p_{Θ|a}⊗p_R)=0`. For the Markov claim: given `(a,r)`, `B`
is determined, so `B ⟂ Θ | (A,R)`; combined with `R ⟂ Θ | A` (just shown) this
gives `B ⟂ Θ | A` after marginalizing `R`. ∎

**Lemma 1.5 (Data-processing inequality).** If `Θ — A — B` then
`I(Θ ; B) ≤ I(Θ ; A)`.
*Proof.* By [Lemma 1.3](#lemmas) two ways:
`I(Θ ; A, B) = I(Θ;A) + I(Θ;B|A) = I(Θ;B) + I(Θ;A|B)`.
The Markov hypothesis gives `I(Θ;B|A)=0`, so `I(Θ;A) = I(Θ;B) + I(Θ;A|B)`.
By [Lemma 1.2](#lemmas) `I(Θ;A|B) ≥ 0`, hence `I(Θ;A) ≥ I(Θ;B)`. ∎

--------------------------------------------------------------------------------
<a id="theorem-1"></a>

## 2. Theorem 1 — Conservation in a closed system

**Hypotheses (closed system).** Let `Θ` be the external truth and `D₀` the
complete initial data of the system. Let `R` be all randomness used after `t=0`
(decoding, scheduling, board rules), with `R ⟂ (Θ, D₀)`. Assume the full future
transcript `Y = (G_{1:T}, A_{1:T}, X_{1:T}^{1:N})` is a deterministic measurable
function `Y = f(D₀, R)` (no new observation of `Θ`'s environment enters).

**Theorem 1.** `I(Θ ; Y) ≤ I(Θ ; D₀)`.
*Proof.* By [Lemma 1.4](#lemmas) with `A=D₀`, `B=Y`, we have `Θ — D₀ — Y`; [Lemma 1.5](#lemmas) gives
`I(Θ;Y) ≤ I(Θ;D₀)`. (Equivalently, chain rule:
`I(Θ;Y) ≤ I(Θ;D₀,R) = I(Θ;D₀) + I(Θ;R|D₀) = I(Θ;D₀) + 0`.) ∎

**Corollary 1.1 (accessibility, not creation).** For any agent `i`,
`I(Θ ; X_t^i, G_t) = I(Θ ; X_t^i) + I(Θ ; G_t | X_t^i)` ([Lemma 1.3](#lemmas)). The board
raises agent `i`'s information by exactly `I(Θ ; G_t | X_t^i) ≥ 0`, while the
system total stays `≤ I(Θ;D₀)` by [Theorem 1](#theorem-1). ∎

--------------------------------------------------------------------------------
<a id="theorem-2"></a>

## 3. Theorem 2 — Synergy (verified finite witnesses)

**Theorem 2 (XOR).** Let `B₁,B₂` be i.i.d. uniform on `{0,1}`, `Θ = B₁⊕B₂`,
`Y₁=B₁`, `Y₂=B₂`. Then `I(Θ;Y₁)=I(Θ;Y₂)=0` and `I(Θ;Y₁,Y₂)=1`, so the net
synergy `σ := I(Θ;Y₁,Y₂) − I(Θ;Y₁) − I(Θ;Y₂) = 1`.
*Proof.* `Θ` is uniform on `{0,1}` and independent of `B₁` alone (for each
`b₁`, `Θ=b₁⊕B₂` is uniform), so `I(Θ;Y₁)=0`; symmetrically `I(Θ;Y₂)=0`. Given
`(B₁,B₂)`, `Θ` is determined, so `H(Θ|Y₁,Y₂)=0` and
`I(Θ;Y₁,Y₂)=H(Θ)=1`. Machine-checked in [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py), Scenario 1. ∎

**Theorem 2′ (additive bits).** `Θ=(B₁,B₂)`, same `Y_i`: `I(Θ;Y₁)=I(Θ;Y₂)=1`,
`I(Θ;Y₁,Y₂)=2`, `σ=0`. (Machine-checked, Scenario 2.) This contrasts with the
XOR case: both are consistent with [Theorem 1](#theorem-1) because the joint information
`I(Θ;Y₁,Y₂)` was already present in `D₀ = (Y₁,Y₂)`; only its accessibility to
individual agents differs. ∎

**Remark (where collective computation lives).** `σ>0` is precisely the failure
of submodularity of the set function `S ↦ I(Θ;S)`. XOR exhibits the extreme:
each singleton carries `0`, the pair carries `1`.

--------------------------------------------------------------------------------
<a id="theorem-3"></a>

## 4. Theorem 3 — Growth requires an external channel

**Hypotheses (open channel).** At round `t` the collective holds history `H_t`,
chooses query `Q_t` measurable w.r.t. `(H_t, ξ)` with `ξ ⟂ Θ`, and the world
returns `O_{t+1} ∼ K(· | Θ, Q_t)` — the kernel conditions on the *true* `Θ`.

**Theorem 3.**
`I(Θ ; O_{t+1} | H_t, Q_t) = E_{Θ ∼ p(·|H_t,Q_t)}[ KL( K(·|Θ,Q_t) ‖ p(O|H_t,Q_t) ) ] ≥ 0`,
with equality iff `K(o|θ,Q_t)` does not depend on `θ` (ℙ-a.s. over the
posterior). Consequently
`I(Θ ; H_t, O_{t+1}) = I(Θ ; H_t) + I(Θ ; O_{t+1} | H_t)` grows by the gain term.
*Proof.* The identity is [Lemma 1.2](#lemmas) written out:
`I(Θ;O|·) = Σ_θ p(θ|·) Σ_o K(o|θ,Q) log[ K(o|θ,Q)/p(o|·) ]
= E_θ KL(K(·|θ,Q) ‖ p(·|·))`. Nonnegativity and the equality case are
[Lemma 1.1](#lemmas). This term is **not** bounded by `I(Θ;D₀)`: the Markov hypothesis of
[Theorem 1](#theorem-1) fails because `O_{t+1}` depends on `Θ` directly through `K`. ∎

--------------------------------------------------------------------------------
<a id="theorem-4"></a>

## 5. Theorem 4 — Tools carry the only information; queries carry none

**Hypotheses.** `D_{t+1} = (D_t, Q_t, O_{t+1})`, with `Q_t = g(D_t, ξ)`,
`ξ ⟂ Θ`, and `O_{t+1}` the tool result.

**Theorem 4.** `I(Θ ; Q_t | D_t) = 0`, and therefore
`I(Θ ; D_{t+1}) − I(Θ ; D_t) = I(Θ ; O_{t+1} | D_t, Q_t)`.
*Proof.* `I(Θ;Q_t|D_t)=0` is [Lemma 1.4](#lemmas) (`Q_t` a function of `D_t` and
`Θ`-independent `ξ`). Then by [Lemma 1.3](#lemmas) applied twice,
`I(Θ;D_t,Q_t,O_{t+1}) = I(Θ;D_t) + I(Θ;Q_t|D_t) + I(Θ;O_{t+1}|D_t,Q_t)`
`= I(Θ;D_t) + 0 + I(Θ;O_{t+1}|D_t,Q_t)`. ∎

--------------------------------------------------------------------------------
<a id="theorem-5"></a>

## 6. Theorem 5 — Routing is value of information

**Lemma 6.1 (posterior-mean martingale).** For any signal `M`,
`E_M[ p(θ | M) ] = p(θ)` for every `θ`.
*Proof.* `Σ_m p(m) p(θ|m) = Σ_m p(θ,m) = p(θ)`. (Machine-checked, Scenario 6.) ∎

**Lemma 6.2 (value of information ≥ 0).** Let `V(b) = max_a Σ_θ b(θ) u(a,θ)` on
the simplex. `V` is convex (a pointwise maximum of linear functionals). With
prior belief `b = p(·)` and posterior `b^M = p(· | M)`,
`E_M[ V(b^M) ] ≥ V(b)`, i.e. `VoI(M) = E_M[V(b^M)] − V(b) ≥ 0`.
*Proof.* Convexity: each `b ↦ Σ_θ b(θ)u(a,θ)` is linear, and `max` over `a` of
convex functions is convex. By [Lemma 6.1](#theorem-5), `E_M[b^M] = b`. Jensen for the convex
`V`: `E_M[V(b^M)] ≥ V(E_M[b^M]) = V(b)`.
*Equality case.* Equality holds iff an action `a₀` optimal at the prior `b` is
still optimal at `b^M` for ℙ-almost every `M`. (If so, `E_M V(b^M) =
E_M Σ_θ b^M(θ)u(a₀,θ) = Σ_θ b(θ)u(a₀,θ) = V(b)`. Conversely, `V(b^M) ≥
Σ_θ b^M(θ)u(a₀,θ)` pointwise and the two sides have the same mean `V(b)` under
equality, so they agree almost surely.) So information has strictly positive
value exactly when it changes the best action with positive probability. ∎

**Theorem 5 (routing is value of information).** Let `V` be the target truth and
`K` a variable some agent holds that affects which query is informative, with
`V ⟂ K`. A query `q` produces `O ∼ p(· | V, K, q)`. Define the **information
yield** of query `q` in state `k` as

        u(q, k) := I(V ; O | Q = q, K = k).

This is a decision problem with action `q`, state `k` and utility `u`. Then:
- an agent that does not know `K` and picks a fixed `q` gets expected yield
  `Σ_k p(k) u(q,k)`, at best `V_u(p_K) := max_q Σ_k p(k) u(q,k)`;
- a collective in which `K` is posted to the board can pick `q = q*(K)` and gets
  `E_K[ max_q u(q, K) ] = E_K[ V_u(e_K) ]`;
- the board's routing value is the gap,
  `E_K[V_u(e_K)] − V_u(p_K) = VoI(K)` for this decision problem, which is `≥ 0`
  by [Lemma 6.2](#theorem-5) and `> 0` iff no single query is optimal for almost every `k`.

(Yields are measured conditional on `K`, i.e. as information about `V` once
`K` is known; when `V ⟂ K` and the query reveals nothing about `K`, as below,
this matches the unconditional quantities.)

*Proof.* The first two items are the definitions of expected utility under a
fixed action and under the best response to `K` (a randomized `q` independent of
`K` cannot beat the best fixed one, since expected yield is linear in the
randomization). The third is [Lemma 6.2](#theorem-5) applied with prior `p_K` and the fully
informative signal `M = K`. ∎

**Witness.** `K` uniform on `{1,2}`, `V` uniform on `{0,1}`, independent; the
tool returns `O=V` iff `q=K`, else `O=⊥`. Then `u(q,k) = 1` if `q=k`, else `0`.
- isolated: `V_u(p_K) = max_q ½ = ½` bit (and a uniformly random `q` also gives
  `I(V ; O | Q) = ½`);
- pooled: `E_K max_q u(q,K) = 1` bit (and `I(V ; O) = 1` with `q=K`);
- routing value = `VoI(K) = ½` bit.
Machine-checked in [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py), Scenario 3, both as mutual informations from
the joint law and as the VoI of `K` in the yield decision problem. ∎

--------------------------------------------------------------------------------
<a id="theorem-6"></a>

## 7. Theorem 6 — Consensus is not knowledge; coupling is not intelligence

**Theorem 6a (coupling).** If `A¹=R` (uniform bit) and `A²=R` (copy), then
`I(A¹;A²)=1`; if `A¹,A²` are independent uniform bits, `I(A¹;A²)=0`. (Scenario
4.) Statistical dependence between agents is produced by the board edge alone
and carries no `Θ`-information.

**Theorem 6b (consensus).** Let `Θ` and the board's first claim `R` be
independent fair bits, and let all agents copy `R`. Then perfect consensus is
reached, yet `I(Θ;R)=0` and `ℙ(R=Θ)=1/2`. (Scenario 5.) High agreement and low
uncertainty can coexist with zero knowledge; a truth-linked criterion
(verification, reproducible experiment, proof check, calibrated held-out
performance) is required to certify knowledge.
*Proof.* Finite computations from the stated joint laws; machine-checked. ∎

--------------------------------------------------------------------------------
<a id="theorem-7"></a>

## 8. Theorem 7 — The capacity budget (agents + tools)

Theorems 1–4 say *where* information comes from. [Theorem 7](#theorem-7) says *how much* can
arrive, for any number of agents, any communication protocol, any internal
computation, and any adaptive query policy.

**Setup.** `Θ` is a fixed truth on a finite alphabet. The collective runs rounds
`t = 0, …, T−1`. `D_t` is the **full record** at the start of round `t`: initial
data `D₀` plus everything generated since (every board post, private state,
query, and tool result). Each agent's state and the board are functions of `D_t`.
In round `t`:

    ξ_t                         fresh internal randomness (sampling, scheduling)
    Q_t   = g_t(D_t, ξ_t)       the round's tool calls (any number, any policy)
    O_{t+1} ∼ K_t(· | Θ, Q_t, D_t)   tool results
    ζ_t                         fresh randomness for communication/computation
    D_{t+1} = (D_t, ξ_t, Q_t, O_{t+1}, ζ_t)

**Assumptions.**
- (A1) *No side channels.* `Θ` reaches the collective only through `D₀` and the
  tool results.
- (A2) *Truth-independent noise.* `ξ_t ⟂ (Θ, D_t)` and
  `ζ_t ⟂ (Θ, D_t, ξ_t, Q_t, O_{t+1})`.
- (A3) *Tools are channels from `Θ`.* Given `(Θ, Q_t, D_t)`, the result
  `O_{t+1}` is independent of `ξ_t` and has law `K_t(· | Θ, Q_t, D_t)`. Tools may
  be adaptive or stateful; the kernel may depend on the query and the history.
- (A4) *Θ is fixed.* The collective's actions do not change `Θ`.

**Per-round capacity.** For each query `q` and history `d`, the map
`θ ↦ K_t(· | θ, q, d)` is a channel with Shannon capacity
`Cap(K_{t,q,d}) := max_{P_Θ} I(Θ ; O)`. Let

        C_t := sup_{q, d} Cap(K_{t,q,d})   ≤   log₂ |𝒪_t|,

where `𝒪_t` is the set of possible results in round `t` (all calls together).

**Theorem 7 (capacity budget).** Let `B_T := I(Θ ; D₀) + Σ_{t<T} C_t`. Then

        I(Θ ; D_T)  ≤  min( H(Θ), B_T ),

and the same bound holds for `I(Θ ; Y)` for every output `Y` the collective can
produce (final answer, board, any agent's state, any vote), since each is a
function of `D_T` and independent noise ([Lemma 1.4](#lemmas)). For any estimator
`Θ̂` built from the collective's record, with `|Θ| ≥ 3`,

        ℙ(Θ̂ ≠ Θ)  ≥  ( H(Θ) − min(H(Θ), B_T) − 1 ) / log₂(|Θ| − 1).

*Proof.*
1. *One round, by the chain rule ([Lemma 1.3](#lemmas)).*
   `I(Θ;D_{t+1}) = I(Θ;D_t) + I(Θ;ξ_t|D_t) + I(Θ;Q_t|D_t,ξ_t)
   + I(Θ;O_{t+1}|D_t,ξ_t,Q_t) + I(Θ;ζ_t|D_t,ξ_t,Q_t,O_{t+1})`.
2. *Three terms vanish.* `I(Θ;ξ_t|D_t) = 0` and `I(Θ;ζ_t|…) = 0` by (A2) and
   [Lemma 1.4](#lemmas). `I(Θ;Q_t|D_t,ξ_t) = 0` because `Q_t` is a function of
   `(D_t, ξ_t)`. Communication and computation therefore contribute nothing
   (this is [Theorem 1](#theorem-1) applied inside each round).
3. *The tool term is at most one channel use.* Write `S = (D_t, ξ_t, Q_t)`. Then
   `I(Θ;O_{t+1}|S) = Σ_s p(s) I(Θ;O_{t+1}|S=s)`. Fix `s = (d, x, q)`. Given
   `S=s`, `Θ` has law `p(·|s)` and, by (A3), `O_{t+1}` has law
   `K_t(·|Θ,q,d)`. So `I(Θ;O_{t+1}|S=s)` is the mutual information of the
   channel `K_{t,q,d}` under the input law `p(·|s)`, which is at most its
   capacity, hence at most `C_t`. Averaging over `s` gives `≤ C_t`.
   *Adaptivity is already accounted for:* the policy only picks the input law
   `p(·|s)` and the channel `K_{t,q,d}`, and capacity is the maximum over both.
4. *Sum over rounds.* `I(Θ;D_T) ≤ I(Θ;D₀) + Σ_{t<T} C_t = B_T`. Separately,
   `I(Θ;D_T) ≤ H(Θ)` because `I(Θ;D_T) = H(Θ) − H(Θ|D_T)` and `H(Θ|D_T) ≥ 0`.
5. *Fano.* For `Θ̂ = f(D_T, noise)`, Fano's inequality gives
   `H(Θ|Θ̂) ≤ 1 + ℙ(Θ̂≠Θ) log₂(|Θ|−1)`, and DPI gives
   `H(Θ|Θ̂) = H(Θ) − I(Θ;Θ̂) ≥ H(Θ) − min(H(Θ), B_T)`. Rearrange. ∎

**Corollary 7.1 (calls needed).** If `Θ` is uniform on `2^k` values (`k ≥ 2`),
`I(Θ;D₀) = 0`, and every round has capacity at most `C`, then reaching error
`ℙ(Θ̂≠Θ) ≤ ε` requires at least `T ≥ ((1−ε)k − 1)/C` rounds of tool use,
**regardless of how many agents there are or how they talk.**
*Proof.* Use `log₂(|Θ|−1) ≤ k` in [Theorem 7](#theorem-7): `ε ≥ (k − TC − 1)/k`. ∎

**Corollary 7.2 (what emergence can and cannot do).** Synergy ([Theorem 2](#theorem-2)),
routing ([Theorem 5](#theorem-5)), debate, voting and the number of agents `N` do not appear
in `B_T`. More agents raise the budget only by issuing more or better tool calls
per round, i.e. by raising `C_t`. Everything else a collective does, however
"emergent," spends a budget of at most `B_T` and cannot enlarge it. ∎

**Tightness.** The bound is attained. Let `Θ` be uniform on `{0,1,2,3}`, let a
call `q ∈ {0,1}` return bit `q` of `Θ` through a binary symmetric channel with
flip probability `p`, so `C = 1 − h(p)`, and query bit 0 then bit 1. Then
`I(Θ;D_2) = 2C`, exactly `B_2`; with `p = 0` this is `2 = H(Θ)` and `Θ` is
recovered without error. A policy that adapts badly (re-querying the same bit)
falls strictly below the budget. Machine-checked in [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py), Scenario 7.

**Scope of the assumptions.**
- (A1) is the substantive one: everything a model learned in training that
  correlates with `Θ` belongs in `D₀`. Missing a side channel breaks the bound.
- (A3) is general. If tools read a hidden world state `W` correlated with
  `Θ`, apply [Theorem 7](#theorem-7) with `W` in place of `Θ`; then, if `D₀` and every tool
  result depend on the world only through `W`, `Θ — W — D_T` and
  `I(Θ;D_T) ≤ I(W;D_T) ≤ min(H(W), I(W;D₀) + Σ C_t)`.
- (A4) excludes control. If agents can *set* `Θ` (e.g. a future state they
  cause), their success at matching it is not limited by this budget.
- The bound limits **information about `Θ`**, not computation on information
  already held. Deriving consequences, proofs or code from `D_T` needs a
  separate compute bound.
- Nothing in the proof inspects the agents. It applies to LLMs, people, and
  programs alike.

**Related work.** The closed-system special case (`C_t = 0`) for multi-agent
debate appears in Shin, *The Reasoning Trap* ([arXiv:2605.01704](https://arxiv.org/abs/2605.01704), 2026), and a
martingale analysis of debate in Choi et al., *Debate or Vote*
([arXiv:2508.17536](https://arxiv.org/abs/2508.17536), 2025). Yang et al., *Understanding Agent Scaling in
LLM-Based Multi-Agent Systems via Diversity* ([arXiv:2602.03794](https://arxiv.org/abs/2602.03794), 2026) bound
multi-agent performance by the task uncertainty `H(Y|X)` and a sum of
per-agent-channel informations, the same "min(H, Σ channels)" shape with agents
as the channels. The ingredients are classical: the chain rule and DPI (Cover &
Thomas, ch. 2), channel capacity, and Fano's inequality as used for minimax and
active-learning lower bounds (e.g. Raginsky & Rakhlin, 2011). *These
references have not yet been read in full; the exact theorem statements need
checking before relying on the comparison.*

--------------------------------------------------------------------------------
<a id="summary"></a>

## 9. Summary of the logical structure

- [Lemmas 1.1–1.5](#lemmas) are the standard information-theoretic core, proved from
  Jensen's inequality alone.
- [Theorem 1](#theorem-1) (conservation) = [Lemma 1.4](#lemmas) + DPI.
- [Theorem 3](#theorem-3) (growth) = the KL identity for conditional MI; it holds *because*
  the world kernel breaks the Markov chain of [Theorem 1](#theorem-1).
- [Theorem 4](#theorem-4) (tools) = chain rule + "queries are functions."
- [Theorem 7](#theorem-7) (capacity budget) = [Theorem 4](#theorem-4) per round + "a tool call is one use
  of a channel, so its gain is at most capacity" + Fano.
- Theorems 2, 5, 6 and the tightness of [Theorem 7](#theorem-7) reduce to finite joint-law computations, each verified
  exactly in [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py) (ALL PASSED).
- [Lemmas 6.1–6.2](#theorem-5) supply the value-of-information half (convex `V`, martingale
  posteriors), unifying routing (Thm 5) with the geometry in [`proof.md`](proof.md): the same
  Jensen gap governs action choice and experiment choice.

Every general theorem is reduced to Jensen's inequality; every numerical claim
is reduced to an exact computation that has been executed.
