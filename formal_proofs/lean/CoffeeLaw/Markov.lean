/-
Copyright (c) 2026 Manish Bhatt. Released under the MIT license.
-/
import CoffeeLaw.Basic
import Mathlib.Algebra.BigOperators.Field

/-!
# The baseline invariant: `W — (D₀, 𝒯) — A` for any collective

`W` is the world (the truth), `D₀` the collective's starting data, `𝒯` the full
tool transcript (every query and every tool result) and `A` anything the
collective outputs. This file proves that `W` and `A` are conditionally
independent given the record `(D₀, 𝒯)`, for

* any number of agents and any communication protocol,
* any adaptive query policy, over any number of rounds,
* any internal randomness, as long as it is drawn independently of the world,
* any tools, as long as they see the agents only through their queries.

Nothing is parametric: the agents' policies, the tools' kernels and the output
map are arbitrary functions of the right arguments. In particular nothing
depends on how an agent is implemented (an LLM, a person, a program).

The core fact is `isMarkov_of_factor`: a joint weight table that splits as
`F(w, x) · G(x, a)` is Markov. `interactive_isMarkov` shows that every
interactive collective produces such a table, and `isMarkov_iff_kernel` shows
the Markov property is exactly "the output is a stochastic function of the
record".
-/

open Finset Real

namespace CoffeeLaw

/-- A probability law on a finite type: any nonnegative table that sums to one. -/
def IsLaw {ι : Type*} [Fintype ι] (p : ι → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ ∑ i, p i = 1

/-- A Markov kernel (stochastic matrix) from `X` to `A`. An agent, a team of agents or a
whole protocol that maps a record `x` to a random output is exactly such an object; there
is no restriction on the form of `k`. -/
structure Kernel (X A : Type*) [Fintype A] where
  /-- `k x a` is the probability of output `a` given record `x`. -/
  k : X → A → ℝ
  nonneg : ∀ x a, 0 ≤ k x a
  sum_one : ∀ x, ∑ a, k x a = 1

variable {W X A : Type*} [Fintype W] [Fintype X] [Fintype A]

section Marginals

variable (P : W → X → A → ℝ)

/-- Marginal of the record. -/
def pX (x : X) : ℝ := ∑ w, ∑ a, P w x a

/-- Joint marginal of world and record. -/
def pWX (w : W) (x : X) : ℝ := ∑ a, P w x a

/-- Joint marginal of record and output. -/
def pXA (x : X) (a : A) : ℝ := ∑ w, P w x a

/-- `W — X — A`: the world and the output are conditionally independent given the record,
`p(w, x, a) p(x) = p(w, x) p(x, a)`. Written without division, so it also covers records of
probability zero. -/
def IsMarkov : Prop :=
  ∀ w x a, P w x a * pX P x = pWX P w x * pXA P x a

end Marginals

omit [Fintype X] in
/-- **Factorization implies Markov.** If the joint table splits as `F(w, x) · G(x, a)`, then
`W — X — A`. No positivity, normalization or parametric form is needed. -/
theorem isMarkov_of_factor (P : W → X → A → ℝ) (F : W → X → ℝ) (G : X → A → ℝ)
    (h : ∀ w x a, P w x a = F w x * G x a) : IsMarkov P := by
  intro w x a
  simp only [pX, pWX, pXA, h]
  have e1 : ∑ w', ∑ a', F w' x * G x a' = (∑ w', F w' x) * ∑ a', G x a' := by
    rw [sum_mul_sum]
  have e2 : ∑ a', F w x * G x a' = F w x * ∑ a', G x a' := by rw [mul_sum]
  have e3 : ∑ w', F w' x * G x a = (∑ w', F w' x) * G x a := by rw [sum_mul]
  rw [e1, e2, e3]
  ring

omit [Fintype X] in
/-- Any agent applied to the record gives a Markov chain: `W — X — A` holds for the joint
`μ(w, x) κ(x, a)`, whatever the law `μ` of world and record and whatever the kernel `κ`. -/
theorem isMarkov_of_kernel (μ : W → X → ℝ) (κ : Kernel X A) :
    IsMarkov (fun w x a => μ w x * κ.k x a) :=
  isMarkov_of_factor _ μ κ.k fun _ _ _ => rfl

/-- **Characterization.** For a probability law on `W × X × A`, the Markov property holds
iff the output is a stochastic function of the record: `p(w, x, a) = p(w, x) κ(x, a)` for
some kernel `κ`. -/
theorem isMarkov_iff_kernel (P : W → X → A → ℝ) (hP : ∀ w x a, 0 ≤ P w x a)
    (h1 : ∑ w, ∑ x, ∑ a, P w x a = 1) :
    IsMarkov P ↔ ∃ κ : Kernel X A, ∀ w x a, P w x a = pWX P w x * κ.k x a := by
  classical
  constructor
  · intro hM
    have hcard : (Fintype.card A : ℝ) ≠ 0 := by
      intro hc
      have : IsEmpty A := Fintype.card_eq_zero_iff.mp (by exact_mod_cast hc)
      simp at h1
    have hpX : ∀ x, 0 ≤ pX P x := fun x => sum_nonneg fun w _ => sum_nonneg fun a _ => hP w x a
    refine ⟨⟨fun x a => if pX P x = 0 then 1 / (Fintype.card A : ℝ)
        else pXA P x a / pX P x, ?_, ?_⟩, ?_⟩
    · intro x a
      split_ifs
      · positivity
      · exact div_nonneg (sum_nonneg fun w _ => hP w x a) (hpX x)
    · intro x
      by_cases hx : pX P x = 0
      · simp only [hx, ite_true, sum_const, card_univ, nsmul_eq_mul]
        field_simp
      · simp only [hx, ite_false]
        rw [← sum_div]
        have : ∑ a, pXA P x a = pX P x := by
          simp only [pXA, pX]
          exact sum_comm
        rw [this, div_self hx]
    · intro w x a
      by_cases hx : pX P x = 0
      · have hw : pWX P w x = 0 :=
          (sum_eq_zero_iff_of_nonneg fun w' _ => sum_nonneg fun a' _ => hP w' x a').1 hx w
            (mem_univ w)
        have ha : P w x a = 0 :=
          (sum_eq_zero_iff_of_nonneg fun a' _ => hP w x a').1 hw a (mem_univ a)
        simp only [hx, ite_true, hw, ha, zero_mul]
      · simp only [hx, ite_false]
        rw [mul_div_assoc', eq_div_iff hx]
        exact hM w x a
  · rintro ⟨κ, hκ⟩
    exact isMarkov_of_factor P (pWX P) κ.k hκ

/-! ### Interactive collectives -/

/-- **The baseline invariant for interactive collectives.** Let

* `μ w d` be the joint law of the world `w` and the starting data `d`,
* `ρ r` be the law of all internal randomness `r` (sampling, scheduling, every agent's
  private coins, for every round), drawn independently of `(w, d)`,
* `σ d r τ` be the agents' side of a transcript `τ`: the weight their policy (any number of
  agents, any protocol, any adaptive choice of queries) gives to the queries in `τ`,
* `e w d τ` be the tools' side: the weight the tools give to the results in `τ` when queried
  as in `τ`. Tools see the world, the starting data and the transcript, but not `r`,
* `φ d r τ a` be the law of the collective's output `a` (an answer, a vote, a board, any
  agent's state).

Then the world and the output are conditionally independent given the record `(d, τ)`.
The functions are arbitrary; causality (queries depend only on earlier results) is what
makes the table a probability law, but the Markov property does not need it. -/
theorem interactive_isMarkov {D R Tr : Type*} [Fintype D] [Fintype R] [Fintype Tr]
    (μ : W → D → ℝ) (ρ : R → ℝ) (σ : D → R → Tr → ℝ) (e : W → D → Tr → ℝ)
    (φ : D → R → Tr → A → ℝ) :
    IsMarkov (fun (w : W) (x : D × Tr) (a : A) =>
      ∑ r, μ w x.1 * ρ r * σ x.1 r x.2 * e w x.1 x.2 * φ x.1 r x.2 a) := by
  refine isMarkov_of_factor _ (fun w x => μ w x.1 * e w x.1 x.2)
    (fun x a => ∑ r, ρ r * σ x.1 r x.2 * φ x.1 r x.2 a) fun w x a => ?_
  rw [mul_sum]
  exact sum_congr rfl fun r _ => by ring

/-- **`T` rounds of adaptive tool use.** The transcript is `τ : Fin T → Q × O` (the query and
the result of each round). In round `t` the collective issues the query `g t d r τ` and the
tool answers with law `K t w d τ`. The agents' side is the indicator that every query in `τ`
is the one the policy would issue, and the tools' side is the product of the tool kernels.
Then `W — (D₀, 𝒯) — A`. (For the table to be a probability law, `g t` should read only
rounds `< t` of `τ` and `K t` only rounds `< t` plus the query of round `t`.) -/
theorem adaptive_isMarkov {D R Q O : Type*} [Fintype D] [Fintype R] [Fintype Q] [Fintype O]
    [DecidableEq Q] (T : ℕ) (μ : W → D → ℝ) (ρ : R → ℝ)
    (g : Fin T → D → R → (Fin T → Q × O) → Q)
    (K : Fin T → W → D → (Fin T → Q × O) → O → ℝ)
    (φ : D → R → (Fin T → Q × O) → A → ℝ) :
    IsMarkov (fun (w : W) (x : D × (Fin T → Q × O)) (a : A) =>
      ∑ r, μ w x.1 * ρ r * (∏ t, if (x.2 t).1 = g t x.1 r x.2 then (1 : ℝ) else 0) *
        (∏ t, K t w x.1 x.2 (x.2 t).2) * φ x.1 r x.2 a) :=
  interactive_isMarkov μ ρ (fun d r τ => ∏ t, if (τ t).1 = g t d r τ then (1 : ℝ) else 0)
    (fun w d τ => ∏ t, K t w d τ (τ t).2) φ

/-! ### Building agents and protocols as kernels -/

namespace Kernel

variable {B R : Type*} [Fintype B] [Fintype R]

/-- An agent that applies an arbitrary function `f` to the record and to private randomness
`r ∼ ρ` drawn independently of everything else. -/
def ofFun [DecidableEq A] (ρ : R → ℝ) (hρ : IsLaw ρ) (f : X → R → A) : Kernel X A where
  k x a := ∑ r, if f x r = a then ρ r else 0
  nonneg x a := sum_nonneg fun r _ => by
    split_ifs
    · exact hρ.1 r
    · exact le_rfl
  sum_one x := by
    rw [sum_comm]
    simp [hρ.2]

/-- Two agents in sequence: the second sees the record and the first agent's message. Any
finite protocol is built by iterating this and `comp`. -/
def seq (κ₁ : Kernel X A) (κ₂ : Kernel (X × A) B) : Kernel X (A × B) where
  k x ab := κ₁.k x ab.1 * κ₂.k (x, ab.1) ab.2
  nonneg _ _ := mul_nonneg (κ₁.nonneg _ _) (κ₂.nonneg _ _)
  sum_one x := by
    rw [Fintype.sum_prod_type]
    simp only [← mul_sum, κ₂.sum_one, mul_one, κ₁.sum_one]

/-- Post-processing: a kernel applied to the output of another kernel. -/
def comp (κ₁ : Kernel X A) (κ₂ : Kernel A B) : Kernel X B where
  k x b := ∑ a, κ₁.k x a * κ₂.k a b
  nonneg _ _ := sum_nonneg fun _ _ => mul_nonneg (κ₁.nonneg _ _) (κ₂.nonneg _ _)
  sum_one x := by
    rw [sum_comm]
    simp only [← mul_sum, κ₂.sum_one, mul_one, κ₁.sum_one]

end Kernel

end CoffeeLaw
