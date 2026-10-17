/-
Copyright (c) 2026 Manish Bhatt. Released under the MIT license.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Gibbs' inequality and the log-sum inequality

Everything in this development is about finite, real-valued weight tables. No
distribution is assumed to belong to a parametric family: a "law" is any
nonnegative table that sums to one, and an "agent" is any stochastic matrix.

Logarithms are natural (`Real.log`). Every inequality proved here is homogeneous
in `log`, so it holds verbatim in bits after dividing by `log 2`.

Lean's conventions `x / 0 = 0` and `log 0 = 0` make the usual convention
`0 · log(0 / q) = 0` automatic: a term with zero weight contributes zero.
-/

open Finset Real

namespace CoffeeLaw

/-- Termwise Gibbs: `a - b ≤ a · log (a / b)` whenever `a, b ≥ 0` and `a > 0 → b > 0`. -/
lemma sub_le_mul_log_div {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a → 0 < b) :
    a - b ≤ a * log (a / b) := by
  rcases ha.eq_or_lt with h | h
  · subst h
    simp only [zero_sub, zero_mul]
    linarith
  · have hb' : 0 < b := hab h
    have h1 : log (b / a) ≤ b / a - 1 := log_le_sub_one_of_pos (div_pos hb' h)
    have h2 : log (a / b) = -log (b / a) := by rw [← log_inv, inv_div]
    have h3 : a * log (b / a) ≤ b - a := by
      calc a * log (b / a) ≤ a * (b / a - 1) := mul_le_mul_of_nonneg_left h1 h.le
        _ = b - a := by field_simp
    rw [h2]
    linarith

/-- **Gibbs' inequality** in unnormalized form:
`∑ aᵢ - ∑ bᵢ ≤ ∑ aᵢ log (aᵢ / bᵢ)`. With `∑ aᵢ = ∑ bᵢ = 1` this is `KL(a ‖ b) ≥ 0`. -/
lemma gibbs {ι : Type*} (s : Finset ι) (a b : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i)
    (hb : ∀ i ∈ s, 0 ≤ b i) (hab : ∀ i ∈ s, 0 < a i → 0 < b i) :
    ∑ i ∈ s, a i - ∑ i ∈ s, b i ≤ ∑ i ∈ s, a i * log (a i / b i) := by
  rw [← sum_sub_distrib]
  exact sum_le_sum fun i hi => sub_le_mul_log_div (ha i hi) (hb i hi) (hab i hi)

/-- **Log-sum inequality**:
`(∑ aᵢ) log ((∑ aᵢ) / (∑ bᵢ)) ≤ ∑ aᵢ log (aᵢ / bᵢ)`. -/
lemma log_sum {ι : Type*} (s : Finset ι) (a b : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i)
    (hb : ∀ i ∈ s, 0 ≤ b i) (hab : ∀ i ∈ s, 0 < a i → 0 < b i) :
    (∑ i ∈ s, a i) * log ((∑ i ∈ s, a i) / ∑ i ∈ s, b i) ≤ ∑ i ∈ s, a i * log (a i / b i) := by
  rcases (sum_nonneg ha).eq_or_lt with h0 | hA
  · -- All weights vanish, so both sides are zero.
    have hz : ∀ i ∈ s, a i = 0 := (sum_eq_zero_iff_of_nonneg ha).1 h0.symm
    rw [← h0, zero_mul]
    exact le_of_eq (sum_eq_zero fun i hi => by rw [hz i hi, zero_mul]).symm
  · obtain ⟨j, hj, haj⟩ : ∃ j ∈ s, 0 < a j := by
      by_contra hc
      have : ∑ i ∈ s, a i ≤ 0 :=
        sum_nonpos fun i hi => not_lt.mp fun h => hc ⟨i, hi, h⟩
      linarith
    have hB : 0 < ∑ i ∈ s, b i := lt_of_lt_of_le (hab j hj haj) (single_le_sum hb hj)
    have hc : 0 < (∑ i ∈ s, a i) / ∑ i ∈ s, b i := div_pos hA hB
    have key := gibbs s a (fun i => b i * ((∑ i ∈ s, a i) / ∑ i ∈ s, b i)) ha
      (fun i hi => mul_nonneg (hb i hi) hc.le) (fun i hi h => mul_pos (hab i hi h) hc)
    have hsum : ∑ i ∈ s, b i * ((∑ i ∈ s, a i) / ∑ i ∈ s, b i) = ∑ i ∈ s, a i := by
      rw [← sum_mul]
      field_simp
    have hsplit : ∑ i ∈ s, a i * log (a i / (b i * ((∑ i ∈ s, a i) / ∑ i ∈ s, b i))) =
        ∑ i ∈ s, a i * log (a i / b i) -
          (∑ i ∈ s, a i) * log ((∑ i ∈ s, a i) / ∑ i ∈ s, b i) := by
      rw [sum_mul, ← sum_sub_distrib]
      refine sum_congr rfl fun i hi => ?_
      rcases (ha i hi).eq_or_lt with h | h
      · rw [← h]; ring
      · have hbi := hab i hi h
        rw [div_mul_eq_div_div, log_div (div_pos h hbi).ne' hc.ne']
        ring
    rw [hsum, hsplit] at key
    linarith

end CoffeeLaw
