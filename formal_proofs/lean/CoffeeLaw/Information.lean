/-
Copyright (c) 2026 Manish Bhatt. Released under the MIT license.
-/
import CoffeeLaw.Markov

/-!
# Data processing and output-entropy bounds

* `dpi`, `dpi_of_isMarkov`, `collective_dpi`: no agent, team or protocol applied to the
  record can know more about the world than the record does, `I(W; A) ≤ I(W; D₀, 𝒯)`.
* `mutualInfo_le_log_card`: a tool whose result takes at most `n` values reveals at most
  `log n` about the world, `I(W; O) ≤ log |𝒪|`, whatever the tool computes.
* `mutualInfo_append_le`: one tool result adds at most `log |𝒪|` to what a record already
  holds, `I(W; D, O) ≤ I(W; D) + log |𝒪|`. This is one round of the Coffee Law with the
  capacity replaced by its output-entropy bound.

Mutual information is in nats; divide by `log 2` for bits.
-/

open Finset Real

namespace CoffeeLaw

variable {W X A : Type*} [Fintype W] [Fintype X] [Fintype A]

/-- Mutual information of a joint weight table `P` on `α × β`, in nats:
`∑ P(a, b) log (P(a, b) / (P(a) P(b)))`. -/
noncomputable def mutualInfo {α β : Type*} [Fintype α] [Fintype β] (P : α → β → ℝ) : ℝ :=
  ∑ a, ∑ b, P a b * log (P a b / ((∑ b', P a b') * ∑ a', P a' b))

/-- `0 < x * y` with `x, y ≥ 0` forces `0 < x` and `0 < y`. -/
lemma pos_of_mul_pos_of_nonneg {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (h : 0 < x * y) :
    0 < x ∧ 0 < y := by
  refine ⟨lt_of_le_of_ne hx fun h' => ?_, lt_of_le_of_ne hy fun h' => ?_⟩
  · rw [← h', zero_mul] at h; exact lt_irrefl _ h
  · rw [← h', mul_zero] at h; exact lt_irrefl _ h

/-- **Data-processing inequality** for an arbitrary kernel. For any nonnegative table `μ`
on world × record and any kernel `κ` from records to outputs,
`I(W; A) ≤ I(W; X)` where `A` is the output of `κ`. -/
theorem dpi (μ : W → X → ℝ) (hμ : ∀ w x, 0 ≤ μ w x) (κ : Kernel X A) :
    mutualInfo (fun w a => ∑ x, μ w x * κ.k x a) ≤ mutualInfo μ := by
  unfold mutualInfo
  -- Marginals of the output table.
  have hW : ∀ w, ∑ a, ∑ x, μ w x * κ.k x a = ∑ x, μ w x := fun w => by
    rw [sum_comm]
    exact sum_congr rfl fun x _ => by rw [← mul_sum, κ.sum_one, mul_one]
  have hA : ∀ a, ∑ w, ∑ x, μ w x * κ.k x a = ∑ x, (∑ w, μ w x) * κ.k x a := fun a => by
    rw [sum_comm]
    exact sum_congr rfl fun x _ => by rw [sum_mul]
  simp only [hW, hA]
  -- The right-hand side, rewritten as a triple sum over (w, a, x).
  have hR : ∀ w, ∑ x, μ w x * log (μ w x / ((∑ x', μ w x') * ∑ w', μ w' x)) =
      ∑ a, ∑ x, μ w x * κ.k x a *
        log (μ w x * κ.k x a / ((∑ x', μ w x') * ((∑ w', μ w' x) * κ.k x a))) := fun w => by
    rw [sum_comm]
    refine sum_congr rfl fun x _ => ?_
    rw [← κ.sum_one x, mul_sum, mul_sum, sum_mul]
    refine sum_congr rfl fun a _ => ?_
    rw [κ.sum_one x, mul_one]
    rcases (κ.nonneg x a).eq_or_lt with h | h
    · rw [← h]; ring
    · rw [← mul_assoc, mul_div_mul_right _ _ h.ne']
      ring
  rw [sum_congr rfl fun w _ => hR w]
  -- For each (w, a), apply the log-sum inequality over x.
  refine sum_le_sum fun w _ => sum_le_sum fun a _ => ?_
  have key := log_sum univ (fun x => μ w x * κ.k x a)
    (fun x => (∑ x', μ w x') * ((∑ w', μ w' x) * κ.k x a))
    (fun x _ => mul_nonneg (hμ w x) (κ.nonneg x a))
    (fun x _ => mul_nonneg (sum_nonneg fun x' _ => hμ w x')
      (mul_nonneg (sum_nonneg fun w' _ => hμ w' x) (κ.nonneg x a)))
    (fun x _ hpos => by
      obtain ⟨h1, h2⟩ := pos_of_mul_pos_of_nonneg (hμ w x) (κ.nonneg x a) hpos
      have hw : 0 < ∑ x', μ w x' :=
        lt_of_lt_of_le h1 (single_le_sum (fun x' _ => hμ w x') (mem_univ x))
      have hx : 0 < ∑ w', μ w' x :=
        lt_of_lt_of_le h1 (single_le_sum (fun w' _ => hμ w' x) (mem_univ w))
      positivity)
  have hb : ∑ x, (∑ x', μ w x') * ((∑ w', μ w' x) * κ.k x a) =
      (∑ x', μ w x') * ∑ x, (∑ w', μ w' x) * κ.k x a := by rw [mul_sum]
  rw [hb] at key
  exact key

/-- **Data-processing inequality under the Markov condition.** If `P` is a probability law
on `W × X × A` with `W — X — A`, then `I(W; A) ≤ I(W; X)`. -/
theorem dpi_of_isMarkov (P : W → X → A → ℝ) (hP : ∀ w x a, 0 ≤ P w x a)
    (h1 : ∑ w, ∑ x, ∑ a, P w x a = 1) (hM : IsMarkov P) :
    mutualInfo (fun w a => ∑ x, P w x a) ≤ mutualInfo (fun w x => ∑ a, P w x a) := by
  obtain ⟨κ, hκ⟩ := (isMarkov_iff_kernel P hP h1).1 hM
  have e : (fun w a => ∑ x, P w x a) = fun w a => ∑ x, pWX P w x * κ.k x a := by
    funext w a
    exact sum_congr rfl fun x _ => hκ w x a
  rw [e]
  exact dpi (pWX P) (fun w x => sum_nonneg fun a _ => hP w x a) κ

/-- **The collective cannot out-know its record.** For any interactive collective (as in
`interactive_isMarkov`) whose table is a probability law, `I(W; A) ≤ I(W; D₀, 𝒯)`. -/
theorem collective_dpi {D R Tr : Type*} [Fintype D] [Fintype R] [Fintype Tr]
    (μ : W → D → ℝ) (ρ : R → ℝ) (σ : D → R → Tr → ℝ) (e : W → D → Tr → ℝ)
    (φ : D → R → Tr → A → ℝ)
    (hμ : ∀ w d, 0 ≤ μ w d) (hρ : ∀ r, 0 ≤ ρ r) (hσ : ∀ d r τ, 0 ≤ σ d r τ)
    (he : ∀ w d τ, 0 ≤ e w d τ) (hφ : ∀ d r τ a, 0 ≤ φ d r τ a)
    (h1 : ∑ w, ∑ x : D × Tr, ∑ a,
      ∑ r, μ w x.1 * ρ r * σ x.1 r x.2 * e w x.1 x.2 * φ x.1 r x.2 a = 1) :
    mutualInfo (fun w a => ∑ x : D × Tr,
        ∑ r, μ w x.1 * ρ r * σ x.1 r x.2 * e w x.1 x.2 * φ x.1 r x.2 a) ≤
      mutualInfo (fun w (x : D × Tr) => ∑ a,
        ∑ r, μ w x.1 * ρ r * σ x.1 r x.2 * e w x.1 x.2 * φ x.1 r x.2 a) :=
  dpi_of_isMarkov _ (fun w x a => sum_nonneg fun r _ => by
      have := hμ w x.1; have := hρ r; have := hσ x.1 r x.2; have := he w x.1 x.2
      have := hφ x.1 r x.2 a; positivity)
    h1 (interactive_isMarkov μ ρ σ e φ)

/-- **Output-entropy bound.** If the second coordinate takes at most `|β|` values, the mutual
information is at most `log |β|`: a tool whose result has `n` possible values reveals at most
`log n` nats (`log₂ n` bits) about the world, whatever it computes. -/
theorem mutualInfo_le_log_card {α β : Type*} [Fintype α] [Fintype β] (P : α → β → ℝ)
    (hP : ∀ a b, 0 ≤ P a b) (h1 : ∑ a, ∑ b, P a b = 1) :
    mutualInfo P ≤ log (Fintype.card β) := by
  unfold mutualInfo
  set q : β → ℝ := fun b => ∑ a', P a' b with hq
  have hq0 : ∀ b, 0 ≤ q b := fun b => sum_nonneg fun a _ => hP a b
  -- Step 1: each term is at most `-P(a, b) log q(b)`.
  have step1 : ∀ a b, P a b * log (P a b / ((∑ b', P a b') * q b)) ≤ -(P a b * log (q b)) := by
    intro a b
    rcases (hP a b).eq_or_lt with h | h
    · rw [← h]; simp
    · have hr : P a b ≤ ∑ b', P a b' := single_le_sum (fun b' _ => hP a b') (mem_univ b)
      have hqb : P a b ≤ q b := single_le_sum (fun a' _ => hP a' b) (mem_univ a)
      have hr0 : 0 < ∑ b', P a b' := lt_of_lt_of_le h hr
      have hq0' : 0 < q b := lt_of_lt_of_le h hqb
      rw [log_div h.ne' (mul_pos hr0 hq0').ne', log_mul hr0.ne' hq0'.ne']
      have := log_le_log h hr
      nlinarith
  have hsum1 : ∑ a, ∑ b, P a b * log (P a b / ((∑ b', P a b') * q b)) ≤
      ∑ b, -(q b * log (q b)) := by
    calc _ ≤ ∑ a, ∑ b, -(P a b * log (q b)) :=
          sum_le_sum fun a _ => sum_le_sum fun b _ => step1 a b
      _ = ∑ b, -(q b * log (q b)) := by
          rw [sum_comm]
          exact sum_congr rfl fun b _ => by rw [hq, sum_neg_distrib, sum_mul]
  -- Step 2: the entropy of `q` is at most `log |β|` (Gibbs against the uniform law).
  have hq1 : ∑ b, q b = 1 := by rw [hq, sum_comm]; exact h1
  have step2 : ∑ b, -(q b * log (q b)) ≤ log (Fintype.card β) := by
    have hcard : (0 : ℝ) < Fintype.card β := by
      by_contra hc
      have : Fintype.card β = 0 := by exact_mod_cast le_antisymm (not_lt.mp hc) (by positivity)
      have : IsEmpty β := Fintype.card_eq_zero_iff.mp this
      simp at hq1
    have g := gibbs univ q (fun _ => 1 / (Fintype.card β : ℝ)) (fun b _ => hq0 b)
      (fun _ _ => by positivity) (fun _ _ _ => by positivity)
    have hu : ∑ _b : β, 1 / (Fintype.card β : ℝ) = 1 := by
      rw [sum_const, card_univ, nsmul_eq_mul]; field_simp
    have hsplit : ∑ b, q b * log (q b / (1 / (Fintype.card β : ℝ))) =
        ∑ b, q b * log (q b) + log (Fintype.card β) := by
      have : ∀ b, q b * log (q b / (1 / (Fintype.card β : ℝ))) =
          q b * log (q b) + q b * log (Fintype.card β) := fun b => by
        rcases (hq0 b).eq_or_lt with h | h
        · rw [← h]; ring
        · rw [div_div_eq_mul_div, div_one, log_mul h.ne' hcard.ne']; ring
      rw [sum_congr rfl fun b _ => this b, sum_add_distrib, ← sum_mul, hq1, one_mul]
    rw [hq1, hu, hsplit] at g
    rw [sum_neg_distrib]
    linarith
  linarith

/-- **One tool result adds at most `log |𝒪|`.** For any probability law on world × record ×
result, `I(W; D, O) ≤ I(W; D) + log |𝒪|`. Applied once per round, with the record extended
by the round's queries (which add nothing, by `dpi`) and results, this gives the Coffee Law
with each round's capacity replaced by the log of the size of its result space. -/
theorem mutualInfo_append_le {D O : Type*} [Fintype D] [Fintype O] (P : W → D → O → ℝ)
    (hP : ∀ w d o, 0 ≤ P w d o) (h1 : ∑ w, ∑ d, ∑ o, P w d o = 1) :
    mutualInfo (fun w (x : D × O) => P w x.1 x.2) ≤
      mutualInfo (fun w d => ∑ o, P w d o) + log (Fintype.card O) := by
  unfold mutualInfo
  simp only [Fintype.sum_prod_type]
  -- Names for the marginals.
  set pw : W → ℝ := fun w => ∑ d, ∑ o, P w d o with hpw
  set pdo : D → O → ℝ := fun d o => ∑ w, P w d o with hpdo
  set pwd : W → D → ℝ := fun w d => ∑ o, P w d o with hpwd
  set pd : D → ℝ := fun d => ∑ w, ∑ o, P w d o with hpd
  -- Termwise: log P - log pw - log pdo ≤ (log pwd - log pw - log pd) + (log pd - log pdo).
  have term : ∀ w d o, P w d o * log (P w d o / (pw w * pdo d o)) ≤
      P w d o * log (pwd w d / (pw w * pd d)) + P w d o * log (pd d / pdo d o) := by
    intro w d o
    rcases (hP w d o).eq_or_lt with h | h
    · rw [← h]; ring_nf; simp
    · have h_wd : P w d o ≤ pwd w d := single_le_sum (fun o' _ => hP w d o') (mem_univ o)
      have h_do : P w d o ≤ pdo d o := single_le_sum (fun w' _ => hP w' d o) (mem_univ w)
      have h_w : pwd w d ≤ pw w :=
        single_le_sum (f := fun d' => ∑ o, P w d' o)
          (fun d' _ => sum_nonneg fun o' _ => hP w d' o') (mem_univ d)
      have h_d : pdo d o ≤ pd d := by
        simp only [hpdo, hpd]
        exact sum_le_sum fun w' _ =>
          single_le_sum (f := fun o' => P w' d o') (fun o' _ => hP w' d o') (mem_univ o)
      have p1 : 0 < pwd w d := lt_of_lt_of_le h h_wd
      have p2 : 0 < pdo d o := lt_of_lt_of_le h h_do
      have p3 : 0 < pw w := lt_of_lt_of_le p1 h_w
      have p4 : 0 < pd d := lt_of_lt_of_le p2 h_d
      rw [log_div h.ne' (mul_pos p3 p2).ne', log_mul p3.ne' p2.ne',
        log_div p1.ne' (mul_pos p3 p4).ne', log_mul p3.ne' p4.ne', log_div p4.ne' p2.ne']
      have := log_le_log h h_wd
      nlinarith
  -- First part sums to I(W; D).
  have part1 : ∑ w, ∑ d, ∑ o, P w d o * log (pwd w d / (pw w * pd d)) =
      ∑ w, ∑ d, (∑ o, P w d o) *
        log ((∑ o, P w d o) / ((∑ d', ∑ o, P w d' o) * ∑ w', ∑ o, P w' d o)) := by
    refine sum_congr rfl fun w _ => sum_congr rfl fun d _ => ?_
    rw [sum_mul]
  -- Second part is the conditional entropy H(O | D), at most log |𝒪|.
  have part2 : ∑ w, ∑ d, ∑ o, P w d o * log (pd d / pdo d o) ≤ log (Fintype.card O) := by
    have swap : ∑ w, ∑ d, ∑ o, P w d o * log (pd d / pdo d o) =
        ∑ d, ∑ o, pdo d o * log (pd d / pdo d o) := by
      rw [sum_comm]
      refine sum_congr rfl fun d _ => ?_
      rw [sum_comm]
      exact sum_congr rfl fun o _ => by rw [hpdo, sum_mul]
    rw [swap]
    have hpdo0 : ∀ d o, 0 ≤ pdo d o := fun d o => sum_nonneg fun w _ => hP w d o
    have hpd_eq : ∀ d, pd d = ∑ o, pdo d o := fun d => by simp only [hpd, hpdo]; exact sum_comm
    have hpdo1 : ∑ d, ∑ o, pdo d o = 1 := by
      simp only [hpdo]
      rw [sum_comm]
      refine (sum_congr rfl fun w _ => ?_).trans h1
      rw [sum_comm]
    -- `O` is nonempty, since the law has total mass one.
    have hcard : (0 : ℝ) < Fintype.card O := by
      by_contra hc
      have : Fintype.card O = 0 := by exact_mod_cast le_antisymm (not_lt.mp hc) (by positivity)
      have : IsEmpty O := Fintype.card_eq_zero_iff.mp this
      simp at h1
    -- Gibbs against `pd d / |𝒪|`, termwise.
    have g : ∀ d o, pdo d o - pd d / Fintype.card O ≤
        log (Fintype.card O) * pdo d o - pdo d o * log (pd d / pdo d o) := by
      intro d o
      have hle : pdo d o ≤ pd d := by
        rw [hpd_eq]; exact single_le_sum (fun o' _ => hpdo0 d o') (mem_univ o)
      have hb : 0 ≤ pd d / Fintype.card O :=
        div_nonneg (le_trans (hpdo0 d o) hle) hcard.le
      have := sub_le_mul_log_div (hpdo0 d o) hb fun h => div_pos (lt_of_lt_of_le h hle) hcard
      rcases (hpdo0 d o).eq_or_lt with h | h
      · rw [← h] at this ⊢; simpa using this
      · have p4 : 0 < pd d := lt_of_lt_of_le h hle
        rw [div_div_eq_mul_div, log_div (mul_pos h hcard).ne' p4.ne',
          log_mul h.ne' hcard.ne'] at this
        rw [log_div p4.ne' h.ne']
        linarith
    have gs := sum_le_sum fun d (_ : d ∈ univ) => sum_le_sum fun o (_ : o ∈ univ) => g d o
    simp only [sum_sub_distrib, ← mul_sum, ← sum_div] at gs
    have hpd1 : ∑ d, ∑ _o : O, pd d / (Fintype.card O : ℝ) = 1 := by
      have : ∀ d, ∑ _o : O, pd d / (Fintype.card O : ℝ) = pd d := fun d => by
        rw [sum_const, card_univ, nsmul_eq_mul]; field_simp
      rw [sum_congr rfl fun d _ => this d]
      simp only [hpd_eq]
      exact hpdo1
    have hlog : ∑ d, log (Fintype.card O) * ∑ o, pdo d o = log (Fintype.card O) := by
      rw [← mul_sum, hpdo1, mul_one]
    rw [hpdo1, hpd1, hlog] at gs
    linarith
  calc _ ≤ ∑ w, ∑ d, ∑ o,
        (P w d o * log (pwd w d / (pw w * pd d)) + P w d o * log (pd d / pdo d o)) :=
        sum_le_sum fun w _ => sum_le_sum fun d _ => sum_le_sum fun o _ => term w d o
    _ = ∑ w, ∑ d, ∑ o, P w d o * log (pwd w d / (pw w * pd d)) +
        ∑ w, ∑ d, ∑ o, P w d o * log (pd d / pdo d o) := by
        simp only [sum_add_distrib]
    _ ≤ _ := by rw [part1]; linarith

end CoffeeLaw
