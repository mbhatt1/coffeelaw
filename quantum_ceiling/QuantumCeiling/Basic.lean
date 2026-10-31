/-
Copyright (c) 2026 Manish Bhatt. Released under the MIT license.
-/

/-!
# A Coffee-Law-style ceiling on quantum registers

The Coffee Law says a collective can never hold more than its information
budget. This file applies the same pattern to a quantum register:

* **usable ≤ budget.** A theory of physics fixes how many distinct joint
  configurations an `n`-qubit register can actually reach and use.
* **advantage needs the full space.** Exploiting all `n` qubits (the regime in
  which Shor-type speedups live) means reaching all `2^n` basis directions.
* **ceiling.** If the theory caps usable configurations at `2^k` regardless of
  register size, no register, however large and however error-corrected, gets
  past `k` usable qubits.

This is a deliberately coarse model with a size-independent budget. The faithful
formalization of Palmer's own counting argument, where the budget grows linearly
as `N * L`, is in `QuantumCeiling/Palmer.lean`.

## What this does and does not prove

Everything here is a *conditional* theorem. `ceiling` shows that **if**
physics has a finite, size-independent configuration budget (the premise
`Budgeted`, an abstraction of a discretized or "rational" Hilbert space),
**then** the ceiling follows. `standardQM_unbounded` shows that standard quantum
mechanics, modeled with no budget, satisfies the advantage condition for every
`n` and is not `Budgeted` for any finite budget. Both models are consistent, so
logic alone cannot decide which one describes nature; experiment does.
`ceiling_falsified` states exactly which experiment would rule the ceiling out.

The premise is a modeling choice, not a reconstruction of any paper's derivation.
-/

namespace QuantumCeiling

/-- A physical theory, seen only through its information budget:
`usable n` is the number of distinct joint configurations an `n`-qubit register
can reach and use in a computation. -/
structure Theory where
  usable : Nat → Nat

/-- Full quantum advantage on `n` qubits: the register reaches all `2^n` basis
directions. -/
def Advantage (T : Theory) (n : Nat) : Prop :=
  2 ^ n ≤ T.usable n

/-- The ceiling premise: the theory's budget is at most `B`, whatever the
register size. -/
def Budgeted (T : Theory) (B : Nat) : Prop :=
  ∀ n, T.usable n ≤ B

/-- Exponents are monotone: `2^n ≤ 2^k` forces `n ≤ k`. -/
theorem le_of_two_pow_le {n k : Nat} (h : 2 ^ n ≤ 2 ^ k) : n ≤ k := by
  apply Nat.le_of_not_lt
  intro hlt
  have : 2 ^ k < 2 ^ n := Nat.pow_lt_pow_right (by decide) hlt
  omega

/-- **Ceiling theorem.** Under a `2^k` budget, advantage on `n` qubits forces
`n ≤ k`. -/
theorem ceiling {T : Theory} {k n : Nat}
    (hB : Budgeted T (2 ^ k)) (hA : Advantage T n) : n ≤ k :=
  le_of_two_pow_le (Nat.le_trans hA (hB n))

/-- The 1,000-qubit instance: a `2^1000` budget allows at most 1,000
usable qubits. -/
theorem ceiling_1000 {T : Theory} {n : Nat}
    (hB : Budgeted T (2 ^ 1000)) (hA : Advantage T n) : n ≤ 1000 :=
  ceiling hB hA

/-- Equivalently: under that budget, advantage at 1,001 qubits is impossible. -/
theorem no_advantage_past_1000 {T : Theory}
    (hB : Budgeted T (2 ^ 1000)) : ¬ Advantage T 1001 := by
  intro hA
  have := ceiling_1000 hB hA
  omega

/-! ## Error correction does not raise the budget

The analogue of "debate adds nothing": encoding `m` logical qubits in `p`
physical qubits does not lift the ceiling. However many physical qubits a code
spends, the logical register's configurations live inside the physical
register's, so the same budget caps the logical count. -/

/-- An error-correcting encoding of `logical` qubits in `physical` qubits that
preserves advantage: every logical basis direction is realized by a usable
physical configuration. -/
structure Encoding (T : Theory) where
  physical : Nat
  logical : Nat
  faithful : 2 ^ logical ≤ T.usable physical

/-- **No code beats the budget.** Every faithful encoding under a `2^k` budget
has at most `k` logical qubits, independent of its physical size. -/
theorem encoding_ceiling {T : Theory} {k : Nat}
    (hB : Budgeted T (2 ^ k)) (E : Encoding T) : E.logical ≤ k :=
  le_of_two_pow_le (Nat.le_trans E.faithful (hB E.physical))

/-! ## Standard quantum mechanics: no ceiling

Modeled with the full `2^n`-dimensional space and no discretization, standard
quantum mechanics has advantage at every size and no finite budget. The ceiling
premise is therefore an independent physical assumption, not a consequence of
the formalism. -/

/-- Standard quantum mechanics: the whole `2^n` space is usable. -/
def standardQM : Theory := ⟨fun n => 2 ^ n⟩

theorem standardQM_advantage (n : Nat) : Advantage standardQM n :=
  Nat.le_refl _

theorem standardQM_unbounded (B : Nat) : ¬ Budgeted standardQM B := by
  intro h
  have h1 : 2 ^ B ≤ B := h B
  have h2 : B < 2 ^ B := Nat.lt_two_pow_self
  omega

/-- A minimal model satisfying the ceiling premise, so the premise is
consistent: usable configurations saturate at `2^k`. -/
def cappedQM (k : Nat) : Theory := ⟨fun n => 2 ^ Nat.min n k⟩

theorem cappedQM_budgeted (k : Nat) : Budgeted (cappedQM k) (2 ^ k) := by
  intro n
  exact Nat.pow_le_pow_right (by decide) (Nat.min_le_right n k)

theorem cappedQM_advantage {k n : Nat} (h : n ≤ k) : Advantage (cappedQM k) n := by
  exact Nat.pow_le_pow_right (by decide) (Nat.le_min.mpr ⟨Nat.le_refl n, h⟩)

/-- **Independence.** Logic admits both a world with a 1,000-qubit ceiling and
one with none. -/
theorem ceiling_independent :
    (∃ T : Theory, Budgeted T (2 ^ 1000) ∧ Advantage T 1000) ∧
    (∃ T : Theory, (∀ n, Advantage T n) ∧ ∀ B, ¬ Budgeted T B) :=
  ⟨⟨cappedQM 1000, cappedQM_budgeted 1000, cappedQM_advantage (Nat.le_refl _)⟩,
   ⟨standardQM, standardQM_advantage, standardQM_unbounded⟩⟩

/-! ## The deciding experiment -/

/-- **Falsification.** A verified demonstration of advantage on more than `k`
qubits refutes every `2^k` budget, for any theory. With `k = 1000`: one
confirmed, fully error-corrected computation using 1,001 logical qubits refutes
the ceiling. -/
theorem ceiling_falsified {T : Theory} {k n : Nat}
    (hA : Advantage T n) (hn : k < n) : ¬ Budgeted T (2 ^ k) := by
  intro hB
  have := ceiling hB hA
  omega

theorem ceiling_1000_falsified_by_encoding {T : Theory} (E : Encoding T)
    (h : 1000 < E.logical) : ¬ Budgeted T (2 ^ 1000) := by
  intro hB
  have := encoding_ceiling hB E
  omega

end QuantumCeiling
