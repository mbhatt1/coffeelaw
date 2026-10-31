/-
Copyright (c) 2026 Manish Bhatt. Released under the MIT license.
-/
import Mathlib.LinearAlgebra.Projectivization.Basic
import Mathlib.Basic.Complex.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Nat.Log
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Positivity

/-!
# Linear information budgets cannot carry `N`-qubit quantum mechanics

T. N. Palmer (*Rational Quantum Mechanics: Testing Quantum Theory with Quantum
Computers*, arXiv:2510.02877) proposes that `N` entangled qubits carry only
`N · L` bits of information, for a finite per-qubit capacity `L` fixed by
gravity. He argues that this cannot cover the `2^(N+1) - 2` real degrees of
freedom of complex Hilbert space once `N` exceeds roughly `log₂ L`, and that
exponential quantum speedups therefore saturate near 1,000 qubits.

Here that counting argument is not assumed but **derived** from the structure of
quantum state space.

* Pure states of `N` qubits are rays in `ℂ^(2^N)`: Mathlib's
  `Projectivization ℂ (Basis N → ℂ)`.
* For every Boolean function `f` on `N` bits, the phase state
  `Σₓ (-1)^f(x) |x⟩` is a pure state. These are the oracle states of
  Deutsch–Jozsa, Bernstein–Vazirani and Grover. Every amplitude has squared
  modulus `1/2^N` and phase `0` or `π`, so these states satisfy RaQM's
  rationality constraints, and the rationality rule cannot be what excludes them.
* `phaseState_injective`: distinct normalized `f` give physically distinct
  states, so there are `2^(2^N - 1)` of them (`card_phaseStates`).

**Main theorem (`budget_forces_ceiling`).** If a theory's `N`-qubit state space
has at most `2^(N·L)` states (an `N·L`-bit budget, Palmer's premise) and still
contains every phase state, then `2^N - 1 ≤ N·L`.

The bound is half of Palmer's `2^(N+1) - 2`. Palmer charges one bit per *real*
degree of freedom. Phase states only use the sign of each amplitude, one bit per
basis state, so this lower bound is weaker but needs no approximation argument.

**Consequences.**
* `ceiling_permanent`: once the inequality fails it fails for every larger `N`.
* `ceiling_log`: `N ≤ log₂ (N·L + 1)`, i.e. `N_max ≈ log₂ L`.
* `ceiling_1000`: for `L ≤ 2^990`, which covers every estimate in the paper,
  `N ≤ 999`.
* `no_linear_budget`: no `N·L`-bit theory, whatever `L` is, contains the phase
  states on every register size. So Palmer's premise and standard quantum
  mechanics are exactly incompatible, and a phase state on more than about
  `log₂ L` qubits tells them apart.

What remains physics, not mathematics: whether nature's `N`-qubit state space
really has an `N·L`-bit budget, and what `L` is. Those are hypotheses here.
-/

namespace QuantumCeiling

open Function

noncomputable section

-- `ceiling_1000` evaluates `2^1000` (kernel GMP arithmetic).
set_option exponentiation.threshold 2048

/-- Computational-basis labels of an `N`-qubit register: bit strings of length `N`. -/
abbrev Basis (N : ℕ) := Fin N → Bool

/-- The `N`-qubit Hilbert space `ℂ^(2^N)`, with coordinates indexed by bit strings. -/
abbrev Hilb (N : ℕ) := Basis N → ℂ

/-- Pure states of `N` qubits: rays in Hilbert space (vectors modulo nonzero
scalars, so normalization and global phase are quotiented out). -/
abbrev PureState (N : ℕ) := Projectivization ℂ (Hilb N)

/-- The all-zeros bit string. -/
def zero (N : ℕ) : Basis N := fun _ => false

/-- The amplitude `(-1)^b`. -/
def sign (b : Bool) : ℂ := if b then -1 else 1

theorem sign_ne_zero (b : Bool) : sign b ≠ 0 := by
  cases b <;> norm_num [sign]

theorem sign_injective : Injective sign := by
  intro a b h
  cases a <;> cases b <;> first | rfl | (exfalso; norm_num [sign] at h)

/-- Every phase amplitude has unit modulus: after normalization by `2^(-N/2)`
each squared amplitude is exactly `1/2^N`, and each phase is `0` or `π`. -/
theorem normSq_sign (b : Bool) : Complex.normSq (sign b) = 1 := by
  cases b <;> simp [sign]

/-- The (unnormalized) phase vector `Σₓ (-1)^f(x) |x⟩` of a Boolean function. -/
def phaseVec {N : ℕ} (f : Basis N → Bool) : Hilb N := fun x => sign (f x)

theorem phaseVec_ne_zero {N : ℕ} (f : Basis N → Bool) : phaseVec f ≠ 0 := by
  intro h
  exact sign_ne_zero (f (zero N)) (congrFun h (zero N))

/-- The phase state of `f`, as a physical (projective) pure state. -/
def phaseState {N : ℕ} (f : Basis N → Bool) : PureState N :=
  Projectivization.mk ℂ (phaseVec f) (phaseVec_ne_zero f)

/-- Two phase states agreeing on the all-zeros string are the same physical state
only if the Boolean functions are equal. -/
theorem phaseState_inj {N : ℕ} {f g : Basis N → Bool}
    (hf : f (zero N) = false) (hg : g (zero N) = false)
    (h : phaseState f = phaseState g) : f = g := by
  rw [phaseState, phaseState, Projectivization.mk_eq_mk_iff] at h
  obtain ⟨a, ha⟩ := h
  have h0 := congrFun ha (zero N)
  simp only [Pi.smul_apply, phaseVec, hf, hg, sign, Units.smul_def, smul_eq_mul] at h0
  norm_num at h0
  funext x
  have hx := congrFun ha x
  simp only [Pi.smul_apply, phaseVec, Units.smul_def, smul_eq_mul, h0, Units.val_one, one_mul] at hx
  exact (sign_injective hx).symm

/-- Bit strings other than all-zeros. -/
abbrev NonZero (N : ℕ) := {x : Basis N // x ≠ zero N}

/-- Extend a sign pattern on nonzero strings to a Boolean function that is
`false` on the all-zeros string (fixing the global phase). -/
def extend {N : ℕ} (s : NonZero N → Bool) : Basis N → Bool := fun x =>
  if h : x = zero N then false else s ⟨x, h⟩

/-- Phase states, indexed by sign patterns with the global phase fixed. -/
def phaseStateOf {N : ℕ} (s : NonZero N → Bool) : PureState N := phaseState (extend s)

theorem phaseState_injective (N : ℕ) : Injective (phaseStateOf (N := N)) := by
  intro s t h
  have := phaseState_inj (by simp [extend]) (by simp [extend]) h
  funext x
  have hx := congrFun this x.1
  simpa [extend, x.2] using hx

theorem card_basis (N : ℕ) : Fintype.card (Basis N) = 2 ^ N := by
  simp

theorem card_nonZero (N : ℕ) : Fintype.card (NonZero N) = 2 ^ N - 1 := by
  simp [NonZero, Fintype.card_subtype_compl]

/-- There are `2^(2^N - 1)` distinct phase states on `N` qubits. -/
theorem card_phaseStates (N : ℕ) : Fintype.card (NonZero N → Bool) = 2 ^ (2 ^ N - 1) := by
  simp

/-! ## The budget theorem -/

/-- **Main theorem.** A set of `N`-qubit pure states that contains every phase
state but has at most `2^(N·L)` elements (an `N·L`-bit information budget)
forces `2^N - 1 ≤ N·L`. -/
theorem budget_forces_ceiling {N L : ℕ} (S : Finset (PureState N))
    (hbudget : S.card ≤ 2 ^ (N * L))
    (hphase : ∀ s : NonZero N → Bool, phaseStateOf s ∈ S) :
    2 ^ N - 1 ≤ N * L := by
  let e : (NonZero N → Bool) → S := fun s => ⟨phaseStateOf s, hphase s⟩
  have he : Injective e := fun s t h =>
    phaseState_injective N (congrArg Subtype.val h)
  have hcard := Fintype.card_le_of_injective e he
  rw [card_phaseStates, Fintype.card_coe] at hcard
  exact (Nat.pow_le_pow_iff_right (by norm_num)).1 (hcard.trans hbudget)

/-! ## Consequences: the ceiling -/

/-- Budget feasibility for register size `N` and per-qubit capacity `L`. -/
def Feasible (L N : ℕ) : Prop := 2 ^ N - 1 ≤ N * L

/-- Once the budget falls short it stays short: demand doubles with each qubit,
the budget only grows by `L`. -/
theorem infeasible_succ {L N : ℕ} (hN : 1 ≤ N) (h : ¬ Feasible L N) :
    ¬ Feasible L (N + 1) := by
  unfold Feasible at *
  have hL : L ≤ N * L := Nat.le_mul_of_pos_left L hN
  have hs : (N + 1) * L = N * L + L := Nat.succ_mul N L
  have hp : 2 ^ (N + 1) = 2 * 2 ^ N := by rw [Nat.pow_succ, Nat.mul_comm]
  rw [hs, hp]
  omega

/-- **The ceiling is permanent.** -/
theorem ceiling_permanent {L N : ℕ} (hN : 1 ≤ N) (h : ¬ Feasible L N) :
    ∀ M, N ≤ M → ¬ Feasible L M := by
  intro M hM
  induction M, hM using Nat.le_induction with
  | base => exact h
  | succ M hNM ih => exact infeasible_succ (hN.trans hNM) ih

/-- **`N_max ≈ log₂ L`.** Any feasible register satisfies `N ≤ log₂ (N·L + 1)`. -/
theorem ceiling_log {L N : ℕ} (h : Feasible L N) : N ≤ Nat.log 2 (N * L + 1) :=
  Nat.le_log_of_pow_le (by norm_num) (by unfold Feasible at h; have := Nat.one_le_two_pow (n := N); omega)

/-- **The 1,000-qubit figure.** For any `L ≤ 2^990` (every estimate in the paper,
including the `L ≈ 10^193 ≈ 2^640` of Palmer's 2025 talk), at most 999 qubits. -/
theorem ceiling_1000 {L N : ℕ} (hL : L ≤ 2 ^ 990) (h : Feasible L N) : N ≤ 999 := by
  by_contra hN
  have h1000 : ¬ Feasible L 1000 := by
    unfold Feasible
    have : 1000 * L ≤ 1000 * 2 ^ 990 := Nat.mul_le_mul_left 1000 hL
    have : 1000 * 2 ^ 990 < 2 ^ 1000 - 1 := by norm_num
    omega
  exact ceiling_permanent (by norm_num) h1000 N (by omega) h

/-- **Combined statement.** Under an `N·L`-bit budget with `L ≤ 2^990`, no
theory can contain every phase state on 1,000 or more qubits. -/
theorem palmer_ceiling {N L : ℕ} (hL : L ≤ 2 ^ 990) (S : Finset (PureState N))
    (hbudget : S.card ≤ 2 ^ (N * L))
    (hphase : ∀ s : NonZero N → Bool, phaseStateOf s ∈ S) : N ≤ 999 :=
  ceiling_1000 hL (budget_forces_ceiling S hbudget hphase)

/-- **Standard QM versus every linear budget.** For each capacity `L` there is a
register size at which no `N·L`-bit theory can contain all phase states. Standard
quantum mechanics contains them at every size, so it is incompatible with every
finite `L`. A phase state on that many qubits separates the two. -/
theorem no_linear_budget (L : ℕ) :
    ∃ N, ∀ S : Finset (PureState N), S.card ≤ 2 ^ (N * L) →
      ¬ ∀ s : NonZero N → Bool, phaseStateOf s ∈ S := by
  refine ⟨2 * L + 4, fun S hS hphase => ?_⟩
  have h := ceiling_log (budget_forces_ceiling S hS hphase)
  have hlt : (2 * L + 4) * L + 1 < 2 ^ (2 * L + 4) := by
    have key : ∀ n : ℕ, (2 * n + 4) * n + 1 < 2 ^ (2 * n + 4) := by
      intro n
      induction n with
      | zero => norm_num
      | succ n ih =>
        have : 2 ^ (2 * (n + 1) + 4) = 4 * 2 ^ (2 * n + 4) := by
          rw [show 2 * (n + 1) + 4 = (2 * n + 4) + 2 by ring, pow_add]; ring
        rw [this]; nlinarith
    exact key L
  have := Nat.log_lt_of_lt_pow (by positivity) hlt
  omega

end

end QuantumCeiling
