/-
Copyright (c) 2026 Manish Bhatt. Released under the MIT license.
-/

/-!
# Palmer's qubit information capacity, formalized

This file formalizes the counting argument in T. N. Palmer, *Rational Quantum
Mechanics: Testing Quantum Theory with Quantum Computers*, arXiv:2510.02877
(v1 title: *Testing Quantum Mechanics with Quantum Computers: Qubit Information
Capacity*).

The argument has the same shape as the Coffee Law: a **budget** that grows
slowly against a **demand** that grows fast.

* **Demand.** A general `N`-qubit state in complex Hilbert space has
  `2^(N+1) - 2` real degrees of freedom (`2^N` complex amplitudes, less
  normalization and global phase).
* **Budget.** In Rational Quantum Mechanics (RaQM) each qubit carries `L` bits,
  so `N` entangled qubits carry `N * L` bits.
* **Ceiling.** Once `2^(N+1) - 2 > N * L` there are not enough bits to give each
  degree of freedom even one bit, and the state cannot be represented.

What is proved here: the arithmetic of that argument. The paper's worked example
(`L = 16` gives `N_max = 5`) checks out, the failure is permanent once it starts
(a true ceiling, not a dip), `L = 2^640` gives `N_max = 648`, and any
`L ≤ 2^990` caps the register at 1,000 qubits.

What is **not** proved here, and cannot be proved in Lean: that nature obeys
RaQM, or the value of `L`. The paper estimates `L` from Diósi–Penrose
gravitational collapse; that is physics, entered here only as a hypothesis.
`no_ceiling_without_L` shows the counting argument alone forces no ceiling.
-/

namespace QuantumCeiling.Palmer

-- The concrete checks evaluate powers of two up to `2^1002` (kernel GMP arithmetic).
set_option exponentiation.threshold 2048

/-- Real degrees of freedom of a general `N`-qubit state in complex Hilbert space. -/
def dof (N : Nat) : Nat := 2 ^ (N + 1) - 2

/-- RaQM information budget of `N` entangled qubits with `L` bits per qubit. -/
def capacity (L N : Nat) : Nat := N * L

/-- An `N`-qubit general state is representable when every degree of freedom
gets at least one bit. -/
def Representable (L N : Nat) : Prop := dof N ≤ capacity L N

instance (L N : Nat) : Decidable (Representable L N) :=
  inferInstanceAs (Decidable (_ ≤ _))

/-- **The paper's worked example.** With `L = 16`: 5 qubits need 62 bits and
have 80; 6 qubits need 126 and have only 96. So `N_max = 5`. -/
theorem worked_example : Representable 16 5 ∧ ¬ Representable 16 6 := by
  decide

/-- One step of the ceiling: once the budget falls short, adding a qubit makes
it fall shorter. Demand doubles, budget grows by `L`. -/
theorem fail_succ {L N : Nat} (hN : 1 ≤ N) (h : ¬ Representable L N) :
    ¬ Representable L (N + 1) := by
  unfold Representable dof capacity at *
  have hL : L ≤ N * L := Nat.le_mul_of_pos_left L hN
  have hs : (N + 1) * L = N * L + L := Nat.succ_mul N L
  have hp : 2 ^ (N + 1 + 1) = 2 * 2 ^ (N + 1) := by
    rw [Nat.pow_succ]; exact Nat.mul_comm _ _
  rw [hs, hp]
  generalize N * L = a at *
  generalize 2 ^ (N + 1) = b at *
  omega

/-- **The ceiling is permanent.** Once `N` qubits cannot be represented, no
larger register can be either. -/
theorem fail_forever {L N : Nat} (hN : 1 ≤ N) (h : ¬ Representable L N) :
    ∀ M, N ≤ M → ¬ Representable L M := by
  intro M hM
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hM
  induction d with
  | zero => exact h
  | succ d ih =>
    exact fail_succ (Nat.le_trans hN (Nat.le_add_right N d)) (ih (Nat.le_add_right N d))

/-- `N_max` from a single failure: if `N` fails, every representable register
is smaller than `N`. -/
theorem ceiling_of_fail {L N M : Nat} (hN : 1 ≤ N) (h : ¬ Representable L N)
    (hM : Representable L M) : M < N :=
  Nat.lt_of_not_le fun hle => fail_forever hN h M hle hM

/-- **`N_max ≈ log₂ L`.** The talk figure `L ≈ 10^193 ≈ 2^640` gives exactly
`N_max = 648`: the `log₂ L` term plus a small `log₂ N` correction. -/
theorem nmax_at_two_pow_640 :
    Representable (2 ^ 640) 648 ∧ ¬ Representable (2 ^ 640) 649 := by
  decide

theorem ceiling_at_two_pow_640 {N : Nat} (h : Representable (2 ^ 640) N) :
    N ≤ 648 :=
  Nat.le_of_lt_succ (ceiling_of_fail (by decide) nmax_at_two_pow_640.2 h)

/-- Representability can only get easier as `L` grows. -/
theorem representable_mono {L L' N : Nat} (hL : L ≤ L') (h : Representable L N) :
    Representable L' N :=
  Nat.le_trans h (Nat.mul_le_mul_left N hL)

/-- **The 1,000-qubit cap.** Any per-qubit capacity `L ≤ 2^990` (which covers
every estimate in the paper) rules out a general state on more than 1,000
qubits. -/
theorem palmer_cap {L N : Nat} (hL : L ≤ 2 ^ 990) (h : Representable L N) :
    N ≤ 1000 := by
  have h1001 : ¬ Representable (2 ^ 990) 1001 := by decide
  have : N < 1001 :=
    ceiling_of_fail (by decide) h1001 (representable_mono hL h)
  omega

/-! ## What the argument does not settle -/

/-- **No ceiling without a value of `L`.** For every register size there is a
budget large enough to represent it; standard quantum mechanics is the limit
`L → ∞`. The ceiling therefore rests entirely on the physical claim that `L` is
finite and small, which is outside logic. -/
theorem no_ceiling_without_L (N : Nat) : ∃ L, Representable L N := by
  refine ⟨2 ^ (N + 1), ?_⟩
  unfold Representable dof capacity
  cases N with
  | zero => decide
  | succ n =>
    have : 2 ^ (n + 1 + 1) ≤ (n + 1) * 2 ^ (n + 1 + 1) :=
      Nat.le_mul_of_pos_left _ (Nat.succ_pos n)
    omega

/-- **The deciding experiment.** A computation that genuinely uses a general
1,001-qubit state shows `L > 2^990`, refuting every estimate in the paper. -/
theorem falsified_by {L : Nat} (h : Representable L 1001) : 2 ^ 990 < L :=
  Nat.lt_of_not_le fun hL => absurd (palmer_cap hL h) (by decide)

end QuantumCeiling.Palmer
