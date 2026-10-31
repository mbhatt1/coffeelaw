# Quantum ceiling: Palmer's qubit bound, derived in Lean 4 + Mathlib

Tim Palmer (*Rational Quantum Mechanics: Testing Quantum Theory with Quantum
Computers*, arXiv:2510.02877; PNAS 2026) proposes that `N` entangled qubits
carry only `N·L` bits, for a finite per-qubit capacity `L` set by gravity, so
exponential quantum speedups saturate near 1,000 qubits.

This project **derives** the resulting ceiling from the structure of quantum
state space, instead of assuming the counting inequality.

```bash
lake exe cache get   # or build Mathlib from source
lake build
```

## Setting

- Pure `N`-qubit states are rays in `ℂ^(2^N)`, Mathlib's `Projectivization ℂ (Basis N → ℂ)`.
- Phase states `Σₓ (-1)^f(x) |x⟩` are the oracle states of Deutsch–Jozsa,
  Bernstein–Vazirani and Grover. Every amplitude has squared modulus `1/2^N`
  and phase `0` or `π`, so they satisfy RaQM's rationality constraints.

## Theorems (`QuantumCeiling/PhaseStates.lean`)

| Theorem | Statement |
|---|---|
| `phaseState_inj`, `phaseState_injective` | distinct sign patterns (global phase fixed) are physically distinct states |
| `card_phaseStates` | there are `2^(2^N − 1)` of them |
| `budget_forces_ceiling` | **main:** a state space with `≤ 2^(N·L)` states that contains every phase state forces `2^N − 1 ≤ N·L` |
| `ceiling_permanent` | once infeasible, infeasible for every larger `N` |
| `ceiling_log` | `N ≤ log₂(N·L + 1)`, i.e. `N_max ≈ log₂ L` |
| `ceiling_1000`, `palmer_ceiling` | `L ≤ 2^990` (every estimate in the paper) ⇒ `N ≤ 999` |
| `no_linear_budget` | for every finite `L`, some register size has no `N·L`-bit theory containing all phase states, so Palmer's premise and standard QM are incompatible |

Our bound `2^N − 1 ≤ N·L` is half of Palmer's `2^(N+1) − 2`. He charges a bit
per real degree of freedom; phase states only use one sign per basis state. The
result is weaker, but it is exact: no approximation argument is needed.

## What is not proved

Lean checks mathematics, not nature. Two hypotheses carry all the physics:
the `N·L`-bit budget, which is Palmer's premise, and the value of `L`, which he
estimates from Diósi–Penrose gravitational collapse. The theorems show the
ceiling follows from the budget, and that a computation which genuinely uses
phase states on more than about `log₂ L` qubits refutes it.
