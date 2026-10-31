# Quantum ceiling: Palmer's qubit bound in Lean 4

Lean 4 formalization of the counting argument behind Tim Palmer's proposed
quantum-computing ceiling (T. N. Palmer, *Rational Quantum Mechanics: Testing
Quantum Theory with Quantum Computers*, arXiv:2510.02877, published in PNAS in 2026). No Mathlib needed and no `sorry`.
Only the standard axioms are used.

```bash
lake build
```

## The argument (Coffee-Law shape: budget vs. demand)

| | Grows like | Formalized as |
|---|---|---|
| Demand: real degrees of freedom of an `N`-qubit state | `2^(N+1) − 2` | `Palmer.dof` |
| Budget: bits in `N` entangled qubits under RaQM | `N · L` | `Palmer.capacity` |
| Ceiling: last `N` with demand ≤ budget | `N_max ≈ log₂ L` | `Palmer.Representable` |

## What is proved (`QuantumCeiling/Palmer.lean`)

- `worked_example`: the paper's example, `L = 16 ⇒ N_max = 5`.
- `fail_forever`: once a register size fails, every larger one fails, so the ceiling is permanent.
- `nmax_at_two_pow_640`: `L = 2^640` (≈ 10^193, the figure from Palmer's 2025 talk) gives `N_max = 648` exactly.
- `palmer_cap`: any `L ≤ 2^990` ⇒ at most 1,000 qubits.
- `no_ceiling_without_L`: for every `N` some `L` represents it, so the counting argument on its own forces no ceiling.
- `falsified_by`: a genuine general 1,001-qubit state implies `L > 2^990`, refuting every estimate in the paper.

`QuantumCeiling/Basic.lean` is a coarser, size-independent-budget model. It adds
`encoding_ceiling` (error correction cannot lift a fixed budget) and
`ceiling_independent` (both a capped world and standard QM are consistent).

## What is not proved

Lean checks the arithmetic, not the physics. That nature is described by RaQM,
and the value of `L` (which the paper estimates from Diósi–Penrose gravitational
collapse), are hypotheses. The paper itself says RaQM and standard QM are
experimentally indistinguishable for small numbers of qubits. Only an
experiment at hundreds of qubits can decide between them.
