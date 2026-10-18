# Lean 4 proofs: the baseline invariant

Machine-checked proofs of the step of the Coffee Law that does not depend on
the agents:

        W  —  (D₀, 𝒯)  —  A

(the world `W` and anything the collective outputs `A` are conditionally
independent given the starting data `D₀` and the full tool transcript `𝒯`),
and of what follows from it: the data-processing inequality and the
output-entropy bounds on tool calls.

The formalization is non-parametric. A law is any nonnegative finite table that
sums to one; agents, query policies, tools and output maps are arbitrary
functions of the right arguments. Nothing refers to how an agent is built, so
the results hold for LLMs, people and programs alike.

| File | Contents |
|---|---|
| `CoffeeLaw/Basic.lean` | Gibbs' inequality and the log-sum inequality, from `log x ≤ x − 1` |
| `CoffeeLaw/Markov.lean` | The Markov condition (division-free), `isMarkov_of_factor`, `interactive_isMarkov`, `adaptive_isMarkov` (`T` rounds of adaptive tool use), `isMarkov_iff_kernel`, and agents as kernels (`Kernel.ofFun`, `Kernel.seq`, `Kernel.comp`) |
| `CoffeeLaw/Information.lean` | Mutual information; `dpi`, `dpi_of_isMarkov`, `collective_dpi` (`I(W; A) ≤ I(W; D₀, 𝒯)`); `mutualInfo_le_log_card` (`I(W; O) ≤ log \|𝒪\|`); `condEntropy_le_log_card` (`H(O \| D) ≤ log \|𝒪\|`); `mutualInfo_append_le` (`I(W; D, O) ≤ I(W; D) + log \|𝒪\|`) |

Mutual information is in nats (`Real.log`); every inequality is homogeneous in
`log`, so it holds in bits too.

**Assumptions as they appear in Lean.** In `interactive_isMarkov` the joint
weight of world `w`, data `d`, internal randomness `r`, transcript `τ` and output
`a` is `μ(w, d) · ρ(r) · σ(d, r, τ) · e(w, d, τ) · φ(d, r, τ, a)`. The product
`μ(w, d) · ρ(r)` is (A2): internal randomness is drawn independently of the
world. The tools' factor `e` does not take `r`, which is (A3): tools see the
agents only through their queries. (A1) is that `W` enters only through `μ` and
`e`. Causality (queries depend only on earlier results) is what makes the table
a probability law; the Markov property holds without it.

**Not covered.** The capacity form of the law (`C_t` as a maximum over input
laws), Fano's inequality, and the other theorems are pen-and-paper proofs
checked numerically by the Python scripts in the parent directory.

## Build

```bash
lake exe cache get   # download prebuilt Mathlib
lake build
```

CI (`.github/workflows/pages.yml`) builds with `--wfail`, so any `sorry` or
warning fails the build.
