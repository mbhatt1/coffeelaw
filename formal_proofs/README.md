# Coffee Law: formal proofs

**Read online:** <https://mbhatt1.github.io/coffeelaw/> · Blog post: [Agent collectives have bounds](agent-collectives-have-bounds.md)

Information-theoretic results on what a group of communicating agents (LLMs or
otherwise) can learn about an external truth `Θ`. The central result,
Theorem 7, is the **Coffee Law**: a collective can never know more about the
world than its starting data plus what its tools can carry.

| File | Contents |
|---|---|
| [`formal_proofs.md`](formal_proofs.md) | Lemmas and Theorems 1–10: conservation in closed systems, synergy, growth only through external channels, tools, routing, consensus vs. knowledge, and the **capacity budget** (Theorem 7): `I(Θ; record) ≤ min(H(Θ), I(Θ; D₀) + Σ_t C_t)` for any agents with tools, with a Fano lower bound on error; the **reasoning ceiling** (Theorem 8): usable ≤ available ≤ budget; and **computational bounds** on closing the reasoning gap (Theorems 9–10). |
| [`proof.md`](proof.md) | The same results as the geometry of one convex function (negative entropy): KL as its Bregman divergence, information gain as value of information under log loss, the Fisher metric, and evidence vs. consensus flows (Lemmas G1–G5). |
| [`extensions.md`](extensions.md) | Theorems 7′, 11–17 and 20, closing the gaps: exact accounting, compute-gated channels (sandboxes), the worst-case law, acting, communication between agents, a changing world, adversaries and provenance, continuous truths; Proposition 18 on what a model's weights know (possible worlds); Proposition 19 on where to spend; Theorem 20 on tool capacity from the output format (byte/token limits, response schemas); measuring the quantities; plus what remains open. |
| [`llm_specialization.md`](llm_specialization.md) | Which results hold for any agent, plus LLM-specific costs (miscalibration, lossy messages, persuasion), combined in a proved regret bound (Lemma B0, Theorem B4). |
| [`verify.py`](verify.py) | Exact checks of the finite examples in `formal_proofs.md`, including Theorem 7's tightness example (Scenario 7) Theorem 8's reasoning-gap examples (Scenario 8), and the parity compute bounds (Scenario 9). |
| [`verify_llm.py`](verify_llm.py) | Checks of the examples and bounds in `llm_specialization.md`. |
| [`verify_geometry.py`](verify_geometry.py) | Checks of Lemmas G1–G5 in `proof.md`. |
| [`verify_extensions.py`](verify_extensions.py) | Checks of Theorems 11–17 and Proposition 18 in `extensions.md` (needs numpy). |
| [`verify_capacity.py`](verify_capacity.py) | Checks of Theorem 20 in `extensions.md`: exact counts of result spaces for byte, token and schema formats, and the entropy bounds on random channels. |
| [`lean/`](lean/) | Lean 4 proofs of the baseline invariant `W — (D₀, 𝒯) — A` for any collective, the data-processing inequality and the output-entropy bounds. See [`lean/README.md`](lean/README.md). |

Run:

```bash
python3 verify.py
python3 verify_llm.py
python3 verify_geometry.py
python3 verify_extensions.py   # needs numpy
python3 verify_capacity.py
cd lean && lake exe cache get && lake build   # Lean 4 + Mathlib
```

The site is built from this directory by `.github/workflows/pages.yml`, which
runs all five scripts and builds the Lean proofs first, and publishes only if
both pass. `index.md` is the
home page.

The "machine-checked" claims in the Python scripts are numerical evaluations of
small examples (exact rational distributions where possible, plus randomized
tests of the general inequalities). The Lean proofs in `lean/` are proofs in a
proof assistant, for the core of Theorem 7 only (the Markov condition, data
processing and the output-entropy bound). The other general theorems are
pen-and-paper proofs.
