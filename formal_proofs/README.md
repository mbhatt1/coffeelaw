# Formal proofs: collective information acquisition

Information-theoretic results on what a group of communicating agents (LLMs or
otherwise) can learn about an external truth `Θ`.

| File | Contents |
|---|---|
| `formal_proofs.md` | Lemmas and Theorems 1–7: conservation in closed systems, synergy, growth only through external channels, tools, routing, consensus vs. knowledge, and the **capacity budget** (Theorem 7): `I(Θ; record) ≤ min(H(Θ), I(Θ; D₀) + Σ_t C_t)` for any agents with tools, with a Fano lower bound on error. |
| `llm_specialization.md` | Which results hold for any agent, plus LLM-specific cost terms: miscalibration, lossy messages, persuasion. |
| `verify.py` | Exact checks of the finite examples in `formal_proofs.md`, including Theorem 7's tightness example (Scenario 7). |
| `verify_llm.py` | Exact checks of the examples in `llm_specialization.md`. |

Run:

```bash
python3 verify.py
python3 verify_llm.py
```

The "machine-checked" claims are exact numerical evaluations of small examples,
not proofs in a proof assistant. The general theorems are pen-and-paper proofs.
