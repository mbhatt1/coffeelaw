# Coffee Law

> **A collective of agents can never know more about the world than its
> starting data plus what its tools can carry.**

```
I(Θ ; everything the collective holds)  ≤  min( H(Θ),  I(Θ ; D₀) + Σ_t C_t )
```

`Θ` is the truth, `D₀` is the starting data and `C_t` is the Shannon capacity of
round `t`'s tool calls. The law holds for any kind of agent — LLMs, people, or
programs — any number of them, any communication protocol and any query
strategy, under four assumptions about the environment. Fano's inequality turns
it into a floor on error. Debate and voting add nothing to the budget; only
starting data and informative tool calls raise it.

Underneath sits a second ceiling, **reasoning**:
`usable knowledge ≤ available information ≤ budget`. More computation (thinking
longer, more agents, a code interpreter) raises what the collective can *use*,
up to what it *holds*, never past it
([Theorem 8](formal_proofs/formal_proofs.md#theorem-8)). Only total compute
counts, and the compute needed ranges from linear to infeasible depending on the
task ([Theorems 9–10](formal_proofs/formal_proofs.md#theorem-9)).

**Read it:** <https://mbhatt1.github.io/coffeelaw/>

- [Blog post: Agent collectives have bounds](formal_proofs/agent-collectives-have-bounds.md)
- [Formal proofs](formal_proofs/formal_proofs.md) — the Coffee Law is
  [Theorem 7](formal_proofs/formal_proofs.md#theorem-7); the reasoning ceiling is
  [Theorem 8](formal_proofs/formal_proofs.md#theorem-8)
- [Paper (IEEE format, PDF)](paper/coffee_law.pdf) — see [`paper/README.md`](paper/README.md) for the target venue
- [Extensions](formal_proofs/extensions.md) — sandboxes, acting, communication,
  a changing world, adversaries, continuous truths, what a model's weights
  know, and what remains open
- [Information geometry](formal_proofs/proof.md)
- [From any agent to LLMs](formal_proofs/llm_specialization.md)

**Check it:**

```bash
cd formal_proofs
python3 verify.py && python3 verify_llm.py && python3 verify_geometry.py
python3 verify_extensions.py   # needs numpy
```

The checks are exact numerical evaluations of the worked examples plus
randomized tests of the general inequalities, not proof-assistant proofs. See
[`formal_proofs/README.md`](formal_proofs/README.md) for details.

[MIT License](LICENSE)
