# Coffee Law

> **Coffee Law.** A collective of agents can never know more about the world
> than its starting data plus what its tools can carry:
>
>     I(Θ ; everything the collective holds)  ≤  min( H(Θ),  I(Θ ; D₀) + Σ_t C_t )
>
> where `Θ` is the truth, `D₀` is the starting data and `C_t` is the Shannon
> capacity of round `t`'s tool calls ([Theorem 7](formal_proofs.md#theorem-7)).

The law is universal in one precise sense: it holds for any kind of agent —
LLMs, people, or programs — any number of them, any protocol and any query
strategy, under [four assumptions](agent-collectives-have-bounds.md#assumptions)
about the environment. These notes prove it in three steps:

1. **Talking cannot create information.** A closed group that only communicates
   and computes knows at most what its starting data already contained
   ([Theorem 1](formal_proofs.md#theorem-1)).
2. **Pooling can unlock information no member could use alone.** Two agents with
   zero bits each can jointly determine a full bit
   ([Theorem 2](formal_proofs.md#theorem-2)).
3. **Only the world adds information, and only at the rate its channels allow.**
   Summing tool-channel capacities over rounds gives the Coffee Law, and Fano's
   inequality turns it into a floor on error
   ([Theorem 7](formal_proofs.md#theorem-7)). Claim 1 is the Coffee Law with no
   tools.

A second ceiling sits underneath: **reasoning**. What agents can *use* is at
most what they *hold*, and only computation closes the gap
([Theorem 8](formal_proofs.md#theorem-8)):

        usable knowledge  ≤  available information  ≤  starting data + tool capacity

Debate and more agents can raise usable knowledge, never available information.

A third ceiling is **communication**: when data is split across agents, no
single agent can use it until enough has been shared, about half the data for
some tasks however capable the agents are
([Theorem 14](extensions.md#theorem-14)).
How much computation that takes is bounded too: only total compute counts, and
depending on the task the cost ranges from reading every input to more than any
efficient computation can supply ([Theorems 9–10](formal_proofs.md#theorem-9)).

None of this inspects the agents, so it applies to any learner. It bounds what
can be *learned* about a fixed truth and what can be *used* of it, not how much
computation the using takes, and not outcomes the agents themselves control.

**New here? Start with the blog post:
[Agent collectives have bounds](agent-collectives-have-bounds.md).**
For the short formal version, read the
[paper (PDF)](https://github.com/mbhatt1/coffeelaw/blob/main/paper/coffee_law.pdf).

## Documents

| Document | What it covers |
|---|---|
| [Formal proofs](formal_proofs.md) | [Definitions](formal_proofs.md#definitions) and [standard lemmas](formal_proofs.md#lemmas); [conservation](formal_proofs.md#theorem-1); [synergy](formal_proofs.md#theorem-2); [growth](formal_proofs.md#theorem-3); [tools](formal_proofs.md#theorem-4); [routing as value of information](formal_proofs.md#theorem-5); [consensus is not knowledge](formal_proofs.md#theorem-6); [the Coffee Law](formal_proofs.md#theorem-7); [the reasoning ceiling](formal_proofs.md#theorem-8); [computational bounds](formal_proofs.md#theorem-9). |
| [Information geometry](proof.md) | The same results as the geometry of negative entropy: [KL as a Bregman divergence](proof.md#g1), [information gain as value of information](proof.md#g2), [the Fisher metric](proof.md#g3), [evidence vs. consensus flows](proof.md#g4), and [what is not proven](proof.md#limits). |
| [Extensions](extensions.md) | Closing the gaps, plus [where to spend](extensions.md#prop-19) and [measuring the quantities](extensions.md#measuring): [exact accounting and when the law is tight](extensions.md#theorem-7-prime), [reasoning that opens channels (sandboxes)](extensions.md#theorem-11), [a worst-case version](extensions.md#theorem-12), [acting](extensions.md#theorem-13), [communication between agents](extensions.md#theorem-14), [a changing world](extensions.md#theorem-15), [adversaries](extensions.md#theorem-16), [continuous quantities](extensions.md#theorem-17), [what a model's weights know](extensions.md#prop-18), [tool capacity from the output format](extensions.md#theorem-20), and [what remains open](extensions.md#open). |
| [From any agent to LLMs](llm_specialization.md) | What holds for [any agent](llm_specialization.md#part-a), then LLM-specific costs: [the regret bound](llm_specialization.md#b0), [miscalibration](llm_specialization.md#b1), [lossy messages](llm_specialization.md#b2), [persuasion](llm_specialization.md#b3), and [the combined per-round bound](llm_specialization.md#b4). |

Suggested reading order: the three numbered claims above, then
[Theorem 7](formal_proofs.md#theorem-7), then the
[LLM specialization](llm_specialization.md) for what changes when the agents
are imperfect.

## Checks

Every worked example is computed exactly from its full joint distribution, and
the general inequalities are tested on randomized cases.

| Script | Checks |
|---|---|
| [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py) | Every example in the formal proofs, including the tightness of [Theorem 7](formal_proofs.md#theorem-7) |
| [`verify_llm.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_llm.py) | The LLM-specialization examples and the regret bounds |
| [`verify_geometry.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_geometry.py) | Lemmas G1–G5 of the geometry notes |
| [`verify_extensions.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_extensions.py) | Theorems 11–17 and Proposition 18 of the extensions |
| [`verify_capacity.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_capacity.py) | [Theorem 20](extensions.md#theorem-20): tool capacity from byte/token limits and response schemas |
| [`lean/`](https://github.com/mbhatt1/coffeelaw/tree/main/formal_proofs/lean) | Lean 4 proofs of the Markov condition `W — (D₀, 𝒯) — A` for any collective, data processing, and the output-entropy bounds ([details](formal_proofs.md#lean)) |

```bash
git clone https://github.com/mbhatt1/coffeelaw.git
cd coffeelaw/formal_proofs
python3 verify.py && python3 verify_llm.py && python3 verify_geometry.py
python3 verify_extensions.py   # needs numpy
python3 verify_capacity.py
```

The scripts are numerical checks. The core of the Coffee Law (the Markov
condition for any collective, data processing, and the per-call output-entropy
bound) is also proved in Lean 4; the other general theorems are pen-and-paper
proofs. CI runs all five scripts and builds the Lean proofs before every
publish of this site.

## Related work

The closed-group bound for multi-agent LLM debate is published:
Shin, *The Reasoning Trap* ([arXiv:2605.01704](https://arxiv.org/abs/2605.01704));
Choi et al., *Debate or Vote* ([arXiv:2508.17536](https://arxiv.org/abs/2508.17536)).
Yang et al., *Understanding Agent Scaling in LLM-Based Multi-Agent Systems via
Diversity* ([arXiv:2602.03794](https://arxiv.org/abs/2602.03794)) gives a bound
of the same "min(H, Σ channels)" shape with agents as the channels. The
reasoning ceiling is usable ("V-") information in the sense of Xu et al.
([arXiv:2002.10689](https://arxiv.org/abs/2002.10689)). Published experiments match both regimes: extra test-time compute
doesn't reliably help on closed-book knowledge tasks ([arXiv:2509.06861](https://arxiv.org/abs/2509.06861)), debate helps on
math and strategy (Du et al., [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)), and debate does about as well as
self-consistency or voting at equal effort (Smit et al., [PMLR 235](https://proceedings.mlr.press/v235/smit24a.html)). The
mathematical ingredients — the chain rule, the data-processing inequality,
channel capacity and Fano's inequality — are textbook (Cover & Thomas,
*Elements of Information Theory*). See the related-work notes under [Theorem 7](formal_proofs.md#theorem-7)
and [Theorem 8](formal_proofs.md#theorem-8) for the comparison and its caveats.

[Source on GitHub](https://github.com/mbhatt1/coffeelaw/tree/main/formal_proofs)
