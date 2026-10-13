# Agent collectives have bounds

*Why a room full of agents can't talk its way to new knowledge, and exactly how
much its tools can buy.*

---

Multi-agent systems are often pitched with a quiet promise: put enough agents
in a room, let them debate, critique and vote, and something smarter than any
one of them will emerge. Sometimes it does. But there is a hard ceiling on what
that emergence can achieve, and it can be stated in one line:

> **A collective can never know more about the world than its starting data
> plus what its tools can carry.**

We call this the **Coffee Law**.

That isn't a hunch or an empirical trend. It's a theorem, and it holds for any
kind of agent — LLMs, people, or programs — under four assumptions, listed
[below](#assumptions). This post explains what it says, why
it's true, and what it does and doesn't rule out. The full proofs and the
scripts that check them are in [the formal proofs](formal_proofs.md).

## Talking moves information around; it doesn't create it

Start with a group of agents that can only talk to each other and think. No
web search, no experiments, no new data after the conversation starts.

Everything such a group will ever say is a function of two things: what it
knew at the start, and randomness (sampling noise, who speaks first) that has
nothing to do with the truth. The data-processing inequality, one of the most
basic facts in information theory, then says:

> the information the whole conversation holds about the truth is at most the
> information the starting data held.

So debate, self-reflection, critique rounds and majority votes cannot add a
single bit about the world. They can only re-express what was already there
([Theorem 1](formal_proofs.md#theorem-1)).

## But emergence is real

That doesn't make collectives useless. Information can be present *jointly*
while no individual can use it.

Here is the cleanest example. Two agents each hold one random bit, `B₁` and
`B₂`. The question is whether the bits are equal (formally, `B₁ ⊕ B₂`). Alone,
each agent knows **exactly nothing** about the answer: whatever its bit is, the
answer is still a coin flip. Together, they know it **with certainty**.

Zero plus zero equals one bit. That is genuine emergence, and it's the strongest
argument for shared boards and agent communication
([Theorem 2](formal_proofs.md#theorem-2)).

But notice where that bit came from: it was in the starting data all along,
split across two agents. Communication made it *accessible*. It didn't make it
*exist*. Emergence rearranges the budget; it doesn't enlarge it.

## Only the world adds information

The moment an agent calls a tool — runs a search, executes code against real
data, queries a sensor, runs an experiment — something different happens. The
tool's answer depends on the actual state of the world, not just on what the
agents already believed. That breaks the chain that capped the closed group,
and total information can grow
([Theorem 3](formal_proofs.md#theorem-3)).

Choosing *which* tool call to make adds nothing by itself: the choice is made
from what the group already knows. All new information arrives in the
*results* ([Theorem 4](formal_proofs.md#theorem-4)).

## How much can tools add? The capacity budget

Every tool call is a communication channel from the world to the collective,
and every channel has a maximum rate — its Shannon capacity. A call that returns
one of 16 possible answers can carry at most 4 bits. A noisy call carries less.

Put the pieces together and you get the Coffee Law
([Theorem 7](formal_proofs.md#theorem-7)):

    information about the truth  ≤  min( total uncertainty,
                                         starting information + Σ capacity of each round's tool calls )

This holds for:

- any number of agents,
- any communication protocol (debate, voting, hierarchies, shared boards),
- any internal reasoning,
- any adaptive strategy for picking the next query.

The number of agents doesn't even appear in the bound. More agents help only if
they make more tool calls, or better ones.

And because information limits accuracy, the bound becomes a floor on error.
Fano's inequality says the chance of a wrong answer is at least roughly
`(remaining uncertainty − 1) / log₂(number of possible answers)`. For a question
with `2^k` equally likely answers, getting the error down to `ε` takes at least
`((1 − ε)·k − 1) / C` rounds of tool use, where `C` is the capacity per round —
however clever the agents are.

## A worked example

Suppose the truth is one of four values (2 bits), and each tool call returns
one bit of it, flipped by noise 10% of the time. That gives a capacity of about
0.531 bits per call. With two calls, the budget is 1.062 bits.
[`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py)
computes the exact information gained:

| Strategy | Information gained |
|---|---|
| Ask for bit 0, then bit 1 | 1.062 bits — exactly the budget |
| "Adaptive" strategy that sometimes re-asks for the same bit | 0.982 bits — below the budget |
| Any amount of debate afterwards | +0 bits |
| Ask for bit 0, then bit 1, with a noise-free tool | 2 bits — the whole truth |

Here the bound is tight: a good strategy reaches it exactly. (That isn't
guaranteed in general; sometimes no strategy can use a tool at full capacity.)
A clever-looking strategy can waste the budget. Nothing exceeds it.

## Why real agents fall short of the bound

The budget is a ceiling for every agent. Ideal reasoners can reach it; real
agents, and LLMs in particular, fall short of it by losing information along the
way. Three losses are worth naming
([LLM specialization](llm_specialization.md)):

- **Miscalibration.** An agent that updates its beliefs too little, or too
  confidently, loses exactly the KL divergence between the correct beliefs and
  its own, measured in log-loss. In decision terms the proved cost is at most
  proportional to the square root of that KL
  ([B1](llm_specialization.md#b1), [B0](llm_specialization.md#b0)).
- **Lossy messages.** When one agent summarizes for another, the information
  lost is exactly `I(Θ;data) − I(Θ;message)`
  ([B2](llm_specialization.md#b2)).
- **Persuasion.** A confident message that carries *zero* information about the
  truth can still flip another agent's answer from right to wrong. In the
  example in B3, an agent with a 60/40 belief in the right answer flips once it
  gives a constant, uninformative claim more than 1/6 of the weight
  ([B3](llm_specialization.md#b3)).

These combine into one upper bound on how far a collective's decisions can
fall short ([B4](llm_specialization.md#b4)). It also explains a familiar failure:
agreement grows while knowledge doesn't. Perfect consensus on a random claim
carries zero information about the truth
([Theorem 6](formal_proofs.md#theorem-6)).

<a id="assumptions"></a>

## The four assumptions

The bound needs exactly these, and nothing about the agents themselves
([Theorem 7](formal_proofs.md#theorem-7)):

1. **No hidden channels.** Information about the truth reaches the collective
   only through its starting data and its tool results.
2. **Truth-independent randomness.** Sampling noise, scheduling and coin flips
   don't secretly depend on the answer.
3. **Tools are channels from the truth.** Given the query, a tool's output
   depends on the world, not on the agents' private state. (Tools may be
   adaptive or stateful.)
4. **The truth is fixed.** The agents' actions don't change the answer.

## What the bound does *not* say

It's tempting to stretch a result like this. Here is where it stops.

- **It doesn't cap intelligence.** It caps *information about a fixed external
  truth*. Working out the consequences of what you already know — proving
  theorems, writing code, planning — is computation, and this bound says
  nothing about it.
- **It isn't a fixed ceiling.** The budget grows with every informative tool
  call. A collective with unlimited access to informative tools is limited only
  by how much uncertainty there was to resolve. (A tool whose output doesn't
  depend on the truth adds nothing, however often it's called.)
- **It doesn't separate AI from people.** The proof never looks inside the
  agents. A lab full of scientists obeys exactly the same bound. It's a law of
  learning, not a limit specific to machines — so it can't be used to argue
  that AGI is impossible.
- **It doesn't cover outcomes the agents control.** If the agents can *make*
  something true, their success at predicting it isn't limited by what they
  can learn.
- **It assumes no hidden channels.** Anything a model absorbed in training
  that correlates with the answer counts as starting data. Forget a channel and
  the accounting is wrong.

## What to do with this

If you're building agent systems, the bound turns into practical advice:

1. **Don't expect debate to add facts.** Use it to surface and combine what the
   agents already know — that's where its value is.
2. **Spend effort on the inputs: tools and starting data.** More informative
   calls, less noisy tools and better choices of what to query raise the
   ceiling during the run; better starting data (a better-trained model, more
   relevant context) raises it before the run. Nothing else does.
3. **Route, don't just talk.** The board's biggest contribution with tools is
   telling the agent with the tool *which* query is informative — worth exactly
   the value of that information ([Theorem 5](formal_proofs.md#theorem-5)).
4. **Guard against persuasion.** Confidence is not evidence. Weight messages by
   what they reveal about the world, not by how certain they sound.
5. **Check knowledge against the world.** Agreement is cheap. Verification,
   experiments and held-out tests are what certify that a collective actually
   knows something.

## Is this new?

Mostly no, and that's a strength. The ingredients — the chain rule, the
data-processing inequality, channel capacity and Fano's inequality — are
textbook information theory. The closed-group case has been written up for
LLM debate by Shin ([arXiv:2605.01704](https://arxiv.org/abs/2605.01704)) and
Choi et al. ([arXiv:2508.17536](https://arxiv.org/abs/2508.17536)), and Yang et
al. ([arXiv:2602.03794](https://arxiv.org/abs/2602.03794)) give a bound of the
same shape with agents as the channels. What these notes add, as far as we found, is
the explicit accounting for tools: per-call capacity, adaptive queries, and an
error floor.
(That comparison is based on the papers' summaries; their exact statements
are worth reading in full.)

The short version, and the Coffee Law in one sentence: **a collective's
intelligence can be emergent, but its knowledge has a budget, and only the
world can raise it.**

---

*Proofs: [formal proofs](formal_proofs.md) ·
[information geometry](proof.md) ·
[LLM specialization](llm_specialization.md).
Checks: [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py),
[`verify_llm.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_llm.py),
[`verify_geometry.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_geometry.py).*
