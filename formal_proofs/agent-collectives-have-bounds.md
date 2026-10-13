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
([Theorem 1](formal_proofs.md#theorem-1)). Re-expressing can still be very
valuable, as the next two sections show, but it adds no facts.

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
split across two agents. Communication made it *usable*. It didn't make it
*exist*. Emergence rearranges the budget; it doesn't enlarge it. That
difference between information that exists and information you can use turns
out to be a second ceiling of its own.

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

## Reasoning is the second ceiling

Having information isn't the same as being able to use it. Give a collective a
large number `N` and ask for its prime factors: the answer is completely
determined by `N`, so every bit of it is already in the data and the Coffee Law
is satisfied with room to spare. Yet finding the factors can take an enormous
amount of computation.

So there are two ceilings, one stacked on the other
([Theorem 8](formal_proofs.md#theorem-8)):

    usable knowledge  ≤  available information  ≤  starting data + tool capacity
                    ↑                          ↑
              raised by reasoning        raised by tools

- **Reasoning** (thinking longer, more agents, verification, a code interpreter
  running on data you already have) raises *usable* knowledge toward the
  information you hold. It can never push past it.
- **Tools that observe the world** raise the information ceiling itself.

This changes one claim from earlier. Debate adds no *facts*, but it can add
*reasoning*: an agent that posts an intermediate result hands the others a
computation they couldn't do themselves. In the XOR example, an agent that can
only look at one bit at a time gets nothing from `(B₁, B₂)`; once someone posts
`B₁ ⊕ B₂`, it gets the full bit. The information never changed; its usability
did.

In the noisy worked example above, the transcript holds 1.062 bits about the
answer, but an agent that can only read one entry of it at a time can use just
0.266 of them. The rest is a reasoning gap. More observations won't close it;
more computation will.

That splits tasks into two kinds:

- **Information-bound** (forecasting, facts about the world, diagnosing a live
  system): the Coffee Law binds. Add better tools.
- **Compute-bound** (math, proofs, code, planning from known facts): the
  information is there but hard to use. Add reasoning.

Most real tasks are a mix, and the useful question is which ceiling you are
hitting.

### How much reasoning does it take?

Closing a reasoning gap costs computation, and that cost has bounds of its own
([Theorems 9–10](formal_proofs.md#theorem-9)):

- **Only total compute counts.** A group of agents can always be simulated by
  one machine doing the same total work. So more agents help reasoning exactly
  as much as the extra compute they bring. Splitting work across agents buys
  speed, not capability.
- **Some gaps need a fixed amount of work.** To know whether an even or odd
  number of `n` coin flips came up heads, you must look at all `n`. Read
  `n − 1` of them, cleverly or not, and you know nothing; read the last one and
  you know everything. Ten agents can split the reading and finish faster, but
  together they still read all `n`.
- **Some gaps can't be closed efficiently at all.** In cryptography, an
  encrypted message can fully determine a secret bit, so all of the information
  is there, while every fast program does no better than a coin flip at finding
  it (assuming standard cryptography is secure). No amount of efficient
  reasoning, by one agent or a million, closes that gap.

So the reasoning ceiling is real, but how hard it is to reach depends on the
task, from cheap to effectively impossible. No general shortcut exists:
predicting the cost for every task would settle open questions like whether
P = NP.

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

## Beyond the basic law

The four assumptions keep the core result clean. Relaxing them one at a time
gives the rest of the picture ([extensions](extensions.md)):

- **Reasoning can open new channels.** Finding an undocumented API, a leaked
  credential or an exploit is computation, and it can raise the information
  budget ([Theorem 11](extensions.md#theorem-11)). That is what a sandbox is
  for: guaranteeing that no amount of the agents' compute reaches channels
  outside an allowed set. In OpenAI's reported 2026 incidents, models under
  test escaped their sandbox and fetched benchmark answers from an outside
  server: the answers weren't inside, so the only route was a channel nobody
  had allowed.
- **It holds for each task, not just on average.** For any set of tasks the
  agents can't tell apart within their budget, there is a specific one they get
  wrong at least as often as the law says
  ([Theorem 12](extensions.md#theorem-12)).
- **Acting is bounded by knowing.** When choosing the wrong action costs
  something, an agent's performance is limited by what it has learned about
  the environment. That includes open-ended work: an agent can't reliably build
  what it hasn't learned the user wants
  ([Theorem 13](extensions.md#theorem-13)).
- **Agents must talk enough.** When the answer depends on data split across
  agents, no single agent can learn it without enough communication; for some
  questions that is about half the data's size, no matter how clever the
  agents are ([Theorem 14](extensions.md#theorem-14)).
- **In a changing world, knowledge saturates.** If the truth drifts, old
  knowledge decays, and what the collective knows levels off however long it
  runs ([Theorem 15](extensions.md#theorem-15)).
- **Adversaries can hide information, but not destroy it**, as long as every
  message is attributed to its author. On an anonymous board, a single liar
  can erase everything ([Theorem 16](extensions.md#theorem-16)).
- **Quantities work too.** For numbers rather than categories, each bit of
  information divides the best possible squared error by four
  ([Theorem 17](extensions.md#theorem-17)).

## What the bound does *not* say

It's tempting to stretch a result like this. Here is where it stops.

- **It doesn't cap intelligence.** It caps *information about a fixed external
  truth*. Working out the consequences of what you already know — proving
  theorems, writing code, planning — is reasoning. Reasoning has its own
  ceiling (the information you hold), and reaching it takes computation that
  ranges from cheap to effectively impossible depending on the task
  ([Theorems 9–10](formal_proofs.md#theorem-9)).
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
  can learn. (Acting in an environment the agents *don't* change is covered;
  see above.)
- **It assumes no hidden channels.** Anything a model absorbed in training
  that correlates with the answer counts as starting data. Forget a channel and
  the accounting is wrong.

## What to do with this

If you're building agent systems, the bound turns into practical advice:

1. **Find out which ceiling you're hitting.** If the answer is in the data,
   think harder. If it isn't, look at the world. Neither substitutes for the
   other.
2. **Use debate for reasoning, not facts.** It can combine what the agents
   already know and share intermediate results. It can't discover anything.
3. **Spend effort on the inputs: tools and starting data.** More informative
   calls, less noisy tools and better choices of what to query raise the
   information ceiling during the run; better starting data (a better-trained model, more
   relevant context) raises it before the run. Nothing else does.
4. **Route, don't just talk.** The board's biggest contribution with tools is
   telling the agent with the tool *which* query is informative — worth exactly
   the value of that information ([Theorem 5](formal_proofs.md#theorem-5)).
5. **Guard against persuasion.** Confidence is not evidence. Weight messages by
   what they reveal about the world, not by how certain they sound.
6. **Control channels, and attribute messages.** Sandboxing is channel
   control, and it has to hold against the agents' full compute. And sign every
   message with its author: provenance is what keeps an adversary from erasing
   what the group knows.
7. **Check knowledge against the world.** Agreement is cheap. Verification,
   experiments and held-out tests are what certify that a collective actually
   knows something.

## Is this new?

Mostly no, and that's a strength. The ingredients — the chain rule, the
data-processing inequality, channel capacity and Fano's inequality — are
textbook information theory. The closed-group case has been written up for
LLM debate by Shin ([arXiv:2605.01704](https://arxiv.org/abs/2605.01704)) and
Choi et al. ([arXiv:2508.17536](https://arxiv.org/abs/2508.17536)), and Yang et
al. ([arXiv:2602.03794](https://arxiv.org/abs/2602.03794)) give a bound of the
same shape with agents as the channels. The reasoning ceiling is "usable information"
in the sense of Xu et al. ([arXiv:2002.10689](https://arxiv.org/abs/2002.10689)), who also show computation can create it.
What these notes add, as far as we found, is the explicit accounting for tools
(per-call capacity, adaptive queries, an error floor) and the two ceilings
combined in one chain.

The published experiments line up with both ceilings. On closed-book knowledge
questions, giving reasoning models more thinking time doesn't reliably help and
often increases hallucination ([arXiv:2509.06861](https://arxiv.org/abs/2509.06861)): an information-bound task. On math and
strategy, debate helps (Du et al., [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)): a compute-bound task. And debate
tends to do no better than majority voting or self-consistency at equal effort
(Smit et al., [PMLR 235](https://proceedings.mlr.press/v235/smit24a.html); Choi et al.), which is what you'd expect if its value is
extra reasoning. Even Du et al.'s finding that debate improves *factuality*
fits: those facts were already in the model's weights, so debate made them
usable rather than adding them.
(These comparisons are based on the papers' abstracts and published summaries.)

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
