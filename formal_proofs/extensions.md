# Extensions: completing the theory

[Theorems 1–10](formal_proofs.md) establish the Coffee Law, the reasoning
ceiling and computational bounds under four assumptions (A1–A4) and a fixed,
finite, average-case setting. This document closes the gaps those assumptions
leave, one theorem per gap, and states plainly what remains open.

Notation and setup follow [Theorem 7](formal_proofs.md#theorem-7): `Θ` is the
truth, `D_t` the collective's full record, `Q_t = g_t(D_t, ξ_t)` the round's tool
calls, `O_{t+1} ∼ K_t(· | Θ, Q_t, D_t)` the results, `S_t = (D_t, ξ_t, Q_t)`, and
`C_t` the round's capacity. Logs are base 2. Every finite claim is checked in
[`verify_extensions.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify_extensions.py).

| Gap | Result |
|---|---|
| The bound is loose for rich tools; when is it reached? | [Theorem 7′](#theorem-7-prime): exact accounting and the equality condition |
| Reasoning can create channels | [Theorem 11](#theorem-11): compute-gated channels |
| The law is average-case | [Theorem 12](#theorem-12): worst-case Coffee Law |
| Agents act, not just know; open-ended tasks | [Theorem 13](#theorem-13): acting is bounded by knowing |
| Bandwidth between agents | [Theorem 14](#theorem-14): the communication ceiling |
| The world changes | [Theorem 15](#theorem-15): drift caps knowledge |
| Adversarial agents | [Theorem 16](#theorem-16): provenance |
| Continuous quantities | [Theorem 17](#theorem-17): the continuous Coffee Law |
| What "knowledge in the weights" means | [Proposition 18](#prop-18): possible worlds |
| Where to spend: tools or reasoning? | [Proposition 19](#prop-19): headroom |
| Can any of this be measured? | [Measuring the quantities](#measuring) |
| What is still open | [Open problems](#open) |

--------------------------------------------------------------------------------
<a id="theorem-7-prime"></a>

## Theorem 7′ — Exact accounting, and when the Coffee Law is tight

**Theorem 7′.** Under (A1)–(A4):

(a) *Exact accounting.*

        I(Θ ; D_T)  =  I(Θ ; D₀)  +  Σ_{t<T}  E_s[ I(Θ ; O_{t+1} | S_t = s) ].

(b) *Refined budget.* Each term is at most the capacity of the channel actually
used, and every capacity is at most `min(log |𝒪_t|, log |supp Θ|)`:

        I(Θ ; D_T)  ≤  I(Θ ; D₀)  +  Σ_{t<T}  E[ Cap(K_{t, Q_t, D_t}) ]  ≤  I(Θ ; D₀) + Σ_t C_t.

(c) *Equality condition.* `I(Θ ; D_T) = I(Θ ; D₀) + Σ_t C_t` holds iff, for
every round and almost every `s`, the posterior `p(· | s)` is a
capacity-achieving input for the channel `K_{t,q,d}` chosen at `s`, and that
channel's capacity equals `C_t`.

*Proof.* (a) is the chain-rule decomposition in the proof of
[Theorem 7](formal_proofs.md#theorem-7), where the noise and query terms are
exactly zero. (b) Each conditional term is the mutual information of the chosen
channel under input law `p(·|s)`, so at most that channel's capacity;
`I(Θ;O) ≤ min(H(O), H(Θ)) ≤ min(log|𝒪|, log|supp Θ|)`. (c) The bound in
Theorem 7 is the sum of the inequalities `I(Θ;O|S=s) ≤ Cap(K_{t,q,d}) ≤ C_t`;
the sum is tight iff each is tight almost surely. ∎

**What this fixes.** A tool that returns rich output (a web page) does not have
"astronomical" capacity *about `Θ`*: capacity is also capped by `log |supp Θ|`,
so one rich call can at most reveal `Θ` completely. When tools are rich, the
binding quantity is the realized gain in (a), which can be estimated per call,
rather than the worst-case capacity. The capacity form bites hardest for
low-capacity tools: tests, yes/no checks, sensors, short API answers. The
worked example in [`verify.py`](https://github.com/mbhatt1/coffeelaw/blob/main/formal_proofs/verify.py)
(Scenario 7) shows both cases of (c): querying the other bit keeps the queried
bit's posterior uniform, which is capacity-achieving for a binary symmetric
channel, so equality holds; re-querying a bit already observed does not.

**Example (software debugging).** A test run returns pass or fail, so it is a
channel with capacity at most 1 bit. If the bad commit is equally likely to be
any of `n`, identifying it with error at most `ε` needs at least
`(1 − ε) log₂ n − 1` test runs by [Corollary 7.1](formal_proofs.md#theorem-7),
whatever agents run the search; near-certain identification needs about
`log₂ n`. Binary search (`git bisect`) uses `⌈log₂ n⌉` and is error-free, so it
is essentially optimal. With a flaky
test that misreports with probability `p`, each run carries at most
`1 − h(p)` bits, so the minimum grows by the factor `1 / (1 − h(p))`.

--------------------------------------------------------------------------------
<a id="theorem-11"></a>

## Theorem 11 — Reasoning can open channels (compute-gated channels)

Theorems 7–10 treat the set of tools as fixed. In practice, issuing a query can
first require *finding* it: an endpoint, a credential, a cached answer, an
exploit. That is computation, and it can raise the information budget.

**Setup.** Let `A_S(d)` be the set of queries a collective can issue from record
`d` using total compute at most `S`, and suppose `Q_t ∈ A_S(D_t)` almost surely.

**Theorem 11.**

(a) *Compute-dependent budget.*

        I(Θ ; D_T)  ≤  I(Θ ; D₀)  +  Σ_{t<T}  sup_{d, q ∈ A_S(d)} Cap(K_{t,q,d})  =:  B_T(S),

and `B_T(S)` is nondecreasing in `S`.

(b) *Reasoning can raise available information.* There are settings in which
available information grows with compute. Let `Θ` be uniform on 4 values, let a
tool reveal `Θ` only after the correct `k`-bit password is entered, and let the
password `W` be uniform and independent of `Θ`. A collective that tries `S`
passwords has

        I(Θ ; record)  =  min(1, S / 2^k) · H(Θ),

and no strategy does better.

*Proof.* (a) Theorem 7′(b) with each capacity taken over the queries that can
actually be issued. Larger `S` enlarges `A_S`, so the supremum cannot decrease.
(b) Each try reveals only whether that guess is right, which is independent of
`Θ`. Let `U` be the event that some try succeeds; `U` is independent of `Θ`.
Given `U`, the record contains `Θ`; given not `U`, the record is independent of
`Θ`. So `I(Θ ; record) = P(U) · H(Θ)`. Each try eliminates at most one of the
`2^k` equally likely passwords, so `P(U) ≤ S / 2^k`, with equality for any
sequence of distinct guesses. ∎

**Corollary 11.1 (what a sandbox is).** A sandbox is a guarantee that, for every
compute level `S` an agent can actually use, `A_S(d) ⊆ A_allowed` for all `d`.
Under that guarantee the budget is at most
`I(Θ ; D₀) + Σ_t sup_{q ∈ A_allowed} Cap(K_{t,q,d})`, no matter how capable the
agents are. An escape is exactly a feasible `S` with `A_S ⊄ A_allowed`. ∎

**The refined claim.** The earlier statement "reasoning raises usable
information, never available information" holds **for a fixed set of reachable
channels**. Reasoning that discovers new channels raises available information,
up to `B_T(S)`. In OpenAI's reported 2026 incidents, models under evaluation
escaped their test sandbox and retrieved benchmark answers from an outside
server. That is this theorem in practice: the answers were not in the sandbox,
so the only route to them was a channel the designers had not allowed.

--------------------------------------------------------------------------------
<a id="theorem-12"></a>

## Theorem 12 — The worst-case Coffee Law (no prior needed)

The Coffee Law is stated for a random `Θ` with a prior. Theorem 12 removes the
prior: it applies to any finite set of possible truths and says there is a
specific truth on which the collective errs.

**Setup.** Let `{θ₁, …, θ_M}` (`M ≥ 3`) be possible truths, with no prior. The
starting data is drawn as `D₀ ∼ K₀(· | θ)`, a channel with capacity `C₀`
(`C₀ = 0` if the starting data does not depend on which truth holds). Tools
satisfy (A2)–(A4) for every `θ_i`.

**Theorem 12.** For every collective strategy and estimator `Θ̂`,

        max_i  P_{θ_i}( Θ̂ ≠ θ_i )  ≥  ( log M − min(log M, C₀ + Σ_{t<T} C_t) − 1 ) / log(M − 1).

*Proof.* Put the uniform prior on `{θ_i}`. The maximum error over `i` is at
least the average error, which is the error under that prior. Under it,
`H(Θ) = log M`, `I(Θ ; D₀) ≤ C₀`, and [Theorem 7](formal_proofs.md#theorem-7)
bounds `I(Θ ; D_T)` by `min(log M, C₀ + Σ_t C_t)`. Fano's inequality finishes. ∎

**Reading.** For any family of `M` tasks the collective cannot distinguish
beyond its budget, there is a particular task on which it is wrong at least
this often. This is Fano's method, standard for minimax lower bounds, applied
to agent collectives. Checked on 300 random channels.

--------------------------------------------------------------------------------
<a id="theorem-13"></a>

## Theorem 13 — Acting is bounded by knowing

Assumption (A4) excludes outcomes the agents control. Most agentic tasks are
about acting: fixing code, operating systems, maximizing a reward. Theorem 13
covers acting in a fixed but unknown environment.

**Setup.** The environment `W` is fixed and unknown, and the agents' actions do
not change it (A4 for `W`). There are `M ≥ 3` actions. The best action
`A* = a*(W)` is unique, with a **gap** `Δ > 0`: `u(a*(W), W) − u(a, W) ≥ Δ` for
every `a ≠ a*(W)`. Actions may themselves be tool calls whose outcomes satisfy
(A2)–(A3) with `W` as the truth. `A_t` is the action at round `t`.

**Theorem 13.** At every round `t`,

        E[ regret_t ]  ≥  Δ · P(A_t ≠ A*)
                       ≥  Δ · ( H(A*) − min(H(A*), I(W ; D₀) + Σ_{s<t} C_s) − 1 ) / log(M − 1),

and cumulative regret is at least the sum of these floors.

*Proof.* The first inequality is the gap condition. `A_t` is a function of
`D_t` and independent noise, so it is an estimator of `A*`. `A*` is a function
of `W`, so `I(A* ; D_t) ≤ I(W ; D_t)`, which [Theorem 7](formal_proofs.md#theorem-7)
bounds by `I(W ; D₀) + Σ_{s<t} C_s`; also `I(A* ; D_t) ≤ H(A*)`. Fano's
inequality applied to `A*` gives the floor. ∎

**Open-ended tasks.** Writing code or a document has no single right answer,
but it has a hidden **specification** — what the user actually wants. Take `W`
to be that specification and the output to be the action. Whenever outputs that
miss the specification lose at least `Δ` in quality, Theorem 13 bounds
performance by information about the specification: an agent cannot reliably
build what it has not learned the user wants.

**Imperfect agents.** For LLM agents this lower bound pairs with the upper
bound of [Theorem B4](llm_specialization.md#b4): regret is at least the
information floor here, and at most the sum of information shortfall, channel
loss, miscalibration and decision error there.

Checked on a 4-armed bandit with a greedy policy: per-round regret stays above
the floor in every round.

--------------------------------------------------------------------------------
<a id="theorem-14"></a>

## Theorem 14 — The communication ceiling

Theorem 9 counts compute but treats the board as free. Real agents have limited
context and message length. When data is split across agents, what any single
agent can learn is limited by what reaches it.

**Theorem 14.**

(a) *Bandwidth bound.* Let agent 1 hold `D¹`, and let the transcript `Π` of all
messages reaching agent 1 take at most `2^L` values (`L` bits). Then, for
anything agent 1 outputs,

        I(Θ ; D¹, Π)  ≤  I(Θ ; D¹)  +  L.

(b) *Some functions need many bits.* Let agents 1 and 2 hold independent uniform
`x, y ∈ {0,1}ⁿ`, and let `Θ = ⟨x, y⟩ mod 2`. Then `I(Θ ; x) = I(Θ ; y) = 0` and
`I(Θ ; x, y) = H(Θ)`: together they know `Θ`. Yet for any protocol, with any
number of rounds and any computation, in which at most `L` bits are exchanged,

        I(Θ ; x, Π)  ≤  log₂(1 + 2^{L + 1 − n/2}).

So agent 1 learns essentially nothing until about `n/2` bits have been
exchanged.

*Proof.* (a) Chain rule: `I(Θ ; D¹, Π) = I(Θ ; D¹) + I(Θ ; Π | D¹)` and
`I(Θ ; Π | D¹) ≤ H(Π) ≤ L`. (b) Append agent 1's output to the transcript,
giving a protocol of `L + 1` bits whose last bit is the output. By Lindsey's
lemma, the inner-product function has discrepancy at most `2^{−n/2}` under the
uniform distribution, so any `c`-bit protocol has advantage at most
`2^c · 2^{−n/2}`: it is correct with probability at most `½ + 2^{L − n/2}`. This
holds for randomized protocols too, by averaging. Sampling a guess from agent
1's exact posterior is such a protocol, so by
[Lemma 10.1](formal_proofs.md#theorem-9),
`H(Θ | x, Π) ≥ −log(½ + 2^{L−n/2})`. With `H(Θ) ≤ 1`, this gives the bound. ∎

**Reading.** This is a third ceiling, distinct from information and compute:
the collective as a whole holds the answer, but no member can learn it without
enough communication. Parity splits cheaply (one bit per agent; see
[Theorem 10](formal_proofs.md#theorem-9)); inner product needs about `n/2` bits.
Unlike Theorem 10(b), this bound holds even with unlimited computation.

Checked: Lindsey's lemma on all `256 × 256` rectangle pairs for `n = 3`, and
optimized one-way protocols for `n = 6`, `L = 1` stay below the bound. The
inner-product lower bound is due to Chor and Goldreich (SIAM J. Computing,
1988); see Kushilevitz and Nisan, *Communication Complexity* (1997).

--------------------------------------------------------------------------------
<a id="theorem-15"></a>

## Theorem 15 — In a changing world, knowledge saturates

Assumption (A4) also fixes `Θ` in time. Here the truth drifts.

**Setup.** The truth is a Markov chain `Θ₀, Θ₁, …` with transition kernel `P`,
unaffected by the agents. In round `t` the tools observe the current state
`Θ_t`, satisfying (A2)–(A3) with `Θ_t` as the truth; then `Θ_{t+1}` is drawn
from `P(· | Θ_t)`. Let `η = η_KL(P) ≤ 1` be the **strong data-processing
constant** of `P`: the smallest `η` with `I(U ; Y) ≤ η · I(U ; X)` whenever
`U — X — Y` and `Y ∼ P(· | X)`.

**Theorem 15.** Let `x_t = I(Θ_t ; D_t)` be what the record knows about the
current state. Then

        x_{t+1}  ≤  η · (x_t + C_t),

so with `C_t ≤ C`,

        x_t  ≤  η^t x₀  +  η C / (1 − η)        (if η < 1).

*Proof.* One round of tool use adds at most `C_t` about the current state, by
the argument of Theorem 7 with `Θ_t` as the truth:
`I(Θ_t ; D_{t+1}) ≤ x_t + C_t`. Because the drift ignores the agents,
`D_{t+1} — Θ_t — Θ_{t+1}` is a Markov chain, so the strong data-processing
inequality gives `I(Θ_{t+1} ; D_{t+1}) ≤ η · I(Θ_t ; D_{t+1})`. Unroll the
recursion. ∎

**Reading.** In a changing world, knowledge does not accumulate forever. It
saturates near `ηC / (1 − η)`, however long the agents run: drift erases old
knowledge, and tools must keep refreshing it. For a bit that flips with
probability `ρ` each round, `η = (1 − 2ρ)²`. Checked exactly for `ρ = p = 0.1`:
the record's knowledge levels off near 0.384 bits by round 9, under the cap of
0.944.
(Strong data-processing inequalities: Ahlswede and Gács, *Annals of
Probability*, 1976.)

--------------------------------------------------------------------------------
<a id="theorem-16"></a>

## Theorem 16 — Adversaries, and why provenance matters

**Setup.** An adversarial agent posts messages `M` chosen by any strategy. The
record **keeps provenance** if each post is stored with its author.

**Theorem 16.**

(a) *With provenance, adversaries cannot remove information.* For any
adversarial posts, `I(Θ ; D, M) ≥ I(Θ ; D)`. For usable information,
`I_V(D, M → Θ) ≥ I_V(D → Θ)` whenever `V` can ignore a named author's posts.

(b) *Adversaries without the truth cannot add information.* If the adversary's
posts are functions of `D₀` and truth-independent noise, they add no available
information ([Theorem 1](formal_proofs.md#theorem-1)). An adversary with access
to `Θ` is a side channel, assumption (A1).

(c) *Without provenance, one adversary can erase everything.* If an honest
agent posts `Θ` (a fair bit) and an adversary posts `1 − Θ`, an unattributed
record (the multiset of posts) carries **zero** bits about `Θ`, while the
attributed record carries 1 bit.

*Proof.* (a) Chain rule: `I(Θ ; D, M) = I(Θ ; D) + I(Θ ; M | D) ≥ I(Θ ; D)`. If
`V` contains, for each predictor, the version that ignores the adversary's
posts, its best loss on `(D, M)` is no worse than on `D`. (b) is Theorem 1.
(c) The multiset is always `{0, 1}`, whatever `Θ` is. ∎

**Reading.** Adversaries cannot destroy information, only hide it, and they
hide it by breaking provenance. The design rule is to authenticate and
attribute every message. With provenance, the remaining damage is to
*decisions*: persuasion of imperfect agents ([B3](llm_specialization.md#b3))
and flooding the context of agents that cannot ignore posts. Checked: all 16
adversary strategies against an attributed record, and the erasure witness.

--------------------------------------------------------------------------------
<a id="theorem-17"></a>

## Theorem 17 — The continuous Coffee Law

**Setup.** `Θ` is real-valued with finite differential entropy `h(Θ)`. Mutual
information is defined in general (Cover & Thomas, ch. 8), and capacities are
taken under each tool's input constraints. For example, a tool returning
`Θ + N(0, σ²)` has capacity `½ log(1 + P/σ²)` per call under the constraint
`E[Θ²] ≤ P`.

**Theorem 17.** The budget of [Theorem 7](formal_proofs.md#theorem-7) holds
verbatim, `I(Θ ; D_T) ≤ I(Θ ; D₀) + Σ_t C_t`, and Fano's inequality is replaced
by its estimation counterpart: for any estimator `Θ̂`,

        E[(Θ − Θ̂)²]  ≥  (1 / (2πe)) · 2^{2 (h(Θ) − I(Θ ; D₀) − Σ_t C_t)}.

*Proof.* The chain-rule argument of Theorem 7 uses only properties that hold
for general random variables. For the Gaussian tool, a posterior's second moment
`P_s` averages to at most `P`, and `½ log(1 + P_s/σ²)` is concave in `P_s`, so
Jensen keeps each round's gain within `½ log(1 + P/σ²)`. For the estimation
bound, `h(Θ | Θ̂) ≥ h(Θ) − I(Θ ; D_T)` by the data-processing inequality, and
a variable with second moment `v` has differential entropy at most
`½ log(2πe v)` (Cover & Thomas, Theorem 8.6.6). ∎

**Reading.** For quantities rather than categories, information buys squared
error at the exchange rate of a factor of 4 per bit. Checked for a Gaussian
truth observed through Gaussian noise, where the bound holds with equality.

--------------------------------------------------------------------------------
<a id="prop-18"></a>

## Proposition 18 — What a model's weights know: possible worlds

The theory counts a pretrained model's weights as part of the starting data
`D₀`. Read literally, that is a puzzle. A trained model's weights are fixed
numbers, so over a distribution of questions they carry **zero** mutual
information about anything; and for a factual question ("capital of France?")
the question alone determines the answer, so all the information is already
"available". Mutual information needs something random, and for facts about
the world the right random object is the world itself.

**Setup.** Let `W` be the state of the world, with a prior over the ways things
could have been. A question `X` is drawn independently of `W`, and its answer is
`Θ = θ(W, X)`. Training is a channel from the world to the weights: a corpus is
drawn from `P(· | W)` and the weights `M` are computed from it, so
`W — corpus — M`, and `M` is independent of `X`.

**Proposition 18.**

(a) *Over possible worlds, weights carry information bounded by the training
data.*

        I(Θ ; X, M)  ≤  I(Θ ; X)  +  I(W ; M)    and    I(W ; M)  ≤  I(W ; corpus).

So the Coffee Law applies with `D₀ = (X, M)`, and what a model can know about the
world is bounded by what its training data carried.

(b) *In one fixed world, knowledge is usable information.* If `W` is fixed, then
`M` is constant, `Θ` is a function of `X`, and
`I(Θ ; X, M) = I(Θ ; X) = H(Θ)`. Everything is available; the model's knowledge
shows up only as **usable** information `I_V(X, M → Θ)`
([Theorem 8](formal_proofs.md#theorem-8)): its ability to compute the answer
from the question.

*Proof.* (a) Chain rule: `I(Θ ; X, M) = I(Θ ; X) + I(Θ ; M | X)`. Given `X`, `Θ`
is a function of `W`, so `I(Θ ; M | X) ≤ I(W ; M | X)` by the data-processing
inequality, and `I(W ; M | X) = I(W ; M)` because `X` is independent of
`(W, M)`. The second inequality is the data-processing inequality for
`W — corpus — M`. (b) With `W` fixed, `M` is constant and `Θ = θ(w, X)`. ∎

**Reading.** When a model "knows" a fact, two statements are true at once.
Across possible worlds, its weights carry information about which world this is,
at most what its training data carried. Within the actual world, that
information shows up as the model's ability to compute answers from questions,
which is usable information, not available information. This is why debate can
improve factuality ([Theorem 8](formal_proofs.md#theorem-8)): within the actual
world, the facts are usable-information problems. It is also why a model cannot
know facts its training data never carried. Checked on a two-fact world in which
training reveals one fact: the weights add exactly the information about the
revealed fact, and in a fixed world they add nothing.

--------------------------------------------------------------------------------
<a id="prop-19"></a>

## Proposition 19 — Where to spend: headroom on each lever

The ceilings say how much each kind of spending can possibly buy.

**Proposition 19.** Let `D_T` be the current record and `V` the agents' current
predictive family.

(a) *Reasoning headroom.* More reasoning on the same record (any larger family
`V′ ⊇ V`) raises usable information by at most the reasoning gap:

        I_{V′}(D_T → Θ) − I_V(D_T → Θ)  ≤  I(Θ ; D_T) − I_V(D_T → Θ).

(b) *Information headroom.* `T′` more tool rounds raise available information by
at most

        min( Σ_{t=T}^{T+T′−1} C_t ,  H(Θ) − I(Θ ; D_T) ).

(c) *Both together* raise usable information by at most the reasoning gap plus
`Σ_t C_t` over the new rounds.

*Proof.* (a) `I_{V′}(D_T → Θ) ≤ I(Θ ; D_T)` by
[Theorem 8](formal_proofs.md#theorem-8)(a). (b) Exact accounting
([Theorem 7′](#theorem-7-prime)) bounds each new round by its capacity, and
`I(Θ ; D) ≤ H(Θ)`. (c) `I_{V′}(D_{T+T′} → Θ) ≤ I(Θ ; D_{T+T′}) ≤ I(Θ ; D_T) + Σ C_t`;
subtract `I_V(D_T → Θ)`. ∎

**A spending rule.** Each lever's possible return is capped by its headroom.
Invest in the lever whose headroom per unit cost is larger, and stop investing in
a lever whose headroom is used up. In the noisy example of
[Theorem 7](formal_proofs.md#theorem-7), after two calls the reasoning gap for a
one-look agent is 0.80 bits and the information headroom is 0.94 bits, so both
levers are open; with a noise-free tool after two calls, information headroom is
zero and only reasoning can help.

--------------------------------------------------------------------------------
<a id="measuring"></a>

## Measuring the quantities

The bounds are only as useful as the quantities in them can be estimated.

- **Usable information** is directly estimable. For any predictor a team
  actually runs, `H(Θ)` minus its held-out log-loss lower-bounds the team's
  usable information. Xu et al. (ICLR 2020) give PAC-style guarantees for
  estimating usable information under computational constraints.
- **Tool capacity** is bounded by the size of the answer space, and for noisy
  tools by measured error rates. A test that flakes with probability `p` has
  capacity at most `1 − h(p)`; a tool returning one of `m` answers has at most
  `log₂ m` bits.
- **Available information**, and `I(W ; M)` for a pretrained model, are well
  defined but hard to estimate: they are maxima over all computations. The best
  usable-information estimate is a lower bound. When no good estimate exists,
  the theory's value is structural: which levers can help at all.

--------------------------------------------------------------------------------
<a id="open"></a>

## What remains open

**Scope note.** None of the proofs inspect the agents, so every result applies
unchanged to people, programs and mixed human–AI teams.

These gaps are not closed here, and some cannot be closed with current
mathematics:

1. **Compute for typical tasks.** [Theorem 10](formal_proofs.md#theorem-9)
   covers the extremes (parity, cryptography). A general formula for the
   compute a task needs would settle open problems such as P vs NP. Bounds for
   specific task families, such as multi-step reasoning chains or search, are
   open.
2. **Reaching the budget in general.** [Theorem 7′(c)](#theorem-7-prime) says
   exactly when the bound is tight, but gives no algorithm that finds a
   capacity-achieving query strategy for arbitrary tools.
3. **Outcomes the agents control.** [Theorem 13](#theorem-13) covers acting in a
   fixed environment. When the agents' actions change the environment itself
   (self-fulfilling predictions, systems they modify), a different theory is
   needed.
4. **Lower bounds for LLM-specific losses.** [Part B](llm_specialization.md#part-b)
   bounds the cost of miscalibration and persuasion from above. Showing that a
   given model *must* incur such losses requires measurement, not proof.
5. **Measuring the inputs.** `I(W ; M)` for a pretrained model
   ([Proposition 18](#prop-18)), and `C_t` for real tools, are well defined but
   hard to estimate. The theory says where they enter, not what they are.
6. **Trading one ceiling for another.** The information, compute and
   communication ceilings each hold, so their minimum does too. Results on
   *trading* them — spending compute to save communication, or tool calls to
   save reasoning — are open, and they are the practically important question.
7. **Agents that learn across episodes.** An agent fine-tuned on its own tool
   results changes its starting data over time. A continual-learning version of
   the law, which tracks how much each episode's results can add to the
   weights, is not given here.
8. **Formal verification.** The proofs are mathematical and checked numerically
   on finite examples; none is machine-verified (for example in Lean).
9. **Incentives.** The theory says what agents *can* learn and do, not what they
   will *try* to do. Reward hacking, persistence on impossible tasks and agents
   adopting one another's goals concern objectives; modeling them needs game
   theory or alignment theory on top of these bounds.
   [Theorem 16](#theorem-16) covers adversaries only in a narrow sense.
