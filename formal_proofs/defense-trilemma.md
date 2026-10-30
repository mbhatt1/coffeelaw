# The Defense Trilemma: a companion result

*A prompt-injection defense that wraps the model and preprocesses its inputs
can't be continuous, utility-preserving and complete all at once. This note
explains what that claim says, what its proofs cover, and how it sits next to
the Coffee Law.*

By Manish Bhatt · October 2026

---

## The result

**"The Defense Trilemma: Why Prompt Injection Defense Wrappers Fail?"**
([arXiv 2604.06436](https://arxiv.org/abs/2604.06436)) proves that a
particular class of defense — a continuous wrapper that preprocesses the
model's inputs — cannot simultaneously satisfy three properties:

1. **Completeness** — it correctly blocks every injection.
2. **Utility preservation** — it passes every benign input unchanged.
3. **Continuity** — it varies continuously with the input.

The paper calls this a trilemma: any defense of this form must sacrifice at
least one. It is proved in Lean 4. The artifact ships 46 files, roughly 360
theorems, zero `sorry` and only Lean's three standard axioms.

## Scope

The authors are careful about scope, and that care is worth quoting:

> The theorems cover a single continuous wrapper that preprocesses inputs.
> They do not cover training-time alignment, architectural changes,
> discontinuous filters, output-side filters, ensembles or human review.

Discontinuous defenses — hard blocklists, classifiers that threshold, rules
that fire or don't — are outside the result. "Boundary fixation" happens
exactly at the decision threshold, and if behavior at that threshold is benign,
the theorem holds but does no practical harm. The trilemma is not a proof that
all defenses are useless; it is a proof that smooth wrappers face a
three-way trade-off.

The companion paper, "Truth Has a Boundary," examines the compatibility between
truth and confidence — what happens when a system is asked to be both accurate
and calibrated about its accuracy.

## Connection to the Coffee Law

The trilemma and the Coffee Law approach the same system from different angles.

The Coffee Law is about **information flow**: how much a collective can learn
from its starting data and its tool calls, and where that learning saturates.
It says nothing about the structure of individual defenses.

The trilemma is about **functional form**: what shapes a defense can and
cannot take, independently of how much information it holds.

The two results are compatible and complementary. A wrapper defense that
satisfies utility preservation is also constrained by the Coffee Law in a
specific way: it cannot *learn* which inputs are malicious without either
(a) holding informative starting data or (b) making informative queries to
the world. The trilemma says that, if it uses continuity to smooth that
judgment, it cannot make it everywhere correctly. The Coffee Law says that even
a discontinuous defense that makes the right call at every point is bounded
by what it knows — its starting data and any probes it is allowed to run.

Put differently: the trilemma pins down what a *continuous* defense cannot
do; the Coffee Law pins down what *any* defense cannot know. Neither subsumes
the other.

## The Lean artifact

The proofs are fully machine-checked. Running `lake build` in the repo
([github.com/mbhatt1/defensetrilemma](https://github.com/mbhatt1/defensetrilemma))
produces zero warnings under `sorry` and uses no custom axioms. The three
admitted axioms are Lean's standard propositional extensionality, functional
extensionality and the axiom of choice — the same three used in every serious
Lean 4 library.

The harder question is definitional: whether the formalization's continuity,
completeness and utility preservation match the real-world concepts they name.
The authors' own scope section is the best guide. The definitions are designed
around smooth functions on a metric input space; a defense implemented as a
neural network score threshold, a regex, or an ensemble sits outside those
definitions and is not covered by the theorems.

## What to cite

If you are citing this work, prefer the arXiv paper ([2604.06436](https://arxiv.org/abs/2604.06436))
over informal summaries. The paper's scope section is explicit about what
the theorems prove, and the Lean artifact lets readers check the proofs
directly.

---

*Related:
[Coffee Law](formal_proofs.md) ·
[extensions](extensions.md) ·
[Defense Trilemma paper](https://arxiv.org/abs/2604.06436) ·
[Defense Trilemma repo](https://github.com/mbhatt1/defensetrilemma)*
