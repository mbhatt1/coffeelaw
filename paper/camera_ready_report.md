# Camera-ready report: The Coffee Law

Produced with the `camera-ready` skill (`.claude/skills/camera-ready`).
Checker: `check_paper.py`; numbers registry: `numbers.json`.

## Status

| Version | File | Pages (body / total) | Limit used | Blocking issues |
|---|---|---|---|---|
| Full (AGENT 2027 full track, named) | `coffee_law.pdf` | 6 / 7 | 8 excl. references | 0 (affiliation set) |
| Short (AGENT 2027 short track, named) | `coffee_law_short.pdf` | 5 / 6 | 5 excl. references | 0 (affiliation set) |
| Anonymous short (double-blind, e.g. ISIT) | `coffee_law_anon.pdf` | 5 / 6, page 6 references only | 5 + references-only page | **0** |
| Camera-ready (full, copyright notice) | `coffee_law_final.pdf` | 6 / 7 | 8 excl. references | 0 (affiliation set) |

- All fonts embedded and subsetted, Type 1 only (IEEE PDF eXpress requirement).
- US letter, no overfull boxes, no undefined references or citations.
- All 7 numbers in the registry match the output of the check scripts.
- Short versions move the detailed extensions, the imperfect-agents paragraph
  and the exchange-rates paragraph to the notes, with a pointer in their place.

## Revision 2: the decomposition as the thesis

The paper now leads with the claim that different mechanisms of collective
intelligence have different kinds of bounds, and with the question "which
resource limits this collective, and what intervention removes that limit?"

- **Abstract** rewritten around the thesis and the three theories (information,
  computational and communication complexity).
- **Introduction**: new thesis paragraph; contributions restated at three levels
  (conceptual, scientific utility, mathematical), claimed in that order.
- **New Section IX, "What the Decomposition Predicts"**: seven predictions that
  a single capability axis does not make, each marked supported, consistent or
  untested: a diagnostic by targeted intervention; model vs prompt diversity;
  debate at matched compute; post-training lifting one ceiling; the
  communication signature; the tool-use error floor; where channel-seeking
  appears; plus exchange rates between levers. The old "testable predictions"
  paragraph is folded in.
- **New evidence row and citation**: Yue et al. (NeurIPS 2025, verified) on RL
  with verifiable rewards — better at small pass@k, base model better at large
  k — supporting prediction 4.
- **Related work** positions the earlier papers as each addressing one ceiling.
- **Conclusion** rewritten around the reframed question.
- **Claim audit of the new text** caught four overclaims, all fixed: "adding
  agents does not raise the budget" (it does through more tool access);
  "infeasible" (conditional on cryptographic assumptions); different corpora
  "carry" different information ("can carry"); post-training "should not help
  information-bound tasks" ("unless its training signal carries new facts").

## Fixed

### Mechanical
- Removed an unused bibliography entry (`shannon1948`).
- Copyright notice: added a camera-ready variant with `\IEEEpubid`. The first
  attempt declared it after `\maketitle`, which printed the notice over the
  body text; the visual check caught it, and the notice now precedes
  `\maketitle` with `\IEEEpubidadjcol` in the second column.
- Inline square root in "Imperfect agents" collided with the line above; now a
  displayed inequality.

### Claims (backed by a proof, a citation or a check, or weakened)
- Intro: "A common belief is that [more agents/debate/thinking] helps with both.
  It does not" → "It helps with only one of them" (it does help the
  compute-bound question).
- "What is random": "In one fixed world, M is constant" → "Θ is a function of X
  and M carries no further information about it" (training can be random even
  in one world).
- Debugging example: "about log n test runs" → the exact statement under a
  uniform prior, `(1−ε) log n − 1`, with a pointer to Corollary 2.
- Tightness: added the second condition (each chosen tool attains `C_t`).
- Theorem 6: "S guesses give exactly …" → "at most …, with equality for
  distinct guesses".
- Continuous truths: "each bit divides the best squared error by 4" → "divides
  the error floor by 4".
- Measuring: held-out log-loss "lower-bounds" → "estimates a lower bound on"
  usable information (it is a finite-sample estimate).
- Evidence: "None contradicts it" → "the published findings we found …,
  including mixed ones; none contradicts it".

### Proofs
- Corollary 2: added `C > 0` (the bound divides by `C`).
- Theorem 2(c): the predictive family "reads at most one coordinate", so it
  contains the prior predictor as the definition requires.
- Lemma 1: states what "attains usable information δ" means (cross-entropy
  `H(Θ) − δ`), which the second claim uses.
- Drift: states `C_t ≤ C`, which the bound uses.

### Citations
- All 32 cited works have full author lists and venues. The 2023–26 LLM papers
  were confirmed by search (authors, venue, year). The classic references
  (Cover & Thomas, Shannon, Massey, Goldreich–Levin, Chor–Goldreich,
  Kushilevitz–Nisan, Ahlswede–Gács, Raginsky–Rakhlin, Williams–Beer, and the
  2020–24 ML papers) are standard but were not individually re-verified in this
  pass; check them against the publishers' listings before camera-ready.
- Table II's caption says findings are summarized from abstracts and published
  summaries.

### Presentation
- Table II caption defines *Supported* and *Consistent*.
- Table I caption defines the budget column.

## Reviewer objections and how the paper now answers each

| Objection | Answer in the paper |
|---|---|
| "This is just DPI and Fano." | Contributions list what does not follow from the textbook statements (Sec. I). |
| "Assumption A1 assumes the conclusion." | New sentence in Sec. II: A1 defines which channels exist; Theorem 6 treats agents that open new ones. |
| "Mutual information over what distribution?" | "What is random" (possible worlds) in Sec. II; worst-case version in Sec. VII. |
| "The bound is vacuous for rich tools." | Exact accounting and where the capacity form bites (Sec. III); measuring the quantities (Sec. VII). |
| "Theorem 3's constant and cost model don't fit LLMs." | New remark: holds in any fixed cost model (RAM steps, FLOPs, tokens), up to interleaving overhead. |
| "Communication theorem is only for two agents." | New sentence: the bounds apply to bits crossing any cut separating the inputs. |
| "Diverse agents help, contradicting 'more agents don't help'." | New sentence in Sec. VIII: diverse models carry different training (larger `D₀`) and compute different things (larger `V`); identical agents add only compute. |
| "The evidence is cherry-picked or overstated." | Mixed findings included; verdicts defined; summary-only sourcing stated. |
| "No experiments." | Limitations says so; P1 is marked open in Table II with a concrete test design. |
| "Debate helps factuality, so talking adds facts." | Sec. VIII: in a fixed world facts are usable-information problems. |

## Unverified
- Classic references: not individually re-verified in this pass (see Citations).
- Full texts of Shin (2026) and Yang et al. (2026), the closest prior work;
  arXiv is blocked in the authoring environment. Targeted searches found no
  tool-capacity budget in either.
- ISIT 2027 page and anonymity rules; the anonymous version follows ISIT 2025
  (5 pages + a references-only page, double-blind). Confirm against the 2027 call.
- AGENT 2027 deadline: November 13 per the ICSE call; the workshop page also
  shows November 27.

## Author decisions
1. Affiliation: done (Security Researcher, OWASP; Shrewd Research). No email is listed.
2. Name: this repository previously hosted a different "Coffee Law"; decide
   whether to distinguish them in the paper or rename.
3. Venue and track: AGENT 2027 full track (`coffee_law.pdf`) or short track
   (`coffee_law_short.pdf`), or ISIT (`coffee_law_anon.pdf`).
4. Copyright line: replace the `979-8-XXXX…` placeholder with the conference's
   exact text at camera-ready time.
5. Supplementary material for the anonymous version: an anonymized copy of the
   proofs and check scripts, if the venue allows it.

## Preprint pass (EasyChair Preprints, full named version)
- Re-verified against sources: Li et al. (30.1% vs 80.7%, ICML 2026), Zhao et al.
  (14 models), Yang et al. (two diverse agents match sixteen identical),
  Du et al. (plateau after 8 calls, from one model in three domains; Table II
  wording narrowed). All 7 registry numbers re-run and match.
- Limitations now says the baseline invariant is machine-verified in Lean.
- "Recently reported incidents" made singular (one cited incident).
- PDF title/author metadata set for the named versions (empty in the anonymous one).
- Not verified: OpenAI (2026) incident report; classic references.

## Full verification pass (web + independent math audit)
Six parallel audits: classic references, 22 evidence-table references, the OpenAI incident, prior art, and every theorem (with re-run of the check scripts).
- **Bib fixes**: Smit et al. author order; Zhao et al. venue (arXiv lists COLM 2026, not ICLR main; unconfirmed); Shin subtitle; InfoMosaic missing author (Xiong); OpenAI date. Five related-work references added (Ao et al., Zhang et al., Lutati, Rizvi-Martel et al., Finzi et al.), authors taken from arXiv pages.
- **Characterizations narrowed**: Shin (faithfulness, not accuracy), Choi (expected belief, martingale conditions), Yang (usable evidence), Kandpal, Li et al. (voting), Pu et al., Liu et al., Smit et al.
- **Math fixes**: Theorem 6 password example is a separate direct computation (the capacity bound is loose there); drift timing stated; one-sided-flake wording; (A3) independence from internal coins; "n >= 4" in the debugging example; pooling model behind 1/6; summary identity needs M a function of D; incident figure 0.266 now states its policy and reader.
- **Abstract**: "verified exactly" -> "checked numerically" (scripts mix exact fractions with float logs and check Theorem 5 only at small n).
- **Sandbox sentence** reworded to match OpenAI's technical report (sandbox bypass and compromise of an external provider; not clearly answer retrieval).
- **Still unverified**: OpenAI post itself (403; read via the technical report and mirrors); Lean build completion (no sorry/axiom found, `lake build` not confirmed); Lindsey's lemma only via secondary sources; Ao et al. and Rizvi-Martel et al. read from abstracts only.
