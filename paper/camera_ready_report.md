# Camera-ready report: The Coffee Law

Produced with the `camera-ready` skill (`.claude/skills/camera-ready`).
Checker: `check_paper.py`; numbers registry: `numbers.json`.

## Status

| Version | File | Pages (body / total) | Limit used | Blocking issues |
|---|---|---|---|---|
| Main (AGENT 2027, named) | `coffee_law.pdf` | 5 / 6 | 5 excl. references | 1: affiliation and email placeholders (author decision) |
| Anonymous (double-blind, e.g. ISIT) | `coffee_law_anon.pdf` | 5 / 6, page 6 references only | 5 + references-only page | **0** |
| Camera-ready (copyright notice) | `coffee_law_final.pdf` | 5 / 6 | 5 excl. references | 1: affiliation and email placeholders (author decision) |

- All fonts embedded and subsetted, Type 1 only (IEEE PDF eXpress requirement).
- US letter, no overfull boxes, no undefined references or citations.
- All 7 numbers in the registry match the output of the check scripts.
- The body fits within 5 pages, so the main version qualifies for the AGENT
  short-paper track (5 pages excluding references) as well as the full-paper
  track (8); no separate short cut is needed.

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
1. Affiliation and email (the only blocking item for the named versions).
2. Name: this repository previously hosted a different "Coffee Law"; decide
   whether to distinguish them in the paper or rename.
3. Venue and track: AGENT 2027 full or short track (both fit), or ISIT (use the
   anonymous version).
4. Copyright line: replace the `979-8-XXXX…` placeholder with the conference's
   exact text at camera-ready time.
5. Supplementary material for the anonymous version: an anonymized copy of the
   proofs and check scripts, if the venue allows it.
