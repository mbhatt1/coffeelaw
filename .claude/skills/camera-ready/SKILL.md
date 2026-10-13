---
name: camera-ready
description: Make an academic LaTeX paper camera-ready and bulletproof before submission or final upload. Runs mechanical checks (build log, overfull boxes, undefined references, bibliography hygiene, embedded fonts for IEEE PDF eXpress, page limits, placeholders, anonymity leaks) and the judgement audits that catch rejections (every claim traced to a proof, citation or check; every proof step justified; numbers matched to code output; reviewer objections anticipated), then produces venue variants (short/full, anonymous, camera-ready with copyright notice) from one source. Use this whenever someone wants to finalize, polish, harden, proofread, "make bulletproof", prepare for submission or camera-ready, check compliance, fit a page limit, anonymize, or retarget a paper to another venue — even if they only say "check my paper" or "is this ready to submit".
---

# Camera-ready

A paper is camera-ready when nothing in it can be knocked down: it builds
cleanly, meets the venue's rules exactly, and every sentence that claims
something is backed by a proof, a citation that says it, or a check that
reproduces it. Most rejections and post-acceptance embarrassments come from a
small set of failures — an overclaim, a proof gap, a wrong citation, a number
that doesn't match the code, a formatting rule missed — so the work below hunts
each one systematically rather than rereading and hoping.

Work through the phases in order. Mechanical problems come first because they
are cheap and they hide judgement problems (an undefined reference can hide a
missing theorem). Apply fixes to the source as you go; collect anything only
the author can decide in an **author decisions** list instead of guessing.

## 1. Inventory

Before changing anything, write down:

- **Venue rules.** Page limit, and whether references count; template and class
  options; anonymity (double-blind or not); camera-ready extras (copyright
  notice, PDF eXpress, no page numbers). Read
  `references/venue-checklists.md` for the usual IEEE, double-blind and arXiv
  rules. Venue pages change, so confirm limits against the current call and
  record the source and date. If you cannot confirm, say so in the report.
- **The paper's supporting material.** Proof notes, code that checks numbers,
  data. These are what claims get traced to.
- **What the author must supply.** Affiliations, names, funding, venue
  choices. Never invent these; leave a visible placeholder and list it.

## 2. Mechanical checks

Run the bundled checker from the paper's directory:

```bash
python3 <skill>/scripts/check_paper.py paper.tex --page-limit N [--refs-count] \
    [--anonymous --names "Author Name,handle"] [--numbers numbers.json]
```

It builds the paper and reports, as BLOCKING / WARNING / INFO:
LaTeX errors, overfull boxes, undefined or duplicate labels, cited-but-missing
and unused bibliography entries, truncated author lists, Type 3 or unembedded
fonts, page size, page count against the limit (body up to the references
heading unless `--refs-count`), leftover placeholders (`TODO`, `to be
completed`), unresolved `??`, and — for anonymous versions — names, handles,
emails, repository links and acknowledgments.

`--numbers` takes a small JSON registry tying numbers printed in the paper to
the command that produces them, so a stale number fails the check:

```json
[{"value": "1.0620", "command": "python3 verify.py", "cwd": "../code",
  "expect": "1\\.0620", "note": "Table I, first row"}]
```

Fix everything BLOCKING. Treat WARNINGs as fixes unless there is a reason not
to, and write the reason down.

## 3. Claim audit

List every claim the paper makes — each theorem, each number, each comparative
or causal sentence, each "we show", each "to our knowledge", each sentence
summarizing someone else's result — and give it one backing:

- **Proved here**: name the theorem or proof.
- **Cited**: the cited source must actually say it. If you only saw an abstract
  or summary, the text must not claim more than the abstract does, and the
  paper should say the comparison rests on summaries.
- **Checked**: a script reproduces it (add it to the number registry).

A claim with no backing gets weakened until it has one, or cut. Watch
especially for: absolutes (*never*, *always*, *any*) whose proof has
conditions; the abstract and introduction promising more than the theorems
deliver; "supported by" where the evidence is only "consistent with";
novelty claims without "to our knowledge"; and rounding that changes a digit.

## 4. Proof audit

For each theorem, lemma and corollary, check:

- every assumption used in the proof is stated in the statement or the model,
  and every assumption stated is used;
- each step follows (write the one-line justification if the paper omits it);
- edge cases: zero, equality cases, small alphabets, division by quantities
  that could vanish, "for all" versus "there exists";
- units and constants (log base, nats versus bits, factors of 2);
- every symbol is defined before use and means one thing throughout;
- examples and witnesses are checked numerically where possible.

When a statement is false as written but true with a condition, add the
condition rather than hand-waving it.

## 5. Citation audit

For every reference: authors (full names, correct order), title, venue and
year match a primary listing (publisher page, proceedings, arXiv). Search to
verify rather than trusting memory; record what could not be verified. Prefer
the published version over the preprint when one exists. Make sure each
in-text use of a citation matches what the source claims.

## 6. Reviewer red-team

Write down the ten objections a sharp, skeptical reviewer would raise. Use
`references/reviewer-objections.md` as a starting catalogue. For each, decide:

- **fix it** in the text (usually one precise sentence where the objection
  would arise),
- **acknowledge it** in Limitations, or
- **rebut it** with a pointer to where the paper already answers it.

A good paper answers its strongest objection before the reviewer reaches it.

## 7. Presentation

- The abstract, introduction and conclusion claim exactly what the body proves.
- Intuition comes before each formal result; a running example helps.
- Figures and tables are legible at print size, with self-contained captions
  and defined terms (state what "Supported" vs "Consistent" means, for example).
- Terminology is consistent; no orphan or forward references to undefined
  material; footnotes appear at first use.

## 8. Venue variants from one source

Keep one source of truth. Put switches at the top of the main file and build
each variant from a tiny wrapper that sets them before `\input`-ing the main
file, so content never forks:

```latex
% main.tex
\providecommand{\PaperShort}{0}   % 1: drop optional material to fit a short limit
\providecommand{\PaperAnon}{0}    % 1: anonymous author block, no identifying links
\providecommand{\PaperFinal}{0}   % 1: camera-ready extras (copyright notice)
% wrapper short.tex
\newcommand{\PaperShort}{1}\input{main}
```

Run the checker on every variant with that venue's limit and anonymity flag.

## 9. Final pass and report

Rebuild every variant, rerun the checker, and render each page to an image to
look at it: layout problems (a table spilling into margins, a figure label
overlapping an axis) are easy to see and easy to miss in logs. Then write the
report using this structure:

```
# Camera-ready report: <paper>
## Status        (one line per variant: pages, limit, blocking issues = 0?)
## Fixed         (grouped: mechanical, claims, proofs, citations, presentation)
## Reviewer objections and how the paper now answers each
## Unverified    (anything you could not confirm, and why)
## Author decisions (things only the author can decide)
```

Be specific in the report: name the section and the before/after for each
substantive change, so the author can review it quickly.
