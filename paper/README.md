# Paper: The Coffee Law

`coffee_law.tex` is a 5-page paper (including references) in the IEEE
conference template, written up from the proofs in
[`../formal_proofs`](../formal_proofs). The compiled PDF is
[`coffee_law.pdf`](coffee_law.pdf).

## Target venue

**AGENT 2027, International Workshop on Agentic Engineering, at ICSE 2027**
(<https://conf.researchr.org/home/icse-2027/agent-2027>).

- Topics include tool use and multi-agent coordination.
- Accepted papers are published by IEEE.
- Short papers: up to 5 pages excluding references, IEEE conference template.
- Single-anonymous review, so author names stay on the submission.
- Deadline: **November 13, 2026** per the ICSE 2027 workshop call. The workshop
  page also shows November 27; confirm on the submission site.

Alternatives:
- **ISIT 2027** (IEEE International Symposium on Information Theory, Sorrento),
  paper deadline January 10, 2027 (<https://2027.ieee-isit.org/>). A symposium,
  not a workshop, and a more theory-focused audience.
- **IEEE CAI 2027** (Singapore). Deadline reported as March 31 or May 15, 2027;
  confirm at <https://cai.ieee.org/2027/>.
- ITW 2027 (IEEE Information Theory Workshop) closed on October 11, 2026.

Check each venue's current call before submitting; these dates come from the
venues' pages and deadline trackers as of October 2026.

## Before submitting

- Fill in the affiliation and email in the author block.
- Read the cited 2025–26 papers in full. The empirical comparisons in
  the "Consistency with Empirical Findings" section are based on their
  abstracts and published summaries.

## Build

```bash
pdflatex coffee_law && bibtex coffee_law && pdflatex coffee_law && pdflatex coffee_law
```
