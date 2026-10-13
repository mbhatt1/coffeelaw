# Paper: The Coffee Law

`coffee_law.tex` is a 6-page paper (including references) in the IEEE
conference template, written up from the proofs in
[`../formal_proofs`](../formal_proofs). The compiled PDF is
[`coffee_law.pdf`](coffee_law.pdf). It leads with intuition: each result opens
with a plain-language explanation, and a running example (an agent team
debugging a production incident) ties the three ceilings together.

## Versions (one source)

All content lives in `coffee_law.tex`; small wrappers set switches before
`\input`-ing it, so versions never drift apart.

| File | Version | Use |
|---|---|---|
| `coffee_law.tex` → `coffee_law.pdf` | Named submission | AGENT 2027 (single-anonymous); fits both the full and short tracks |
| `coffee_law_anon.tex` → `coffee_law_anon.pdf` | Anonymous | Double-blind venues such as ISIT |
| `coffee_law_final.tex` → `coffee_law_final.pdf` | Camera-ready | Adds the IEEE copyright notice (placeholder code until the conference supplies it) |

## Checking

The `camera-ready` skill (`.claude/skills/camera-ready/`) holds the checklist and
the checker. To recheck after edits:

```bash
cd paper
python3 ../.claude/skills/camera-ready/scripts/check_paper.py coffee_law.tex --page-limit 5 --numbers numbers.json
python3 ../.claude/skills/camera-ready/scripts/check_paper.py coffee_law_anon.tex --page-limit 5 --anonymous --names "Manish Bhatt,mbhatt1"
```

`numbers.json` ties every number in the paper to the check script that produces
it. The latest audit is in [`camera_ready_report.md`](camera_ready_report.md).

## Target venue

**AGENT 2027, International Workshop on Agentic Engineering, at ICSE 2027**
(<https://conf.researchr.org/home/icse-2027/agent-2027>).

- Topics include tool use and multi-agent coordination.
- Accepted papers are published by IEEE.
- Full papers: up to 8 pages excluding references; short papers: up to 5. The
  body is 5 pages, so the paper fits **either track** as is.
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
- Decide on the name: this repository previously hosted a different "Coffee
  Law" (context-quality power laws). If that work circulated, consider a
  sentence distinguishing them, or a new name.
- Read the cited 2025–26 papers in full. The empirical comparisons in
  the "Consistency with Empirical Findings" section are based on their
  abstracts and published summaries.

## Build

```bash
pdflatex coffee_law && bibtex coffee_law && pdflatex coffee_law && pdflatex coffee_law
```
