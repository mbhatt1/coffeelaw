# Reviewer objections catalogue

A starting list for the red-team phase. For each objection that applies, the
paper should answer it where the reader would raise it, usually in one precise
sentence, or concede it in Limitations.

## Novelty

- **"This is just a known result applied to X."** State plainly what does *not*
  follow from the textbook statements: what drops out, what composes, what new
  object is defined. List contributions by what is new, not by which tools are
  used.
- **"Prior work already did this."** Name the closest prior work and the precise
  difference. Use "to our knowledge" for novelty claims, and say how the
  literature was searched if it was limited.

## Assumptions

- **"The assumptions make the result trivial or circular."** Explain what each
  assumption rules out and why it is the right model; show what happens when it
  is relaxed (even one theorem helps).
- **"Over what distribution?"** Say what is random. For claims about fixed
  objects, give a worst-case or per-instance version, or explain the averaging.
- **"Does this apply to real systems?"** Give a concrete instance where the
  bound is tight or nearly tight, and say plainly where it is loose.

## Results

- **"The bound is vacuous in practice."** Identify the regime where it bites;
  give the exact or refined quantity for the regime where it doesn't.
- **"The constant hides everything."** State what the constant depends on.
- **"The example is cherry-picked."** Add a randomized or exhaustive check, or
  explain why the example is canonical.
- **"Proof step X doesn't follow."** Add the missing justification; if a
  condition is needed, add it to the statement.

## Evidence

- **"No experiments."** Either add them or frame the paper as theory, list
  testable predictions, and say which are untested.
- **"The cited evidence doesn't say that."** Quote or paraphrase narrowly; mark
  summary-only sources; distinguish "supported" from "consistent with".
- **"Selective citation."** Include mixed and contrary findings and explain them.

## Presentation

- **"Overclaiming in the abstract or title."** Every abstract sentence should
  map to a theorem, table row or clearly hedged statement.
- **"Hard to follow."** Intuition before formalism, one running example, a
  figure of the main idea.
- **"Undefined terms / inconsistent notation."** Define on first use; one
  symbol, one meaning.
- **"An unusual name."** If a result is given a name, say why, or be ready for
  the question.
