# Venue checklists

Venue rules change every year. Use these as the default list of things to
confirm, then check the current call for papers and record the source and date.

## IEEE conference (IEEEtran `conference` class)

Submission:
- `\documentclass[conference]{IEEEtran}`; don't change margins, fonts or
  spacing (altered templates can be kept out of IEEE Xplore).
- US letter, two columns, 10 pt.
- Page limit: confirm whether references count. Common patterns: "N pages
  excluding references", "N pages plus one page of references only".
- No page numbers, headers or footers (the class omits them by default).

Camera-ready:
- **Fonts**: every font embedded and subsetted; no Type 3 fonts. Check with
  `pdffonts paper.pdf` (all rows `emb yes`). pdfLaTeX with standard packages
  is usually fine; bitmap fonts from old `dvips` workflows are the usual cause
  of Type 3.
- **IEEE PDF eXpress**: upload with the conference ID; submit the PDF it returns.
- **Copyright notice** at the bottom of page 1, with the exact text the
  conference supplies (the code differs per event and per author category:
  standard, US government, Crown government, EU). With IEEEtran:
  ```latex
  \IEEEoverridecommandlockouts
  \IEEEpubid{\makebox[\columnwidth]{979-8-XXXX-XXXX-X/YY/\$31.00~\copyright~20YY IEEE \hfill}
  \hspace{\columnsep}\makebox[\columnwidth]{\hfill}}
  % then \IEEEpubidadjcol in the second column of page 1
  ```
  `\IEEEpubid` must come **before** `\maketitle`; declared after it, the notice
  prints over the body text. Put `\IEEEpubidadjcol` in a paragraph that falls
  in the second column of page 1, then render page 1 and look at the bottom of
  both columns.
  Leave the code as a visible placeholder until the conference provides it.
- Author block: full names, affiliations, emails; ORCID if requested.
- Remove submission-only material (line numbers, "under review" notes).

## Double-blind submission

- Author block replaced by "Anonymous Author(s)"; no affiliations, emails.
- No identifying URLs (personal GitHub, lab pages). Use an anonymized
  repository link or "available on acceptance".
- Self-citations in the third person ("Smith et al. showed"), never "our
  prior work [3]"; don't cite non-public work by the authors.
- No acknowledgments or funding section.
- PDF metadata: clear the Author field (`\hypersetup{pdfauthor={}}`).

## arXiv preprint

- Any readable format; the conference format is fine.
- Include the `.bbl` file (arXiv may not run bibtex), all figures, and no
  absolute paths.
- Add the venue and status if accepted ("To appear in ...").

## Counting pages

- "Excluding references": count pages up to and including the page where the
  references heading appears (that page usually holds body text too).
- "Plus one page of references only": the extra page may contain references and
  nothing else; body text must end by the last allowed page.
- Appendices usually count unless the call says otherwise.
