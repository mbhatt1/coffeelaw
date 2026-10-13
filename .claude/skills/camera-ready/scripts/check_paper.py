#!/usr/bin/env python3
"""
Mechanical camera-ready checks for a LaTeX paper.

Builds the paper (pdflatex + bibtex), then reports every problem a program can
find: LaTeX warnings, bibliography hygiene, PDF compliance (fonts, page size,
page limits), leftover placeholders, anonymity leaks, and numbers that must match
the output of check scripts. Judgement calls (are the claims right?) are left to
the person or model running the skill.

Usage:
  check_paper.py paper.tex [--page-limit N] [--refs-count]
                           [--anonymous --names "Ada Lovelace,lovelace"]
                           [--numbers numbers.json] [--no-build]

Exit status is 1 if any BLOCKING issue is found, else 0.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

BLOCK, WARN, INFO = "BLOCKING", "WARNING", "INFO"
findings = []


def add(level, area, msg):
    findings.append((level, area, msg))


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")


def tool(name):
    if shutil.which(name) is None:
        add(WARN, "tools", f"`{name}` not found; related checks skipped")
        return False
    return True


# ---------------------------------------------------------------- build
def build(tex_dir, stem):
    if not (tool("pdflatex") and tool("bibtex")):
        return
    for step in (["pdflatex", "-interaction=nonstopmode", stem],
                 ["bibtex", stem],
                 ["pdflatex", "-interaction=nonstopmode", stem],
                 ["pdflatex", "-interaction=nonstopmode", stem]):
        r = run(step, tex_dir)
        if step[0] == "bibtex" and r.returncode not in (0, 1):
            add(BLOCK, "build", f"bibtex failed:\n{r.stdout[-800:]}")


# ---------------------------------------------------------------- log
def scan_log(log_path, overfull_pt):
    if not os.path.exists(log_path):
        add(BLOCK, "build", f"no log at {log_path}")
        return
    log = open(log_path, errors="replace").read()
    for m in re.finditer(r"^! (.*)$", log, re.M):
        add(BLOCK, "latex", f"error: {m.group(1)}")
    for m in re.finditer(r"Overfull \\[hv]box \(([\d.]+)pt too \w+\)[^\n]*", log):
        if float(m.group(1)) > overfull_pt:
            add(BLOCK, "layout", m.group(0).strip())
    for m in re.finditer(r"(Reference|Citation) `([^']+)' on page \d+ undefined", log):
        add(BLOCK, "refs", f"undefined {m.group(1).lower()}: {m.group(2)}")
    for m in re.finditer(r"Label `([^']+)' multiply defined", log):
        add(BLOCK, "refs", f"label defined twice: {m.group(1)}")
    if "Rerun to get" in log or "Label(s) may have changed" in log:
        add(WARN, "build", "LaTeX asks for a rerun; cross-references may be stale")
    for m in re.finditer(r"File `([^']+)' not found", log):
        add(BLOCK, "latex", f"missing file: {m.group(1)}")


# ---------------------------------------------------------------- bibliography
def tex_sources(tex_path):
    """The main file plus anything it \\input{}s or \\include{}s."""
    seen, stack, text = set(), [tex_path], ""
    while stack:
        p = stack.pop()
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        t = open(p, errors="replace").read()
        t = re.sub(r"(?<!\\)%.*", "", t)          # strip comments
        text += t + "\n"
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", t):
            f = m.group(1)
            f = f if f.endswith(".tex") else f + ".tex"
            stack.append(os.path.join(os.path.dirname(p), f))
    return text


REQUIRED = {
    "article": ["author", "title", "journal", "year"],
    "inproceedings": ["author", "title", "booktitle", "year"],
    "book": ["author", "title", "publisher", "year"],
    "misc": ["author", "title", "year"],
}


def scan_bib(tex_text, tex_dir):
    bibs = re.findall(r"\\bibliography\{([^}]+)\}", tex_text)
    cited = set()
    for m in re.finditer(r"\\(?:no)?cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex_text):
        cited.update(k.strip() for k in m.group(1).split(","))
    cited.discard("*")
    entries = {}
    for b in bibs:
        for name in b.split(","):
            path = os.path.join(tex_dir, name.strip() + ("" if name.endswith(".bib") else ".bib"))
            if not os.path.exists(path):
                add(BLOCK, "bib", f"bibliography file missing: {path}")
                continue
            src = open(path, errors="replace").read()
            for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", src, re.S):
                kind, key, body = m.group(1).lower(), m.group(2), m.group(3)
                if key in entries:
                    add(BLOCK, "bib", f"duplicate key: {key}")
                fields = {f.lower() for f in re.findall(r"^\s*(\w+)\s*=", body, re.M)}
                entries[key] = (kind, fields, body)
    for k in sorted(cited - entries.keys()):
        add(BLOCK, "bib", f"cited but not in bibliography: {k}")
    for k in sorted(entries.keys() - cited):
        add(INFO, "bib", f"in bibliography but never cited (not printed): {k}")
    for k in sorted(cited & entries.keys()):
        kind, fields, body = entries[k]
        missing = [f for f in REQUIRED.get(kind, ["author", "title", "year"]) if f not in fields]
        if missing:
            add(WARN, "bib", f"{k} ({kind}) lacks {', '.join(missing)}")
        if re.search(r"\b(et al|others)\b", body):
            add(WARN, "bib", f"{k}: author list truncated ('et al'/'others'); give full names")
        if kind == "misc" and "arxiv" not in body.lower() and "url" not in fields and "howpublished" not in fields:
            add(WARN, "bib", f"{k}: misc entry with no locator (arXiv id, URL)")
    return cited


# ---------------------------------------------------------------- pdf
def page_texts(pdf):
    # reading order, not layout: in two-column pages a heading shares its line
    # with the other column under -layout
    out = run(["pdftotext", pdf, "-"], ".").stdout
    return out.split("\f")


def scan_pdf(pdf, page_limit, refs_count):
    if not os.path.exists(pdf):
        add(BLOCK, "pdf", f"no PDF at {pdf}")
        return []
    texts = []
    if tool("pdfinfo"):
        info = run(["pdfinfo", pdf], ".").stdout
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
        if size:
            w, h = float(size.group(1)), float(size.group(2))
            name = "US letter" if abs(w - 612) < 2 and abs(h - 792) < 2 else \
                   "A4" if abs(w - 595) < 2 and abs(h - 842) < 2 else f"{w}x{h}pt"
            add(INFO, "pdf", f"page size: {name}")
        add(INFO, "pdf", f"pages: {pages}")
    if tool("pdffonts"):
        rows = run(["pdffonts", pdf], ".").stdout.splitlines()[2:]
        for r in rows:
            cols = r.split()
            if len(cols) >= 5:
                if "Type 3" in r:
                    add(BLOCK, "fonts", f"Type 3 font (rejected by IEEE PDF eXpress): {cols[0]}")
                # columns: name type [encoding] emb sub uni object ID
                yes_no = [c for c in cols if c in ("yes", "no")]
                if len(yes_no) >= 2 and yes_no[0] == "no":
                    add(BLOCK, "fonts", f"font not embedded: {cols[0]}")
    if tool("pdftotext"):
        texts = page_texts(pdf)
        texts = [t for t in texts if t.strip()]
        ref_page = None
        for i, t in enumerate(texts):
            if re.search(r"^\s*(R\s?EFERENCES|REFERENCES|References|Bibliography)\s*$", t, re.M):
                ref_page = i + 1
                break
        if ref_page:
            add(INFO, "pdf", f"references start on page {ref_page}")
        if page_limit:
            counted = len(texts) if refs_count or not ref_page else ref_page
            # if references start mid-page, that page still holds body text
            if counted > page_limit:
                add(BLOCK, "length", f"{counted} counted pages > limit {page_limit} "
                    f"({'references count' if refs_count else 'body up to the references heading'})")
            else:
                add(INFO, "length", f"{counted} counted pages <= limit {page_limit}")
        joined = "\n".join(texts)
        if re.search(r"\?\?|\[\?\]", joined):
            add(BLOCK, "refs", "unresolved '??' or '[?]' appears in the PDF")
    return texts


# ---------------------------------------------------------------- placeholders, anonymity
PLACEHOLDERS = r"\b(TODO|FIXME|XXX|TBD|to be completed|lorem ipsum)\b"


def scan_text(tex_text, pdf_texts, anonymous, names):
    # A placeholder printed in the PDF blocks; one only in the source (e.g. in a
    # branch this variant doesn't build) is a warning to check before release.
    joined = "\n".join(pdf_texts)
    printed = {m.group(0).lower() for m in re.finditer(PLACEHOLDERS, joined, re.I)}
    for m in re.finditer(PLACEHOLDERS, tex_text, re.I):
        line = tex_text[: m.start()].count("\n") + 1
        if m.group(0).lower() in printed:
            add(BLOCK, "placeholder", f"'{m.group(0)}' printed in the PDF (source line {line})")
        else:
            add(WARN, "placeholder", f"'{m.group(0)}' in source line {line} (not printed in this variant)")
    if anonymous:
        for n in [x.strip() for x in names.split(",") if x.strip()]:
            if re.search(re.escape(n), joined, re.I):
                add(BLOCK, "anonymity", f"identifying string in PDF: {n}")
        for m in re.finditer(r"github\.com/[\w.-]+|[\w.+-]+@[\w-]+\.[\w.]+", joined):
            add(BLOCK, "anonymity", f"possible identifying link/email: {m.group(0)}")
        if re.search(r"^\s*Acknowledg", joined, re.M | re.I):
            add(BLOCK, "anonymity", "acknowledgments section present in an anonymous version")


# ---------------------------------------------------------------- number registry
def scan_numbers(registry_path, pdf_texts, base_dir):
    """registry: [{"value": "1.0620", "command": "python3 verify.py", "cwd": "..", "expect": "regex"}]"""
    reg = json.load(open(registry_path))
    joined = re.sub(r"\s+", " ", "\n".join(pdf_texts))
    for item in reg:
        v = item["value"]
        if v not in joined:
            add(WARN, "numbers", f"registry value {v} not found in the PDF ({item.get('note','')})")
        cmd = item.get("command")
        if cmd:
            cwd = os.path.join(base_dir, item.get("cwd", "."))
            r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
            if not re.search(item.get("expect", re.escape(v)), r.stdout):
                add(BLOCK, "numbers", f"{v}: check output of `{cmd}` does not match /{item.get('expect', v)}/")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex")
    ap.add_argument("--page-limit", type=int, default=0)
    ap.add_argument("--refs-count", action="store_true", help="references count toward the page limit")
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--names", default="")
    ap.add_argument("--numbers", help="JSON registry of numbers that must match check outputs")
    ap.add_argument("--overfull-pt", type=float, default=1.0)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()

    tex_path = os.path.abspath(a.tex)
    tex_dir, stem = os.path.dirname(tex_path), os.path.splitext(os.path.basename(tex_path))[0]
    if not a.no_build:
        build(tex_dir, stem)
    scan_log(os.path.join(tex_dir, stem + ".log"), a.overfull_pt)
    tex_text = tex_sources(tex_path)
    scan_bib(tex_text, tex_dir)
    pdf_texts = scan_pdf(os.path.join(tex_dir, stem + ".pdf"), a.page_limit, a.refs_count)
    scan_text(tex_text, pdf_texts, a.anonymous, a.names)
    if a.numbers:
        scan_numbers(a.numbers, pdf_texts, os.path.dirname(os.path.abspath(a.numbers)))

    order = {BLOCK: 0, WARN: 1, INFO: 2}
    findings.sort(key=lambda f: (order[f[0]], f[1]))
    print(f"# Camera-ready check: {os.path.basename(tex_path)}\n")
    for level in (BLOCK, WARN, INFO):
        rows = [f for f in findings if f[0] == level]
        print(f"## {level} ({len(rows)})")
        for _, area, msg in rows:
            print(f"- [{area}] {msg}")
        print()
    blocking = any(f[0] == BLOCK for f in findings)
    print("RESULT:", "BLOCKING ISSUES FOUND" if blocking else "no blocking issues")
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
