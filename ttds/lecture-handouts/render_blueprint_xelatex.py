#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/Users/huez/Documents/ttds")
LECTURES = ROOT / "lec"
OUT_DIR = ROOT / "lecture-handouts/d-rendered"
BUILD_DIR = ROOT / "lecture-handouts/d-build"
PREVIEW_DIR = ROOT / "lecture-handouts/d-previews"


def esc(text: object) -> str:
    s = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "→": r"\(\rightarrow\)",
        "←": r"\(\leftarrow\)",
        "↔": r"\(\leftrightarrow\)",
        "⇒": r"\(\Rightarrow\)",
        "⇔": r"\(\Leftrightarrow\)",
        "≤": r"\(\leq\)",
        "≥": r"\(\geq\)",
        "≠": r"\(\neq\)",
        "≈": r"\(\approx\)",
        "≃": r"\(\simeq\)",
        "∼": r"\(\sim\)",
        "×": r"\(\times\)",
        "±": r"\(\pm\)",
        "÷": r"\(\div\)",
        "∈": r"\(\in\)",
        "∉": r"\(\notin\)",
        "∅": r"\(\emptyset\)",
        "∂": r"\(\partial\)",
        "∇": r"\(\nabla\)",
        "∑": r"\(\sum\)",
        "∏": r"\(\prod\)",
        "√": r"\(\sqrt{\ }\)",
        "∞": r"\(\infty\)",
        "∥": r"\(\parallel\)",
        "∣": r"\(\mid\)",
        "∪": r"\(\cup\)",
        "∩": r"\(\cap\)",
        "⊂": r"\(\subset\)",
        "⊆": r"\(\subseteq\)",
        "⊃": r"\(\supset\)",
        "⊇": r"\(\supseteq\)",
        "⊕": r"\(\oplus\)",
        "⊗": r"\(\otimes\)",
        "⊙": r"\(\odot\)",
        "⊥": r"\(\perp\)",
        "∧": r"\(\wedge\)",
        "∨": r"\(\vee\)",
        "¬": r"\(\neg\)",
        "∀": r"\(\forall\)",
        "∃": r"\(\exists\)",
        "∝": r"\(\propto\)",
        "ℓ": r"\(\ell\)",
        "α": r"\(\alpha\)",
        "β": r"\(\beta\)",
        "γ": r"\(\gamma\)",
        "δ": r"\(\delta\)",
        "ε": r"\(\epsilon\)",
        "η": r"\(\eta\)",
        "θ": r"\(\theta\)",
        "κ": r"\(\kappa\)",
        "λ": r"\(\lambda\)",
        "μ": r"\(\mu\)",
        "π": r"\(\pi\)",
        "ρ": r"\(\rho\)",
        "σ": r"\(\sigma\)",
        "τ": r"\(\tau\)",
        "φ": r"\(\phi\)",
        "ϕ": r"\(\phi\)",
        "χ": r"\(\chi\)",
        "ψ": r"\(\psi\)",
        "ω": r"\(\omega\)",
        "Δ": r"\(\Delta\)",
        "Λ": r"\(\Lambda\)",
        "Φ": r"\(\Phi\)",
        "Ψ": r"\(\Psi\)",
        "Ω": r"\(\Omega\)",
        "₂": r"\(_2\)",
        "₀": r"\(_0\)",
        "₁": r"\(_1\)",
        "ₙ": r"\(_n\)",
        "⁺": r"\(^{+}\)",
        "⁻": r"\(^{-}\)",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


MATH_MARKER_RE = re.compile(
    r"""
    (?:
        \\[A-Za-z]+
        | \\[{}]
        | [A-Za-z0-9]+(?:_[A-Za-z0-9{}]+|\^[A-Za-z0-9{}]+)+
        | \b[A-Za-z]\([A-Za-z0-9_{}\\^+\-*/|<>,]+\)
    )
    """,
    re.VERBOSE,
)
EXPLICIT_MATH_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\))", re.DOTALL)
MATH_CHARS = set(r"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_{}\^=+-*/()|<>[]@'")
MATH_SEPARATORS = set(",;:.")
MATH_COMMANDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "sigma",
    "ldots",
    "cdots",
    "mid",
    "in",
    "notin",
    "forall",
    "exists",
    "sum",
    "prod",
    "sqrt",
    "frac",
    "log",
    "exp",
    "text",
}


def math_braces_balanced(text: str) -> bool:
    depth = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def looks_like_inline_math(text: str) -> bool:
    if not math_braces_balanced(text):
        return False
    commands = re.findall(r"\\([A-Za-z]+)", text)
    if commands and not any(command in MATH_COMMANDS for command in commands):
        return False
    if "\\" in text or "_" in text or "^" in text:
        return True
    if re.search(r"\b[A-Za-z]\([^()\s]*[|=<>][^()\s]*\)", text):
        return True
    return False


def normalize_inline_math(text: str) -> str:
    for name in ("alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "sigma"):
        text = re.sub(rf"(?<!\\)\b{name}\b(?=\()", rf"\\{name}", text)
    return text


def is_math_char(ch: str) -> bool:
    return ch in MATH_CHARS


def left_math_atom(text: str, index: int) -> bool:
    if index < 0:
        return False
    ch = text[index]
    if ch == "\\":
        return True
    if ch in ")]}":
        return True
    if ch.isdigit():
        return True
    if ch.isalpha():
        start = index
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        word = text[start : index + 1]
        return start > 0 and text[start - 1] == "\\" or len(word) == 1
    return ch in "_^+-*/=|<>"


def right_math_atom(text: str, index: int) -> bool:
    if index >= len(text):
        return False
    ch = text[index]
    if ch == "\\":
        return True
    if ch in "([{":
        return True
    if ch.isdigit():
        return True
    if ch.isalpha():
        end = index
        while end + 1 < len(text) and text[end + 1].isalpha():
            end += 1
        word = text[index : end + 1]
        following = text[end + 1] if end + 1 < len(text) else ""
        return len(word) == 1 or following in "_^("
    return ch in "_^+-*/=|<>"


def next_nonspace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def prev_nonspace(text: str, index: int) -> int:
    while index >= 0 and text[index].isspace():
        index -= 1
    return index


def should_include_separator(text: str, start: int, end: int) -> bool:
    left = prev_nonspace(text, start - 1)
    right = next_nonspace(text, end)
    return left_math_atom(text, left) and right_math_atom(text, right)


def expand_math_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0:
        ch = text[start - 1]
        if is_math_char(ch):
            start -= 1
        elif ch.isspace():
            new_start = prev_nonspace(text, start - 1) + 1
            if should_include_separator(text, new_start, start):
                start = new_start
            else:
                break
        elif ch in MATH_SEPARATORS and should_include_separator(text, start - 1, start):
            start -= 1
        else:
            break

    while end < len(text):
        ch = text[end]
        if is_math_char(ch):
            end += 1
        elif ch.isspace():
            new_end = next_nonspace(text, end)
            if should_include_separator(text, end, new_end):
                end = new_end
            else:
                break
        elif ch in MATH_SEPARATORS and should_include_separator(text, end, end + 1):
            end += 1
        else:
            break
    return start, end


def emit_inline_segment(segment: str, out: list[str]) -> None:
    last = 0
    for match in MATH_MARKER_RE.finditer(segment):
        if match.start() < last:
            continue
        start, end = expand_math_span(segment, *match.span())
        token = segment[start:end].strip()
        if not looks_like_inline_math(token):
            continue
        out.append(esc(segment[last:start]))
        out.append(r"\(" + normalize_inline_math(token) + r"\)")
        last = end
    out.append(esc(segment[last:]))


def emit_explicit_math(token: str, out: list[str]) -> None:
    if token.startswith(r"\(") and token.endswith(r"\)"):
        out.append(token)
    elif token.startswith("$$") and token.endswith("$$"):
        out.append(r"\[" + token[2:-2].strip() + r"\]")
    elif token.startswith("$") and token.endswith("$"):
        out.append(r"\(" + token[1:-1].strip() + r"\)")
    else:
        out.append(esc(token))


def esc_rich(text: object) -> str:
    """Escape prose while preserving lightweight inline LaTeX math snippets."""
    s = "" if text is None else str(text)
    out: list[str] = []
    last = 0
    for match in EXPLICIT_MATH_RE.finditer(s):
        start, end = match.span()
        emit_inline_segment(s[last:start], out)
        emit_explicit_math(match.group(0), out)
        last = end
    emit_inline_segment(s[last:], out)
    return "".join(out)


def normalize_formula(formula: str | None) -> str | None:
    if not formula:
        return None
    formula = formula.strip()
    formula = re.sub(r"^\$\$|\$\$?$", "", formula).strip()
    formula = formula.replace("\n\n", "\n")
    return formula


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, text=True)


def pdf_for(bp: dict) -> Path:
    source = bp.get("source_pdf")
    if not source:
        raise ValueError(f"{bp.get('lecture', '<unknown>')} missing source_pdf")
    pdf = LECTURES / source
    if not pdf.exists():
        raise FileNotFoundError(f"source_pdf not found: {pdf}")
    return pdf


def tex_document(bp: dict) -> str:
    pdf = pdf_for(bp)
    pdf_path = pdf.as_posix()
    title = esc(bp.get("title", bp["lecture"]))
    lecture = esc(bp["lecture"])
    concept_chain = " $\\rightarrow$ ".join(esc(x) for x in bp.get("concept_chain", []))
    sections = bp.get("sections", [])
    quick = bp.get("quick_revision", [])
    slides = bp.get("slides", [])

    parts: list[str] = []
    parts.append(r"""\documentclass[11pt,a4paper]{ctexart}
\usepackage[margin=18mm]{geometry}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{fontspec}
\usepackage{amsmath}
\usepackage{unicode-math}
\usepackage[most]{tcolorbox}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{needspace}
\setCJKmainfont{Songti SC}
\setCJKsansfont{Hiragino Sans GB}
\setmainfont{Avenir Next}
\setsansfont{Avenir Next}
\setmathfont{STIX Two Math}
\definecolor{ttblue}{HTML}{25508E}
\definecolor{ttmuted}{HTML}{5C6069}
\definecolor{ttlight}{HTML}{E8F0FC}
\definecolor{ttaccent}{HTML}{BE502D}
\definecolor{ttwarm}{HTML}{FFF6E8}
\titleformat{\section}{\Large\bfseries\sffamily\color{ttblue}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\sffamily\color{ttblue}}{}{0pt}{}
\setlist[itemize]{leftmargin=1.5em,itemsep=0.22em,topsep=0.25em}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt
\emergencystretch=3em
\sloppy
\pagestyle{empty}
\newtcolorbox{chainbox}{colback=ttlight,colframe=ttblue!20,boxrule=0.6pt,arc=2mm,left=2mm,right=2mm,top=1mm,bottom=1mm}
\newtcolorbox{callout}[1]{colback=ttwarm,colframe=ttaccent!45,title={#1},fonttitle=\sffamily\bfseries\color{ttaccent},boxrule=0.7pt,arc=2mm,left=2.5mm,right=2.5mm,top=1.5mm,bottom=1.5mm}
\newcommand{\slideimage}[3]{%
\begin{center}
\includegraphics[page=#1,width=#2\linewidth]{#3}\\[-2pt]
{\footnotesize\color{ttmuted}原 PPT 第 #1 页}
\end{center}}
\begin{document}
""")
    parts.append(f"{{\\sffamily\\color{{ttmuted}}{lecture}}}\\\\[3pt]\n")
    parts.append(f"{{\\Huge\\sffamily\\bfseries\\color{{ttblue}}{title}}}\\\\[6pt]\n")
    parts.append("图文讲义版：用原 PPT 作为视觉锚点，按“概念故事线 + 直觉解释 + 考试速记”整理。\\\\[6pt]\n")
    if concept_chain:
        parts.append("\\begin{chainbox}\n\\sffamily " + concept_chain + "\n\\end{chainbox}\n")
    first_slide = slides[0]["page"] if slides else 1
    parts.append(f"\\slideimage{{{int(first_slide)}}}{{0.72}}{{{pdf_path}}}\n")
    parts.append("\\begin{callout}{这节课一句话}\n" + esc_rich(bp.get("one_sentence", "")) + "\n\\end{callout}\n")

    for idx, section in enumerate(sections, start=1):
        parts.append("\\Needspace{0.38\\textheight}\n")
        parts.append(f"\\section{{{idx}. {esc(section.get('title', ''))}}}\n")
        slide_page = section.get("slide_page")
        if slide_page:
            parts.append(f"\\slideimage{{{int(slide_page)}}}{{0.62}}{{{pdf_path}}}\n")
        parts.append("\\begin{callout}{人话直觉}\n" + esc_rich(section.get("intuition", "")) + "\n\\end{callout}\n")
        parts.append("\\noindent\\textbf{精确定义 / 机制：} " + esc_rich(section.get("explanation", "")) + "\n\n")
        formula = normalize_formula(section.get("formula_latex"))
        if formula:
            parts.append("\\[\n" + formula + "\n\\]\n")
        parts.append("\\noindent\\textbf{考试问法：} " + esc_rich(section.get("exam_angle", "")) + "\n\n")

    parts.append("\\section{考试速记版}\n\\begin{itemize}\n")
    for item in quick:
        parts.append("\\item " + esc_rich(item) + "\n")
    parts.append("\\end{itemize}\n")
    parts.append("\\begin{callout}{一段稳妥考场回答}\n" + esc_rich(bp.get("exam_answer", "")) + "\n\\end{callout}\n")
    parts.append("\\end{document}\n")
    return "".join(parts)


def render(blueprint: Path) -> Path:
    bp = json.loads(blueprint.read_text(encoding="utf-8"))
    safe = blueprint.stem
    build = BUILD_DIR / safe
    build.mkdir(parents=True, exist_ok=True)
    tex = build / f"{safe}.tex"
    tex.write_text(tex_document(bp), encoding="utf-8")
    run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=build)
    pdf = build / f"{safe}.pdf"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{safe}-tutor-handout.pdf"
    shutil.copy2(pdf, out)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for stale in PREVIEW_DIR.glob(f"{out.stem}-*.png"):
        stale.unlink()
    run(["pdftoppm", "-png", "-f", "1", "-l", "3", "-r", "140", str(out), str(PREVIEW_DIR / out.stem)], cwd=ROOT)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blueprints", nargs="+")
    args = parser.parse_args()
    for item in args.blueprints:
        out = render(Path(item))
        info = subprocess.check_output(["pdfinfo", str(out)], text=True)
        pages = re.search(r"Pages:\s+(\d+)", info)
        print(f"{out} pages={pages.group(1) if pages else '?'} size={out.stat().st_size}")


if __name__ == "__main__":
    main()
