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
    parts.append("\\begin{callout}{这节课一句话}\n" + esc(bp.get("one_sentence", "")) + "\n\\end{callout}\n")

    for idx, section in enumerate(sections, start=1):
        parts.append("\\Needspace{0.38\\textheight}\n")
        parts.append(f"\\section{{{idx}. {esc(section.get('title', ''))}}}\n")
        slide_page = section.get("slide_page")
        if slide_page:
            parts.append(f"\\slideimage{{{int(slide_page)}}}{{0.62}}{{{pdf_path}}}\n")
        parts.append("\\begin{callout}{人话直觉}\n" + esc(section.get("intuition", "")) + "\n\\end{callout}\n")
        parts.append("\\noindent\\textbf{精确定义 / 机制：} " + esc(section.get("explanation", "")) + "\n\n")
        formula = normalize_formula(section.get("formula_latex"))
        if formula:
            parts.append("\\[\n" + formula + "\n\\]\n")
        parts.append("\\noindent\\textbf{考试问法：} " + esc(section.get("exam_angle", "")) + "\n\n")

    parts.append("\\section{考试速记版}\n\\begin{itemize}\n")
    for item in quick:
        parts.append("\\item " + esc(item) + "\n")
    parts.append("\\end{itemize}\n")
    parts.append("\\begin{callout}{一段稳妥考场回答}\n" + esc(bp.get("exam_answer", "")) + "\n\\end{callout}\n")
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
