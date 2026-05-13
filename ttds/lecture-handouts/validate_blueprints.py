#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/huez/Documents/ttds")
LECTURES = ROOT / "lec"
REQUIRED = ["lecture", "source_pdf", "title", "one_sentence", "concept_chain", "slides", "sections", "quick_revision", "exam_answer"]
SECTION_REQUIRED = ["title", "slide_page", "intuition", "explanation", "exam_angle"]


def pdf_pages(path: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise RuntimeError(f"Cannot read page count for {path}")


def check(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        bp = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid JSON: {exc}"]
    for key in REQUIRED:
        if key not in bp:
            errors.append(f"missing key {key}")
    if errors:
        return errors
    pdf = LECTURES / bp["source_pdf"]
    if not pdf.exists():
        errors.append(f"missing source_pdf {pdf}")
        pages = 0
    else:
        pages = pdf_pages(pdf)
    if not isinstance(bp.get("concept_chain"), list) or len(bp["concept_chain"]) < 4:
        errors.append("concept_chain too short")
    if not isinstance(bp.get("slides"), list) or len(bp["slides"]) < 4:
        errors.append("slides too few")
    if not isinstance(bp.get("sections"), list) or len(bp["sections"]) < 6:
        errors.append("sections too few")
    if not isinstance(bp.get("quick_revision"), list) or len(bp["quick_revision"]) < 5:
        errors.append("quick_revision too few")
    for idx, slide in enumerate(bp.get("slides", []), 1):
        page = slide.get("page") if isinstance(slide, dict) else None
        if not isinstance(page, int) or page < 1 or (pages and page > pages):
            errors.append(f"slide {idx} bad page {page}; pages={pages}")
        for key in ("reason", "caption"):
            if not isinstance(slide, dict) or not str(slide.get(key, "")).strip():
                errors.append(f"slide {idx} missing {key}")
    for idx, section in enumerate(bp.get("sections", []), 1):
        if not isinstance(section, dict):
            errors.append(f"section {idx} not object")
            continue
        for key in SECTION_REQUIRED:
            if key not in section or not str(section.get(key, "")).strip():
                errors.append(f"section {idx} missing {key}")
        page = section.get("slide_page")
        if not isinstance(page, int) or page < 1 or (pages and page > pages):
            errors.append(f"section {idx} bad slide_page {page}; pages={pages}")
    return errors


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]] or sorted((ROOT / "lecture-handouts/blueprints").glob("*.json"))
    failed = False
    for path in paths:
        errors = check(path)
        if errors:
            failed = True
            print(f"FAIL {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
