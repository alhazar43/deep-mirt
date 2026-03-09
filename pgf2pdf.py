#!/usr/bin/env python3
"""Convert PGF figures to tightly cropped standalone PDFs.

Strategy: compile PGF with pdflatex on a large page, then use PyPDF2
to crop the PDF mediabox to the exact PGF bounding-box dimensions.
"""

import subprocess, tempfile, shutil, re, os
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter

FIGURES = {
    "learner_trajectories": "kt-gpcm/outputs/q200_k4_static_item/trajectory_plots/learner_trajectories.pgf",
    "recovery_k4_student":  "kt-gpcm/outputs/large_q200_k4_dkvmn_ordinal_student.pgf",
    "recovery_k4_item":     "kt-gpcm/outputs/large_q200_k4_dkvmn_ordinal_item.pgf",
    "recovery_k3_student":  "kt-gpcm/outputs/large_q200_k3_dkvmn_ordinal/recovery_split_student.pgf",
    "recovery_k3_item":     "kt-gpcm/outputs/large_q200_k3_dkvmn_ordinal/recovery_split_item.pgf",
    "recovery_k5_student":  "kt-gpcm/outputs/large_q200_k5_dkvmn_ordinal_student.pgf",
    "recovery_k5_item":     "kt-gpcm/outputs/large_q200_k5_dkvmn_ordinal_item.pgf",
    "recovery_k6_student":  "kt-gpcm/outputs/large_q200_k6_dkvmn_ordinal/recovery_split_student.pgf",
    "recovery_k6_item":     "kt-gpcm/outputs/large_q200_k6_dkvmn_ordinal/recovery_split_item.pgf",
}

PROJECT = Path(__file__).resolve().parent
OUTDIR = PROJECT / "figures"

# Simple wrapper — compile on default page, crop later with PyPDF2
WRAPPER = r"""\documentclass{article}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{amsmath,amssymb}
\def\mathdefault#1{#1}
\pagestyle{empty}
\setlength{\topmargin}{-1in}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0pt}
\setlength{\oddsidemargin}{-1in}
\setlength{\evensidemargin}{-1in}
\setlength{\topskip}{0pt}
\setlength{\footskip}{0pt}
\setlength{\parindent}{0pt}
\begin{document}
\noindent\input{figure.pgf}
\end{document}
"""


def get_bbox(pgf_path):
    """Extract bounding box (width, height in inches) from PGF file."""
    text = pgf_path.read_text()
    # Look for pgfpathrectangle with pgfqpoint giving width/height
    m = re.search(r'pgfpathrectangle.*?pgfqpoint\{([\d.]+)in\}\{([\d.]+)in\}', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # Fallback: scan bounding box clip path
    points = re.findall(r'pgfqpoint\{([\d.]+)in\}\{([\d.]+)in\}', text[:2000])
    if points:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return max(xs), max(ys)
    return None, None


def crop_pdf(pdf_path, w_in, h_in, dest_path):
    """Crop PDF mediabox to w_in x h_in (content at top-left origin)."""
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    page = reader.pages[0]

    # PGF content is placed at (0, page_height - h) in PDF coords
    # (TeX origin is top-left, PDF origin is bottom-left)
    page_h = float(page.mediabox.height)
    w_pt = w_in * 72
    h_pt = h_in * 72

    # Content starts at bottom-left = (0, page_h - h_pt), top-right = (w_pt, page_h)
    page.mediabox.lower_left = (0, page_h - h_pt)
    page.mediabox.upper_right = (w_pt, page_h)
    # Also set cropbox to ensure viewers respect it
    page.cropbox = page.mediabox

    writer.add_page(page)
    with open(str(dest_path), "wb") as f:
        writer.write(f)


def convert(name, pgf_rel):
    src = PROJECT / pgf_rel
    if not src.exists():
        print(f"  SKIP {name}: not found")
        return False

    w, h = get_bbox(src)
    if w is None:
        print(f"  SKIP {name}: cannot determine bbox")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(src, tmp / "figure.pgf")
        (tmp / "wrapper.tex").write_text(WRAPPER)
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "wrapper.tex"],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        pdf = tmp / "wrapper.pdf"
        if pdf.exists():
            dest = OUTDIR / f"{name}.pdf"
            crop_pdf(pdf, w, h, dest)
            print(f"  OK  {name}.pdf  ({w}x{h}in, {dest.stat().st_size//1024} KB)")
            return True
        else:
            print(f"  FAIL {name}")
            log = tmp / "wrapper.log"
            if log.exists():
                for l in log.read_text().splitlines()[-10:]:
                    print(f"       {l}")
            return False


if __name__ == "__main__":
    os.environ["PATH"] = "/usr/bin:/home/yuan/.TinyTeX/bin/x86_64-linux:" + os.environ.get("PATH", "")
    OUTDIR.mkdir(exist_ok=True)
    ok = fail = 0
    for name, path in FIGURES.items():
        print(f"Converting {name}...")
        if convert(name, path):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} converted, {fail} failed")
