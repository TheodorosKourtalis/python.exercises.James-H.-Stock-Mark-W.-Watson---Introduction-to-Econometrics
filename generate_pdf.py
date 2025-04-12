#!/usr/bin/env python3
import sys
import tempfile
import subprocess
import os
import shutil

def main():
    sample_text = sys.stdin.read()
    tmp_dir = tempfile.mkdtemp()
    try:
        tex_path = os.path.join(tmp_dir, "doc.tex")
        pdf_path = os.path.join(tmp_dir, "doc.pdf")
        # Create a minimal LaTeX document using the provided source.
        latex_content = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
%s
\end{document}
""" % sample_text
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        # Run pdflatex
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path],
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode("utf-8"))
            sys.exit(proc.returncode)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        sys.stdout.buffer.write(pdf_bytes)
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
