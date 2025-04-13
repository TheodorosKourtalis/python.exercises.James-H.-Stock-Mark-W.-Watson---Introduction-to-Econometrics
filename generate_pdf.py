#!/usr/bin/env python3
import sys
import os
import re
import tempfile
import subprocess
import shutil

def escape_text(text: str) -> str:
    """Escapes LaTeX special characters in non-math text."""
    special_chars = {
        '&':  r'\&',
        '%':  r'\%',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\textasciitilde{}',
        '^':  r'\^{}',
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text

def markdown_to_latex_fixed(md_text: str) -> str:
    """
    Converts Markdown to LaTeX:
      - Converts **bold** to \textbf{}
      - Converts *italic* to \textit{}
      - Leaves math environments ($...$ or $$...$$) untouched
      - Escapes LaTeX special characters only in non-math parts.
    """
    # Split the input into math and non-math segments:
    segments = re.split(r'(\$\$.*?\$\$|\$.*?\$)', md_text, flags=re.DOTALL)
    output_segments = []
    for seg in segments:
        if seg.startswith('$'):
            # Math segment: leave unchanged.
            output_segments.append(seg)
        else:
            # Convert Markdown styling in non-math text.
            seg = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', seg)
            seg = re.sub(r'\*(.+?)\*', r'\\textit{\1}', seg)
            # Escape remaining special characters.
            seg = escape_text(seg)
            output_segments.append(seg)
    return ''.join(output_segments)

def generate_latex_document(md_content: str) -> str:
    """Wraps the converted LaTeX content in a minimal LaTeX document."""
    converted = markdown_to_latex_fixed(md_content)
    document = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}
%s
\end{document}""" % converted
    return document

def main():
    md_input = sys.stdin.read()
    latex_source = generate_latex_document(md_input)
    tmp_dir = tempfile.mkdtemp()
    try:
        tex_file = os.path.join(tmp_dir, "doc.tex")
        pdf_file = os.path.join(tmp_dir, "doc.pdf")
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(latex_source)
        # Run pdflatex
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode("utf-8"))
            sys.exit(proc.returncode)
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()
        sys.stdout.buffer.write(pdf_bytes)
    finally:
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
