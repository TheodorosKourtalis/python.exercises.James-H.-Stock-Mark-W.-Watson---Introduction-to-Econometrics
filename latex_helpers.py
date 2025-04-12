import re
import tempfile
import os
import subprocess
import shutil

def markdown_to_latex_fixed(md_text: str) -> str:
    """
    Converts Markdown-formatted text to LaTeX.
    - Converts **bold** to \textbf{...} and *italic* to \textit{...} in non-math parts.
    - Leaves math segments (delimited by $...$ or $$...$$) untouched.
    - Escapes LaTeX special characters in non-math text.
    """
    # Function to escape special characters (except for curly braces and backslashes already used)
    def escape_text(text: str) -> str:
        special_chars = {
            '&':  r'\&',
            '%':  r'\%',
            '#':  r'\#',
            '_':  r'\_',
            '~':  r'\textasciitilde{}',
            '^':  r'\^{}'
        }
        for ch, repl in special_chars.items():
            text = text.replace(ch, repl)
        return text

    # Split text into math and non-math segments
    segments = re.split(r'(\$\$.*?\$\$|\$.*?\$)', md_text, flags=re.DOTALL)
    output_segments = []
    for seg in segments:
        if seg.startswith('$'):
            # It's a math segment; leave it unchanged.
            output_segments.append(seg)
        else:
            # Process non-math text.
            seg = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', seg)  # Bold
            seg = re.sub(r'\*(.+?)\*', r'\\textit{\1}', seg)       # Italic
            seg = escape_text(seg)
            output_segments.append(seg)
    return ''.join(output_segments)

def generate_latex_document(md_content: str) -> str:
    """
    Wraps the converted LaTeX content inside a minimal LaTeX document.
    """
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

def generate_pdf_with_pdflatex(sample_md: str) -> bytes:
    """
    Converts the given Markdown to a complete LaTeX document using the above helper,
    then compiles it with pdflatex, returning the PDF as bytes.
    """
    latex_source = generate_latex_document(sample_md)
    tmp_dir = tempfile.mkdtemp()
    try:
        tex_filename = os.path.join(tmp_dir, "document.tex")
        pdf_filename = os.path.join(tmp_dir, "document.pdf")
        with open(tex_filename, "w", encoding="utf-8") as f:
            f.write(latex_source)
        # Run pdflatex with nonstop mode
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise Exception("pdflatex error:\n" + result.stderr.decode("utf-8"))
        with open(pdf_filename, "rb") as f:
            pdf_data = f.read()
        return pdf_data
    finally:
        shutil.rmtree(tmp_dir)

def show_sample_answer(sample_md: str) -> None:
    """
    Checks the global small_screen flag (stored in st.session_state). If True, creates
    a PDF using pdflatex and shows a download button; otherwise, displays the sample answer as Markdown.
    """
    import streamlit as st
    if st.session_state.get("small_screen", False):
        try:
            pdf_bytes = generate_pdf_with_pdflatex(sample_md)
            st.download_button(
                label="Download Sample Answer PDF",
                data=pdf_bytes,
                file_name="sample_answer.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(str(e))
    else:
        st.markdown(sample_md)
