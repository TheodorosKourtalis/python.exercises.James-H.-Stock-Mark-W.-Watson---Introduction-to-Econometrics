import re
import tempfile
import os
import subprocess
import shutil
import streamlit as st

def markdown_to_latex_fixed(md_text: str) -> str:
    """
    Converts Markdown-formatted text to LaTeX.
    - Preserves math segments (delimited by $$...$$ or $...$).
    - Converts **bold** to \textbf{...} and *italic* to \textit{...}.
    - Escapes special characters in non-math segments.
    """
    def escape_text(text: str) -> str:
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '#': r'\#',
            '_': r'\_',
            '~': r'\textasciitilde{}',
            '^': r'\^{}'
        }
        for ch, repl in special_chars.items():
            text = text.replace(ch, repl)
        return text

    segments = re.split(r'(\$\$.*?\$\$|\$.*?\$)', md_text, flags=re.DOTALL)
    output_segments = []
    for seg in segments:
        if seg.startswith('$'):
            output_segments.append(seg)
        else:
            seg = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', seg)
            seg = re.sub(r'\*(.+?)\*', r'\\textit{\1}', seg)
            seg = escape_text(seg)
            output_segments.append(seg)
    return ''.join(output_segments)

def generate_latex_document(md_content: str) -> str:
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

def generate_pdf_with_pdflatex(sample_md: str, pdf_base_name: str = "document") -> bytes:
    """
    Generates a PDF from sample_md using pdflatex.
    The temporary .tex and .pdf file names are built using pdf_base_name.
    """
    latex_source = generate_latex_document(sample_md)
    tmp_dir = tempfile.mkdtemp()
    try:
        tex_filename = os.path.join(tmp_dir, pdf_base_name + ".tex")
        pdf_filename = os.path.join(tmp_dir, pdf_base_name + ".pdf")
        with open(tex_filename, "w", encoding="utf-8") as f:
            f.write(latex_source)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            cwd=tmp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise Exception("pdflatex error:\n" + result.stderr.decode("utf-8"))
        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes
    finally:
        shutil.rmtree(tmp_dir)

def show_sample_answer(sample_md: str, key_suffix: str = "") -> None:
    """
    Displays the sample answer.
    If st.session_state["small_screen"] is True, it generates a PDF using pdflatex and shows a download button.
    
    If key_suffix is empty, it uses st.session_state["current_pdf_label"] as the base for the PDF file name.
    """
    if key_suffix == "":
        chapter_label = st.session_state.get("current_pdf_label", "sample_answer")
        key_suffix_used = chapter_label
    else:
        key_suffix_used = key_suffix

    if st.session_state.get("small_screen", False):
        try:
            pdf_bytes = generate_pdf_with_pdflatex(sample_md, pdf_base_name=st.session_state.get("current_pdf_label", "sample_answer"))
            if pdf_bytes:
                st.download_button(
                    label="Download Sample Answer PDF",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.get('current_pdf_label', 'sample_answer')}.pdf",
                    mime="application/pdf",
                    key="download_sample_" + key_suffix_used
                )
        except Exception as e:
            st.error(str(e))
    else:
        st.markdown(sample_md)
