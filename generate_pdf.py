import subprocess
import io
from fpdf import FPDF  # Ensure you have fpdf2 installed: pip install fpdf2

def generate_pdf_via_subprocess(sample_text):
    """
    Calls the external generate_pdf.py script via subprocess, passing sample_text via stdin.
    Returns the PDF data as bytes.
    """
    try:
        process = subprocess.Popen(
            ['python', 'generate_pdf.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        pdf_data, err = process.communicate(input=sample_text.encode('utf-8'))
        if process.returncode != 0:
            st.error("Error generating PDF: " + err.decode('utf-8'))
            return None
        return pdf_data
    except Exception as e:
        st.error("Exception during PDF generation: " + str(e))
        return None

def show_sample_answer(sample_md, key_suffix="default"):
    """
    Displays the interactive sample answer by default.
    If the user checks the "Show PDF version instead" checkbox,
    then it generates and displays a download button for a PDF.
    The key_suffix parameter provides a unique key for the checkbox.
    """
    show_pdf = st.checkbox("Show PDF version instead", key=f"pdf_toggle_{key_suffix}")
    if show_pdf:
        pdf_data = generate_pdf_via_subprocess(sample_md)
        if pdf_data is not None:
            st.download_button(
                label="Download Sample Answer PDF",
                data=pdf_data,
                file_name="sample_answer.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown(
            """
            <style>
            .sample-answer {
                width: 95%;
                max-width: 100%;
                margin: 0 auto;
                text-align: left;
                font-size: 1rem;
                line-height: 1.4;
                word-wrap: break-word;
                overflow-x: auto;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<div class="sample-answer">', unsafe_allow_html=True)
        st.markdown(sample_md)
        st.markdown("</div>", unsafe_allow_html=True)
