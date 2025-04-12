import streamlit as st
import subprocess

# Global setup on your main page:
st.set_page_config(page_title="Econometrics Exercises Notebook", layout="wide")
st.title("Econometrics Exercises Notebook")
st.caption("By Thodoris Kourtalis")

if "small_screen" not in st.session_state:
    st.session_state["small_screen"] = False

st.sidebar.markdown("### Display Options")
small_screen = st.sidebar.checkbox("I'm on a small screen", value=st.session_state["small_screen"])
st.session_state["small_screen"] = small_screen

# Helper function to generate a PDF by calling the external script.
def generate_pdf_via_subprocess(sample_text):
    """
    Calls the external generate_pdf.py script via subprocess, passing the sample_text
    on stdin. Returns the PDF data as bytes.
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

def show_sample_answer(sample_md):
    """
    Checks the global small_screen flag.  
    If enabled, it generates a temporary PDF from the sample_md and displays a download button.
    Otherwise, it displays the sample_md interactively with custom CSS.
    """
    if st.session_state.get("small_screen", False):
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
