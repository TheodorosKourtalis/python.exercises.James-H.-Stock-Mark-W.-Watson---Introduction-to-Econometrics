import streamlit as st
import subprocess

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="Econometrics Exercises Notebook",
    layout="wide"
)

# -----------------------------------------------------------
# TITLE & SUBHEADER (PASSION PROJECT)
# -----------------------------------------------------------
st.title("Econometrics Exercises Notebook")
st.subheader("A Passion Project by Thodoris Kourtalis")

# -----------------------------------------------------------
# INTRODUCTORY REMARKS
# -----------------------------------------------------------
st.markdown("""
Welcome to this **Econometrics Exercises** repository, assembled out of pure enthusiasm 
for sharing knowledge. Some exercises have been **originally crafted or inspired** 
by standard econometrics references, while **others are directly transferred** 
(or adapted) from external sources.  

No special rights are claimed over the material; it is provided as **educational demonstration** only.  
Be aware that **some solutions or questions may be incomplete or incorrect**.  
We encourage users to verify and cross-check with reputable references—particularly 
the textbook [“Introduction to Econometrics”](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000006421/9780136879787) for a rigorous treatment.

---
### How to Navigate
- **Streamlit Pages**:  
  Each chapter or set of exercises is on a separate **Page** in this application.  
  Access them from the top-left menu or your Streamlit sidebar, depending on your interface.

- **Small-Screen Mode**:  
  If you’re on a mobile device or prefer a simpler layout, enable “I’m on a small screen” in the sidebar (see below).  
  This mode allows you to download sample answers as **PDF** files instead of viewing them inline.

Thank you for exploring this **passion project**—I hope it aids your study and sparks new insights.
""")

# -----------------------------------------------------------
# SIDEBAR: SMALL SCREEN FLAG
# -----------------------------------------------------------
if "small_screen" not in st.session_state:
    st.session_state["small_screen"] = False

st.sidebar.markdown("### Display Options")
small_screen_checkbox = st.sidebar.checkbox(
    "I'm on a small screen", 
    value=st.session_state["small_screen"]
)
st.session_state["small_screen"] = small_screen_checkbox

# -----------------------------------------------------------
# PDF GENERATION FUNCTION
# -----------------------------------------------------------
def generate_pdf_via_subprocess(sample_text):
    """
    Calls the external generate_pdf.py script via subprocess,
    passing 'sample_text' on stdin. Returns the PDF bytes.
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
            st.error("Error generating PDF:\n" + err.decode('utf-8'))
            return None
        return pdf_data
    except Exception as e:
        st.error(f"Exception during PDF generation: {str(e)}")
        return None

# -----------------------------------------------------------
# SAMPLE ANSWER RENDERING FUNCTION
# -----------------------------------------------------------
def show_sample_answer(sample_md):
    """
    Checks 'small_screen' in st.session_state.
    If True, generates a PDF from 'sample_md' and offers a download.
    Otherwise, displays the Markdown inline with custom styling.
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
                line-height: 1.5;
                word-wrap: break-word;
                overflow-x: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="sample-answer">', unsafe_allow_html=True)
        st.markdown(sample_md)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# END: All specialized exercises or pages can be accessed
# via Streamlit's "Pages" feature, not in this file.
# -----------------------------------------------------------
