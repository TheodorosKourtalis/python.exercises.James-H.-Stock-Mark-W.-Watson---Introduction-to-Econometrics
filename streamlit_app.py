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
# TITLE & BYLINE
# -----------------------------------------------------------
st.title("Econometrics Exercises Notebook")
st.subheader("Curated by Thodoris Kourtalis")

# -----------------------------------------------------------
# DISCALIMER & GUIDANCE
# -----------------------------------------------------------
st.markdown("""
In this notebook, you will find **exercises and solutions** inspired by standard  
econometrics syllabi. **All content is purely for educational demonstration.**  

I hold **no proprietary rights** to the concepts or methods described within. 
For rigorous coverage, please consider purchasing the textbook 
[“Introduction to Econometrics”](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000006421/9780136879787).

---

### How to Navigate
1. **Streamlit Pages**  
   Use the **Pages** panel (usually found in the upper-left menu or as separate pages) to switch between the different *chapters* or *sections*. Each page hosts a unique collection of exercises.

2. **Small-Screen Mode**  
   If you’re on a mobile device or narrow display, enable the “**I’m on a small screen**” option in the sidebar (see below). Instead of revealing solutions inline, this mode will generate **PDF** files for you to download.

3. **Exercises & Solutions**  
   - Each exercise page includes a **question prompt** (or multiple prompts).  
   - You may type your own solution in the designated text area.  
   - **Show Sample Answer** reveals (or lets you download) a reference solution, which often includes both text and \\(\\LaTeX\\) math.

---

We trust you will find this resource beneficial as you explore and master the fundamentals 
of econometrics.
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
    passing the sample_text on stdin. Returns the PDF bytes.
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
        st.error("Exception during PDF generation: " + str(e))
        return None

# -----------------------------------------------------------
# SAMPLE ANSWER RENDERING
# -----------------------------------------------------------
def show_sample_answer(sample_md):
    """
    Checks the global 'small_screen' flag.
    If enabled, generates a PDF from sample_md and offers it for download.
    Otherwise, displays sample_md inline.
    """
    if st.session_state.get("small_screen", False):
        # Generate PDF and provide as download
        pdf_data = generate_pdf_via_subprocess(sample_md)
        if pdf_data is not None:
            st.download_button(
                label="Download Sample Answer PDF",
                data=pdf_data,
                file_name="sample_answer.pdf",
                mime="application/pdf"
            )
    else:
        # Inline display with custom styling
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
# END: This script is primarily for the main page.
# Individual exercises and additional content can be placed in separate pages.
# -----------------------------------------------------------
