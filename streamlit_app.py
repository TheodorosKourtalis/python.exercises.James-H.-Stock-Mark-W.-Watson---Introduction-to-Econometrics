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
# TITLE & SUBHEADER
# -----------------------------------------------------------
st.title("Econometrics Exercises Notebook")
st.caption("by Thodoris Kourtalis")

# -----------------------------------------------------------
# INTRODUCTORY REMARKS
# -----------------------------------------------------------
st.markdown("""
A personal collection of **econometrics exercises**, assembled out of passion. 

Certain problems are **inspired** or loosely **transferred** from external sources—
notably [“Introduction to Econometrics”](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000006421/9780136879787) so I try to reference them as much as possible.

The **solutions** and the **website** you see here are entirely my own work so please if you find any mistakes contact me at theodoroskourtalisgithub@gmail.com.

If you’re curious about how the exercises are run or how this website is built, 
feel free to browse the **open-source code** available on my connected 
GitHub repository.

---

### How to Use This Site

1. **Pages for Each Topic**  
   In the top-left corner (or via your Streamlit sidebar), you will see a **Pages** menu.  
   Select a page to view a set of exercises.

2. **Answering & Sample Solutions**  
   Each page has:
   - A question prompt  
   - A text area for your own answer  
   - A “Show Sample Answer” expander with a reference solution

**2.5 – Interactive Exercises**  
Some pages include interactive elements, where you can adjust parameters and see the results in real time.

3. **Small-Screen Mode**  
   If you’re on a mobile device or narrow display, enable the **“I’m on a small screen”** checkbox in the sidebar.  
   This mode renders solutions as **PDFs** to download, rather than inline text.

Enjoy exploring, and I hope it fosters a deeper understanding of econometrics!
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
    Calls the external generate_pdf.py script, passing 'sample_text' on stdin.
    Returns the PDF bytes if successful, else None.
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
    If 'small_screen' is True, generates a PDF of sample_md for download.
    Otherwise, displays sample_md inline.
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
# END NOTE
# -----------------------------------------------------------
st.markdown("---")
st.markdown("*Use the **Bullet** menu of each chapter to access specific exercises.*")
