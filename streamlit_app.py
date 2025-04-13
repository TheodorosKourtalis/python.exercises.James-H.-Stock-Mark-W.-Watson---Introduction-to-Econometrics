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
st.subheader("Created by Thodoris Kourtalis")

# -----------------------------------------------------------
# INTRODUCTORY REMARKS
# -----------------------------------------------------------
st.markdown("""
I built this collection of **econometrics exercises** from my own notes, 
with a **significant portion** inspired by the textbook 
[“Introduction to Econometrics”](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000006421/9780136879787).  
Though I claim **no special rights** over these ideas, I'm sharing them 
as a personal project, in hopes they support your learning process.

---

### How This Website Works

1. **Pages for Each Chapter or Topic**  
   - In the **top-left corner** (or via the navigation sidebar), you'll see a **Pages** menu.
   - Each **page** contains a different set of exercises.

2. **Your Responses**  
   - Each exercise page provides:
     - A **Question** or prompt.
     - A text area for you to type your own answer.
     - A **Show Sample Answer** expander, which reveals a reference solution.

3. **Small-Screen Mode**  
   - Use the checkbox labeled **“I’m on a small screen”** in the sidebar if you’re on a phone or narrow display.
   - When checked, solutions are generated as **PDF** files for download instead of being displayed inline.

Feel free to explore, experiment, and learn!
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
st.markdown("*Use the **Pages** menu to access specific exercises.*")
