import streamlit as st
import subprocess

# -----------------------------------------------------------
# PAGE & SESSION CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Econometrics Exercises Notebook", layout="wide")
st.title("Econometrics Exercises Notebook")

# Author & Disclaimers
st.caption("By Thodoris Kourtalis")
st.markdown("""
**Disclaimer:**  
- I claim **no rights** on this work; it is purely for educational demonstration.  
- You are encouraged to purchase and study the accompanying **econometrics textbook** to gain the full benefit of the material.
""")

# Informational text on how to use the app
st.markdown("""
### How to Use This Website

1. **Navigation Sidebar**  
   - On the **left side**, you’ll find a dropdown list of **Chapters**. Select a chapter to see its available exercises.  
   - Then pick an **Exercise** from that chapter to load it below.

2. **Small-Screen Mode**  
   - If you’re on a **mobile device or narrow screen**, check the box in the sidebar labeled "**I'm on a small screen**".  
   - This mode allows you to **download sample answers as a PDF** rather than viewing them inline.

3. **Working Through an Exercise**  
   - Each exercise may provide a question prompt, input fields, sliders, or other interactive elements.  
   - You can type your own solution in a text area labeled “Your Answer.”  
   - Click **Show Sample Answer** to view or download a reference solution.

Enjoy exploring each chapter’s exercises, and **happy learning**!
""")

# Set up the small-screen flag
if "small_screen" not in st.session_state:
    st.session_state["small_screen"] = False

st.sidebar.markdown("### Display Options")
small_screen = st.sidebar.checkbox("I'm on a small screen", value=st.session_state["small_screen"])
st.session_state["small_screen"] = small_screen

# -----------------------------------------------------------
# DYNAMIC CHAPTERS & EXERCISES
# -----------------------------------------------------------
# Below is a dictionary that maps each chapter to its list of exercises.
# Add new chapters or exercises simply by adding more entries.
chapters_exercises = {
    "Chapter 1: Probability Basics": [
        "1.1: Basic Probability Review",
        "1.2: Bernoulli & Binomial",
    ],
    "Chapter 2: Random Variables": [
        "2.1: Understanding Distributions",
        "2.2: Expected Value Calculation",
        "2.3: Joint and Conditional Probabilities",
        "2.4: Normal Distribution Application",
        "2.5: Bayes’ Rule Challenge",
        "2.6: Skewness & Kurtosis Calculator (Interactive)",
        "2.7: Variance and Std Calculator (Interactive)",
        "2.8: Expected Value Calculator (Interactive)",
        "2.9: Discrete Distribution Plotter (Interactive)",
        "2.10: Bernoulli Simulator (Interactive)",
        "2.11: Joint & Marginal Distribution Table Generator",
        "2.12: Conditional Distribution Calculator",
        "2.13: Law of Iterated Expectations Verifier",
        "2.14: Normal Distribution Probability Calculator (Interactive)",
        "2.15: Bayes’ Rule Visualizer (Interactive)",
        "2.16: Covariance and Correlation Analyzer (Interactive)",
        "2.17: Mean Squared Error Minimizer (Interactive)"
    ],
    # Feel free to add more chapters here...
    # "Chapter 3: Advanced Topics": ["3.1: ...", "3.2: ..."],
}

# Sidebar: pick a chapter
st.sidebar.markdown("---")
selected_chapter = st.sidebar.selectbox("Select a Chapter:", list(chapters_exercises.keys()))

# Sidebar: pick an exercise from the chosen chapter
selected_exercise = st.sidebar.selectbox("Select an Exercise:", chapters_exercises[selected_chapter])

# -----------------------------------------------------------
# PDF GENERATION FUNCTION (unchanged from your original)
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# SHOW SAMPLE ANSWER FUNCTION (unchanged, but with docstring updated)
# -----------------------------------------------------------
def show_sample_answer(sample_md):
    """
    Checks the global small_screen flag.
    If enabled (mobile mode), generates a PDF from sample_md and displays a download button.
    Otherwise, displays sample_md inline using styled Markdown.
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
        # Custom styling for inline sample answer
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


# -----------------------------------------------------------
# DEMO HANDLER: Show or do something based on "selected_exercise"
# -----------------------------------------------------------
# In a real app, you'd define different functions or code blocks for each exercise.
# For demo purposes, we'll just show the selected exercise name, and a pretend sample answer.

st.markdown(f"## You Selected: {selected_exercise}")

# Let's give each exercise a mock or sample structure:
st.markdown("""
**Question:**  
(This is where the question prompt for the selected exercise would go.)
""")

# A user text area for their own solution
st.text_area("Your Answer:", height=150, key="user_answer_for_selected_exercise")

# Expandable sample answer
with st.expander("Show Sample Answer"):
    sample_answer_md = f"""
**Sample Answer for {selected_exercise}:**  

This is a placeholder sample answer in Markdown. 

- You can write LaTeX with double-dollar signs:
$$
\\mathbb{{E}}(X) = \\int x \\, f_X(x) \\, dx.
$$

- Or list items, bullet points, etc.

Remember to **update** this logic to match your real exercises!
"""
    show_sample_answer(sample_answer_md)

st.markdown("""
---
**Note:** You can add or remove chapters and their exercises by editing the `chapters_exercises` dictionary in this script. The UI will adapt automatically!
""")
