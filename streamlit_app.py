import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Econometrics Exercises Notebook",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page header and information
st.title("Econometrics Exercises Notebook")
st.markdown(
    """
**Inspired by:** *Introduction to Econometrics*  
James H. Stock & Mark W. Watson  
Global Edition, Pearson Education Limited  
**Disclaimer:** This non-profit app is a passion project by **Thodoris Kourtalis**.  
All rights belong to the original creators.  
[Buy the book here](https://www.pearson.com)
"""
)

# Define chapters and exercises
chapters = {
    "Chapter 1: Introduction": [
        "Exercise 1.1 - Overview",
        "Exercise 1.2 - Basic Concepts",
        "Exercise 1.3 - Data Interpretation"
    ],
    "Chapter 2: Simple Regression": [
        "Exercise 2.1 - Regression Basics",
        "Exercise 2.2 - Model Fitting",
        "Exercise 2.3 - Residual Analysis"
    ],
    "Chapter 3: Multiple Regression": [
        "Exercise 3.1 - Multicollinearity",
        "Exercise 3.2 - Hypothesis Testing"
    ],
    "Chapter 4: Advanced Topics": [
        "Exercise 4.1 - Instrumental Variables",
        "Exercise 4.2 - Panel Data Analysis"
    ],
    # Add additional chapters and exercises as necessary
}

# Sidebar: Show chapter expanders with exercises as bullet points
st.sidebar.header("Chapters & Exercises")
for chapter, exercises in chapters.items():
    expander = st.sidebar.expander(chapter, expanded=False)
    # Display exercises as a bullet list
    for exercise in exercises:
        expander.write(f"- {exercise}")

# Main page content for instructions
st.header("Welcome to Your Econometrics Notebook")
st.write(
    """
This interactive application provides curated exercise notebooks organized by chapters.
Select a chapter from the sidebar to view a summary of its exercises.
For interactive content on an exercise, consider using the dropdown selections on the main page.
"""
)
