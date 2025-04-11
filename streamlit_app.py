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
    # Add more chapters and exercises as required
}

# Sidebar: chapter selection
st.sidebar.header("Exercises Navigation")
selected_chapter = st.sidebar.selectbox(
    "Choose a Chapter:",
    list(chapters.keys())
)

# Sidebar: exercise selection within the chosen chapter
if selected_chapter:
    selected_exercise = st.sidebar.selectbox(
        "Choose an Exercise:",
        chapters[selected_chapter]
    )
    
    # Display selected content on the main page
    st.header(f"{selected_chapter} - {selected_exercise}")
    st.write(f"Notebook content for **{selected_exercise}** goes here.")
    # You can replace the above line with the actual interactive notebook content
