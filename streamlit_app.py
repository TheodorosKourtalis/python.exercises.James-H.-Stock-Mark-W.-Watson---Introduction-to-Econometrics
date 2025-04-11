import streamlit as st

# Configure the main page settings
st.set_page_config(
    page_title="Econometrics Exercises Notebook",
    page_icon="📚",
    layout="wide",
)

# Main header and introduction
st.title("Econometrics Exercises Notebook")
st.markdown(
    """
**Inspired by:** *Introduction to Econometrics*  
*James H. Stock & Mark W. Watson, Global Edition (Pearson Education Limited)*

**Disclaimer:** This non-profit app is a passion project by **Thodoris Kourtalis**.  
All rights and intellectual property belong to the original creators.  
[Buy the book here](https://www.pearson.com)
"""
)
st.markdown("---")

# Instructional message for navigation
st.info("Use the sidebar to explore the chapters. Click on any chapter to see a preview of its exercises. Then, use the app's menu to navigate to a specific chapter's exercises.")

# Sidebar for chapter navigation preview
st.sidebar.header("Chapters Overview")

chapters = {
    "Chapter 1: Economic Questions and Data": [
        "1.1 - Effect of Reading on Vocabulary",
        "1.2 - Alcohol Consumption and Memory Loss",
        "1.3 - Training and Worker Productivity"
    ],
    "Chapter 2: Simple Regression": [
        "Coming soon..."
    ],
    "Chapter 3: Multiple Regression": [
        "Coming soon..."
    ],
    # Add further chapters as you build more content
}

for chapter, exercises in chapters.items():
    with st.sidebar.expander(chapter, expanded=False):
        for exercise in exercises:
            st.write(f"• {exercise}")

# A call-to-action directing the users to the interactive exercise pages.
st.markdown("""
### How to Get Started
1. **Read through the homepage for an overview.**
2. **Use the app's page navigation (top-left menu) to select a chapter.**
3. **Complete the exercises interactively by typing your answers and comparing them to the sample answers.**

Enjoy exploring econometrics in a practical, interactive way!
""")
