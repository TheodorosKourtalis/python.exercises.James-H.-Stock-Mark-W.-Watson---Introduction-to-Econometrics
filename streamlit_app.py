# streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="Econometrics Exercises Notebook",
    page_icon="📚",
    layout="wide",
)

st.title("Econometrics Exercises Notebook")
st.markdown("""
**Inspired by:** *Introduction to Econometrics*  
James H. Stock & Mark W. Watson  
Global Edition, Pearson Education Limited

**Disclaimer:** This non-profit app is a passion project by **Thodoris Kourtalis**.  
All rights belong to the original creators.  
[Buy the book here](https://www.pearson.com)
""")

st.markdown("---")

st.header("Chapters & Exercises Preview")

chapters = {
    "Chapter 1: Economic Questions and Data": [
        "1.1 - Effect of Reading on Vocabulary",
        "1.2 - Alcohol Consumption and Memory Loss",
        "1.3 - Training and Worker Productivity"
    ],
    "Chapter 2: Simple Regression": [
        "Coming soon..."
    ]
}

for chapter, exercises in chapters.items():
    with st.expander(chapter):
        for exercise in exercises:
            st.write(f"• {exercise}")

st.info("To begin, use the sidebar to navigate to a chapter and start solving exercises.")
