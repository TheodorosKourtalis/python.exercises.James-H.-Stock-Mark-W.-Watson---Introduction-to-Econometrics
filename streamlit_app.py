import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Econometrics Exercises – Thodoris Kourtalis",
    page_icon="📚",
    layout="wide",
)

# Minimal header
st.title("Econometrics Exercises")
st.caption("By Thodoris Kourtalis")

st.markdown(
    """
This open educational tool presents exercises inspired by  
**James H. Stock & Mark W. Watson – Introduction to Econometrics**  
(Global Edition, Pearson Education Limited)

All intellectual property remains with the original authors.  
This is a non-profit academic initiative.
"""
)

st.markdown("---")

# Chapters dictionary
chapters = {
    "Chapter 1 – Economic Questions and Data": [
        "1.1 – Reading and Vocabulary",
        "1.2 – Alcohol and Memory Loss",
        "1.3 – Training and Productivity"
    ],
    "Chapter 2 – Review of Probability": [
        "2.1 – Discrete vs. Continuous Variables",
        "2.2 – Expected Value of Network Failures",
        "2.3 – Joint and Conditional Probabilities",
        "2.4 – Stock Returns and the Normal Distribution",
        "2.5 – Bayes’ Rule in Diagnostics"
    ],
    "Chapter 3 – Multiple Regression": [
        "Coming soon"
    ]
}

# Clean sidebar preview
st.sidebar.header("Navigation")
for chapter, exercises in chapters.items():
    with st.sidebar.expander(chapter):
        for ex in exercises:
            st.markdown(f"- {ex}")

# Main chapter preview
st.subheader("Contents")
for chapter, exercises in chapters.items():
    st.markdown(f"**{chapter}**")
    for ex in exercises:
        st.markdown(f"- {ex}")
    st.markdown("")

# Footer
st.markdown("---")
st.markdown(
    """
For reference or to support the original authors,  
consider [purchasing the textbook](https://www.pearson.com).
"""
)
