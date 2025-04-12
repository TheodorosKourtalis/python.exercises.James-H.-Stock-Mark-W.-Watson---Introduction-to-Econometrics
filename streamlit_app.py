import streamlit as st

st.set_page_config(
    page_title="Econometrics Exercises Notebook",
    page_icon="📚",
    layout="wide",
)

st.title("Econometrics Exercises Notebook")
st.caption("By Thodoris Kourtalis")

st.markdown(
    """
Inspired by *Introduction to Econometrics* by **James H. Stock & Mark W. Watson**  
(Global Edition, Pearson Education Limited)

This is a non-profit educational tool—crafted for rigorous yet accessible learning.

---

Navigate through chapters to access a mix of directly transferred and inspired exercises. Use the sidebar for a quick preview of available content.

[Buy the textbook](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-econometrics/P200000006421)

---
"""
)

# Global display option for small screens.
if "small_screen" not in st.session_state:
    st.session_state["small_screen"] = False

st.sidebar.markdown("### Display Options")
small_screen = st.sidebar.checkbox("I'm on a small screen", value=st.session_state["small_screen"])
st.session_state["small_screen"] = small_screen

st.markdown("---")
