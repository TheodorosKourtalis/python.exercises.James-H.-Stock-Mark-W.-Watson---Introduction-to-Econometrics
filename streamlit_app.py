#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 00:33:44 2025

@author: thodoreskourtales
"""

import streamlit as st

# Page configuration: title, icon, and layout
st.set_page_config(
    page_title="Econometrics Exercises Notebook",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main header and description on the main page
st.title("Econometrics Exercises Notebook")
st.markdown(
    """
**Inspired by:** *Introduction to Econometrics*  
James H. Stock & Mark W. Watson  
Global Edition, Pearson Education Limited
"""
)
st.markdown(
    """
**Disclaimer:** This non-profit app is a passion project developed solely by **Thodoris Kourtalis**.  
All intellectual property and rights to the book belong to the original creators.
"""
)
st.markdown(
    "If you wish to support the original work, you can [buy the book here](https://www.pearson.com)."
)

st.markdown("---")

# Sidebar section for chapters and exercises
st.sidebar.header("Chapters & Exercises")

# Define the chapters and exercises in a dictionary
# You can update these with the actual exercise titles as your project evolves.
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
    # Add additional chapters and exercises as needed
}

# Create expandable sections in the sidebar for every chapter
for chapter, exercises in chapters.items():
    with st.sidebar.expander(chapter, expanded=False):
        # Create nested expanders (or bullet lists) for each exercise within the chapter
        for exercise in exercises:
            with st.expander(exercise, expanded=False):
                st.write(f"**Notebook content for {exercise}:**")
                st.write("Place your interactive code, examples, or further instructions here.")

# Additional instructions on the main page
st.header("Welcome to Your Econometrics Notebook")
st.write(
    """
This interactive application provides curated exercise notebooks organized by chapters.
Each chapter includes exercises that are either direct transferrals from or inspired by the original book.
You can explore the notebooks via the sidebar. Click on a chapter to see available exercises, and expand each exercise to reveal its details.

---
**How to Use This App:**
- **Navigation:** Use the sidebar to browse through chapters and exercises.
- **Learning:** Each exercise notebook may include explanations, example code, and interactive widgets to enhance your econometrics learning experience.
- **Feedback:** As this is a passion project by Thodoris Kourtalis, feel free to reach out for any suggestions or comments.
"""
)