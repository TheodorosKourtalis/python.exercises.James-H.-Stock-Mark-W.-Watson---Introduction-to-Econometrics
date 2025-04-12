#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 2: Review of Probability – Exercises
Created on Sat Apr 12 00:53:06 2025
By Thodoris Kourtalis
"""

import streamlit as st
import math
import numpy as np
from scipy.stats import norm, skew, kurtosis
import matplotlib.pyplot as plt
import subprocess
import sys
import os

# Import the helper functions
from latex_helpers import show_sample_answer

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Chapter 2: Exercises - Review of Probability",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Chapter 2: Review of Probability – Exercises")
st.markdown("""
This page presents exercises from *Introduction to Econometrics*.  
Select an exercise, work interactively, and click **Show Sample Answer** to compare your solution.
""")

exercise_choice = st.radio("Select an Exercise:",
    [
        "2.1: Understanding Distributions",
        "2.2: Expected Value Calculation",
        "2.3: Joint and Conditional Probabilities",
        "2.4: Normal Distribution Application",
        "2.5: Bayes’ Rule Challenge",
        "2.6: Skewness & Kurtosis Calculator",
        "2.7: Variance and Std Calculator",
        "2.8: Expected Value Calculator (Interactive)",
        "2.9: Discrete Distribution Plotter",
        "2.10: Bernoulli Simulator"
    ])
st.markdown("---")

# -------------------------------------------------------------------
# GLOBAL SETUP: SMALL SCREEN FLAG (set elsewhere globally, fallback here)
# -------------------------------------------------------------------
if "small_screen" not in st.session_state:
    st.session_state["small_screen"] = False

# -------------------------------------------------------------------
# EXERCISE FUNCTIONS (Examples)
# -------------------------------------------------------------------

def exercise_2_5():
    st.subheader("Exercise 2.5: Bayes’ Rule Challenge")
    st.markdown("""
**Question:**  
A medical test for a particular disease has a sensitivity of 98% and a specificity of 95%. The disease prevalence is 1%.  
If a person tests positive, calculate the probability that they actually have the disease using Bayes’ rule.  
Show all your calculation steps.
    """)
    st.text_area("Your Answer:", height=200, key="ex2_5")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

Let:
$$
P(\text{Disease})\;=\;0.01,\quad P(\text{No Disease})\;=\;0.99,
$$
$$
P(\text{Test Positive}\mid\text{Disease})\;=\;0.98,\quad P(\text{Test Negative}\mid\text{No Disease})\;=\;0.95.
$$
Thus,
$$
P(\text{Test Positive}\mid\text{No Disease})\;=\;1-0.95\;=\;0.05.
$$
Apply Bayes’ rule:
$$
P(\text{Disease}\mid\text{Test Positive})\;=\;\frac{P(\text{Test Positive}\mid\text{Disease})\times P(\text{Disease})}{P(\text{Test Positive})},
$$
where
$$
P(\text{Test Positive})\;=\;(0.98\times0.01)+(0.05\times0.99)\;=\;0.0098+0.0495\;=\;0.0593.
$$
Hence,
$$
P(\text{Disease}\mid\text{Test Positive})\;\approx\;\frac{0.0098}{0.0593}\;\approx\;0.165.
$$
Thus, a person who tests positive has roughly a **16.5% chance** of having the disease.
        """
        show_sample_answer(sample_md)

# (Other exercise functions would follow in a similar pattern)

# -------------------------------------------------------------------
# MAIN EXECUTION: Display the selected exercise.
# -------------------------------------------------------------------
if exercise_choice == "2.5: Bayes’ Rule Challenge":
    exercise_2_5()
# ... (handle other exercises similarly)
