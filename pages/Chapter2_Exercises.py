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
import tempfile
import shutil
import re

# -------------------------------------------------------------------
# IMPORT HELPER FUNCTIONS (assumes latex_helpers.py is in your repository root)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# SIDEBAR OPTION: Small Screen Flag
# -------------------------------------------------------------------
st.sidebar.header("Display Options")
# Provide a checkbox that sets the small_screen flag.
small_screen_flag = st.sidebar.checkbox("I'm on a small screen", value=st.session_state.get("small_screen", False))
st.session_state["small_screen"] = small_screen_flag

# -------------------------------------------------------------------
# EXERCISE SELECTION
# -------------------------------------------------------------------
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
        "2.11: Joint & Marginal Distribution Table Generator",
        "2.12: Conditional Distribution Calculator",
        "2.13: Law of Iterated Expectations Verifier"
    ])
st.markdown("---")

# -------------------------------------------------------------------
# GLOBAL SETUP FOR SMALL SCREEN FLAG (fallback)
# -------------------------------------------------------------------
if "small_screen" not in st.session_state:
    st.session_state["small_screen"] = False

# -------------------------------------------------------------------
# EXERCISE FUNCTIONS
# -------------------------------------------------------------------

def exercise_2_1():
    st.subheader("Exercise 2.1: Understanding Distributions")
    st.markdown("""
**Question:**  
Give one example each of a discrete random variable and a continuous random variable from everyday life. Explain why.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_1")
    with st.expander("Show Sample Answer"):
        st.markdown("""
**Sample Answer:**
- **Discrete:** Number of emails received in a day.
- **Continuous:** Time taken to commute.
        """)

def exercise_2_2():
    st.subheader("Exercise 2.2: Expected Value Calculation")
    st.markdown(r"""
**Question:**  
Consider a random variable \( M \) (the number of times your wireless connection fails) with:
$$
\begin{array}{rcl}
P(M=0) & = & 0.80,\\[4mm]
P(M=1) & = & 0.10,\\[4mm]
P(M=2) & = & 0.06,\\[4mm]
P(M=3) & = & 0.03,\\[4mm]
P(M=4) & = & 0.01.
\end{array}
$$
Calculate the expected value \(E(M)\) and explain your steps.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_2")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**
$$
E(M)=\sum_{m} m\,P(M=m)=0\times0.80+1\times0.10+2\times0.06+3\times0.03+4\times0.01=0.35.
$$
Thus, \(E(M)=0.35\).
        """
        show_sample_answer(sample_md, key_suffix="2_2")

def exercise_2_3():
    st.subheader("Exercise 2.3: Joint and Conditional Probabilities")
    st.markdown(r"""
**Question:**  
Suppose we have two binary variables:
- **\(X\)**: Weather (0 = rainy, 1 = clear)
- **\(Y\)**: Commute length (0 = long, 1 = short)

Their joint distribution is given by:
    
|                | \(Y=0\) (Long) | \(Y=1\) (Short) | Total   |
|----------------|----------------|-----------------|---------|
| **\(X=0\)** (Rainy)  | 0.15           | 0.15            | 0.30    |
| **\(X=1\)** (Clear)  | 0.07           | 0.63            | 0.70    |
| **Total**      | 0.22           | 0.78            | 1.00    |
""")
    st.markdown(r"""**Calculate:**  
a) $P(Y=1)$, the marginal probability of a short commute, and  
b) $P(Y=0 \mid X=0)$, the conditional probability of a long commute given that it is rainy.
""")
    st.text_area("Your Answer:", height=200, key="ex2_3")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

a) Marginal Probability:
$$
P(Y=1)=P(X=0,Y=1)+P(X=1,Y=1)=0.15+0.63=0.78.
$$

b) Conditional Probability:
$$
P(Y=0 \mid X=0)=\frac{P(X=0,Y=0)}{P(X=0)}=\frac{0.15}{0.30}=0.50.
$$

Thus, \(P(Y=1)=0.78\) and \(P(Y=0 \mid X=0)=0.50\).
        """
        show_sample_answer(sample_md, key_suffix="2_3")

def exercise_2_4():
    st.subheader("Exercise 2.4: Normal Distribution Application")
    st.markdown("""
**Question:**  
Assume the daily percentage change in a stock price is normally distributed with a mean of 0% and a standard deviation of 1.2%.  
Calculate the probability that, on a given day, the percentage change is less than -2%.  
Outline your steps using the standard normal transformation.
    """)
    st.text_area("Your Answer:", height=180, key="ex2_4")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

1. **Standardize:**
$$
z=\frac{-2-0}{1.2}\approx-1.67.
$$

2. **Find Probability:**
$$
P(Z\le-1.67)\approx0.0475.
$$

Thus, the probability is approximately **4.75%**.
        """
        show_sample_answer(sample_md, key_suffix="2_4")

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
        show_sample_answer(sample_md, key_suffix="2_5")

def exercise_2_6():
    st.subheader("Exercise 2.6: Skewness & Kurtosis Calculator")
    st.markdown("""
**Question:**  
Generate a random sample and examine its skewness and kurtosis.  
Select a distribution and sample size.
    """)
    dist_type = st.selectbox("Select Distribution:", ["Normal", "Uniform", "Exponential"], key="ex2_6_dist")
    sample_size = st.slider("Sample Size:", min_value=50, max_value=1000, value=200, step=50, key="ex2_6_size")
    if dist_type == "Normal":
        data = np.random.normal(0, 1, sample_size)
    elif dist_type == "Uniform":
        data = np.random.uniform(0, 1, sample_size)
    else:
        data = np.random.exponential(1, sample_size)
    skew_val = skew(data)
    kurt_val = kurtosis(data, fisher=False)
    st.markdown(f"**Skewness:** {skew_val:.4f} | **Kurtosis:** {kurt_val:.4f}")
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="mediumseagreen", edgecolor="black")
    ax.set_title(f"{dist_type} Distribution Histogram (n={sample_size})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def exercise_2_7():
    st.subheader("Exercise 2.7: Variance and Standard Deviation Calculator")
    st.markdown("""
**Question:**  
Generate a sample from a chosen distribution and compute its variance and standard deviation.
    """)
    dist_type = st.selectbox("Select Distribution:", ["Normal", "Uniform", "Exponential"], key="ex2_7_dist")
    sample_size = st.slider("Sample Size:", min_value=50, max_value=1000, value=200, step=50, key="ex2_7_size")
    if dist_type == "Normal":
        data = np.random.normal(0, 1, sample_size)
    elif dist_type == "Uniform":
        data = np.random.uniform(0, 1, sample_size)
    else:
        data = np.random.exponential(1, sample_size)
    variance = np.var(data, ddof=1)
    std_dev = np.std(data, ddof=1)
    st.markdown(f"**Variance:** {variance:.4f} | **Standard Deviation:** {std_dev:.4f}")
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="cornflowerblue", edgecolor="black")
    ax.set_title(f"{dist_type} Distribution (n={sample_size})")
    st.pyplot(fig)

def exercise_2_8():
    st.subheader("Exercise 2.8: Expected Value Calculator (Interactive)")
    st.markdown("""
**Question:**  
Specify outcomes and their probabilities using sliders and calculate the expected value.
    """)
    n_outcomes = st.slider("Number of Outcomes:", min_value=2, max_value=10, value=4, key="ex2_8_n")
    outcomes = []
    probs = []
    cols = st.columns(2)
    default_prob = round(1/n_outcomes, 2)
    for i in range(n_outcomes):
        with cols[0]:
            outcome = st.number_input(f"Outcome {i+1} Value:", value=float(i), key=f"ex2_8_out_{i}")
            outcomes.append(outcome)
        with cols[1]:
            prob = st.slider(f"Probability for Outcome {i+1}:", min_value=0.0, max_value=1.0, value=default_prob, step=0.01, key=f"ex2_8_prob_{i}")
            probs.append(prob)
    total_prob = sum(probs)
    if not np.isclose(total_prob, 1):
        st.warning(f"Total probability = {total_prob:.4f}. Normalizing probabilities.")
        probs = [p/total_prob for p in probs]
    exp_val = sum(o * p for o, p in zip(outcomes, probs))
    st.markdown(f"**Expected Value:** {exp_val:.4f}")
    fig, ax = plt.subplots()
    ax.bar(range(n_outcomes), probs, color="mediumpurple", edgecolor="black")
    ax.set_xlabel("Outcome Index")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(n_outcomes))
    ax.set_title("Outcome Probabilities")
    st.pyplot(fig)

def exercise_2_9():
    st.subheader("Exercise 2.9: Discrete Distribution Plotter")
    st.markdown("""
**Question:**  
Use sliders to set probabilities for each outcome of a discrete distribution and plot the distribution.
    """)
    n_outcomes = st.slider("Number of Outcomes:", min_value=2, max_value=10, value=4, key="ex2_9_n")
    outcomes = list(range(n_outcomes))
    probs = []
    for i in range(n_outcomes):
        prob = st.slider(f"Probability for Outcome {i}:", min_value=0.0, max_value=1.0, value=1/n_outcomes, step=0.01, key=f"ex2_9_prob_{i}")
        probs.append(prob)
    total_prob = sum(probs)
    if not np.isclose(total_prob, 1):
        st.warning(f"Total probability = {total_prob:.4f}. Normalizing probabilities.")
        probs = [p/total_prob for p in probs]
    fig, ax = plt.subplots()
    ax.bar(outcomes, probs, color="tomato", edgecolor="black")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Probability")
    ax.set_title("Discrete Probability Distribution")
    st.pyplot(fig)

def exercise_2_10():
    st.subheader("Exercise 2.10: Bernoulli Simulator")
    st.markdown("""
**Question:**  
Simulate Bernoulli trials interactively. Adjust the probability of success and the number of trials using sliders, then view the sample mean and a histogram of outcomes.
    """)
    p = st.slider("Probability of Success:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="ex2_10_p")
    n_trials = st.slider("Number of Trials:", min_value=10, max_value=10000, value=100, step=10, key="ex2_10_n")
    if st.button("Simulate Bernoulli Trials", key="simulate_bernoulli"):
        trials = np.random.binomial(1, p, int(n_trials))
        sample_mean = np.mean(trials)
        st.markdown(f"**Sample Mean:** {sample_mean:.4f}")
        fig, ax = plt.subplots()
        ax.hist(trials, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color="salmon", edgecolor="black")
        ax.set_xticks([0, 1])
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Bernoulli Trials")
        st.pyplot(fig)
def exercise_2_11():
    st.subheader("Exercise 2.11: Joint and Marginal Distribution Table Generator")
    st.markdown("""
**Question:**  
Given a joint probability distribution for two variables X and Y, generate a table showing:
- The joint distribution,
- The marginal distributions for X and Y, and 
- Display a heatmap of the joint distribution.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_11")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

Suppose the joint distribution is given by:

|       | Y=0  | Y=1  | Total |
|-------|------|------|-------|
| X=0   | 0.10 | 0.20 | 0.30  |
| X=1   | 0.25 | 0.45 | 0.70  |
| Total | 0.35 | 0.65 | 1.00  |

Then:
- Marginal for X: \(P(X=0)=0.30\) and \(P(X=1)=0.70\).
- Marginal for Y: \(P(Y=0)=0.10+0.25=0.35\) and \(P(Y=1)=0.20+0.45=0.65\).

A heatmap can be generated using matplotlib (e.g., with `imshow`) by reshaping the joint probabilities into a 2×2 matrix.
        """
        show_sample_answer(sample_md, key_suffix="2_11")

def exercise_2_12():
    st.subheader("Exercise 2.12: Conditional Distribution Calculator")
    st.markdown("""
**Question:**  
Given a joint probability distribution for two variables X and Y, calculate the conditional distributions 
\(P(Y\mid X)\) for each possible value of X.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_12")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

Using the joint distribution from Exercise 2.11:

For \(X=0\):
$$
P(Y=0\mid X=0)=\frac{0.10}{0.30}\approx 0.333,\quad P(Y=1\mid X=0)=\frac{0.20}{0.30}\approx 0.667.
$$

For \(X=1\):
$$
P(Y=0\mid X=1)=\frac{0.25}{0.70}\approx 0.357,\quad P(Y=1\mid X=1)=\frac{0.45}{0.70}\approx 0.643.
$$
        """
        show_sample_answer(sample_md, key_suffix="2_12")

def exercise_2_13():
    st.subheader("Exercise 2.13: Law of Iterated Expectations Verifier")
    st.markdown(r"""
**Question:**  
For random variables \(X\) and \(Y\) with a given joint distribution, verify the law of 
iterated expectations:
$$
\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y\mid X]].
$$
Compute the conditional expectations for each value of \(X\) and show that the overall expectation 
matches the weighted average of these conditional expectations.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_13")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

Assume:
- \(X\) takes values 0 and 1 with \(P(X=0)=0.4\) and \(P(X=1)=0.6\).
- The conditional expectations are: 
  \(\mathbb{E}[Y\mid X=0]=3\) and \(\mathbb{E}[Y\mid X=1]=5\).

By the law of iterated expectations:
$$
\mathbb{E}[Y]= 0.4\times 3 + 0.6\times 5=1.2+3=4.2.
$$

Thus, we verify that:
$$
\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y\mid X]] = 4.2.
$$
        """
        show_sample_answer(sample_md, key_suffix="2_13")
# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
if exercise_choice == "2.1: Understanding Distributions":
    exercise_2_1()
elif exercise_choice == "2.2: Expected Value Calculation":
    exercise_2_2()
elif exercise_choice == "2.3: Joint and Conditional Probabilities":
    exercise_2_3()
elif exercise_choice == "2.4: Normal Distribution Application":
    exercise_2_4()
elif exercise_choice == "2.5: Bayes’ Rule Challenge":
    exercise_2_5()
elif exercise_choice == "2.6: Skewness & Kurtosis Calculator":
    exercise_2_6()
elif exercise_choice == "2.7: Variance and Std Calculator":
    exercise_2_7()
elif exercise_choice == "2.8: Expected Value Calculator (Interactive)":
    exercise_2_8()
elif exercise_choice == "2.9: Discrete Distribution Plotter":
    exercise_2_9()
elif exercise_choice == "2.10: Bernoulli Simulator":
    exercise_2_10()
elif exercise_choice == "2.11: Joint & Marginal Distribution Table Generator":
    exercise_2_11()
elif exercise_choice == "2.12: Conditional Distribution Calculator":
    exercise_2_12()
elif exercise_choice == "2.13: Law of Iterated Expectations Verifier":
    exercise_2_13()    
