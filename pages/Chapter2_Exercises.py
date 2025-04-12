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
This page presents exercises from Chapter 2 of *Introduction to Econometrics*.  
Select an exercise, work through it interactively, and click **Show Sample Answer** to compare your solution.
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
# HELPER FUNCTIONS
# -------------------------------------------------------------------
def generate_pdf_via_subprocess(sample_text):
    try:
        process = subprocess.Popen(
            [sys.executable, "generate_pdf.py"],
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

def show_sample_answer(sample_md, key_suffix="default"):
    if st.session_state.get("small_screen", False):
        pdf_data = generate_pdf_via_subprocess(sample_md)
        if pdf_data:
            st.download_button(
                label="Download Sample Answer PDF",
                data=pdf_data,
                file_name="sample_answer.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown("""
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
        """, unsafe_allow_html=True)
        st.markdown('<div class="sample-answer">', unsafe_allow_html=True)
        st.markdown(sample_md)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# GLOBAL SMALL SCREEN FLAG (fallback standalone)
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
Give one example of a discrete random variable and one example of a continuous random variable from everyday life. Explain why.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_1")
    with st.expander("Show Sample Answer"):
        st.markdown("""
**Sample Answer:**
- **Discrete:** Number of emails received in a day (μόνο ακέραιοι αριθμοί).  
- **Continuous:** Time taken to commute (μπορεί να έχει δεκαδικά).
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
Calculate \(E(M)\) and explain your steps.
    """)
    st.text_area("Your Answer:", height=150, key="ex2_2")
    with st.expander("Show Sample Answer"):
        st.markdown(r"""
**Sample Answer:**
$$
E(M) = \sum_{m} m\,P(M=m) = 0\times0.80+ 1\times0.10+ 2\times0.06+ 3\times0.03+ 4\times0.01 = 0.35.
$$
Thus, \(E(M)=0.35\).
        """)

def exercise_2_3():
    st.subheader("Exercise 2.3: Joint and Conditional Probabilities")
    st.markdown(r"""
**Question:**  
Suppose we have two binary variables:
- **\(X\)**: Weather (0 = rainy, 1 = clear)
- **\(Y\)**: Commute length (0 = long, 1 = short)

Their joint distribution is:

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
        st.markdown(r"""
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
        """)

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
P(Z\leq -1.67)\approx0.0475.
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

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
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
