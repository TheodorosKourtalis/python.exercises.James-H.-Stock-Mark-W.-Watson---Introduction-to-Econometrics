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
        "2.6: Skewness & Kurtosis Calculator (Interactive)",
        "2.7: Variance and Std Calculator (Interactive)",
        "2.8: Expected Value Calculator (Interactive)",
        "2.9: Discrete Distribution Plotter (Interactive)",
        "2.10: Bernoulli Simulator (Interactive)",
        "2.11: Joint & Marginal Distribution Table Generator",
        "2.12: Conditional Distribution Calculator",
        "2.13: Law of Iterated Expectations Verifier",
        "2.14: Normal Distribution Probability Calculator (Interactive)",
        "2.15: Bayes’ Rule Visualizer (Interactive)",
        "2.16: Covariance and Correlation Analyzer (Interactive)",
        "2.17: Mean Squared Error Minimizer (Interactive)"
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
    st.subheader("Exercise 2.6: Skewness & Kurtosis Calculator (Interactive)")
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
    st.subheader("Exercise 2.7: Variance and Standard Deviation Calculator (Interactive)")
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
    st.subheader("Exercise 2.9: Discrete Distribution Plotter (Interactive)")
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
    st.subheader("Exercise 2.10: Bernoulli Simulator (Interactive)")
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
Using a 2×2 joint probability distribution for two variables X and Y, generate a table that shows:
- The joint distribution,
- The marginal distributions for X and Y, and 
- A heatmap of the joint distribution.

Additionally, provide a brief theoretical explanation (or proof) for why summing the joint probabilities along each row (or column) yields the marginal distributions.
    """)
    # Button to generate a random 2x2 joint distribution
    if st.button("Generate Random Joint Distribution"):
        import numpy as np
        joint = np.random.rand(2, 2)
        joint = joint / joint.sum()  # Normalize so total sum = 1
        st.write("**Random Joint Probability Distribution:**")
        st.table(joint)
        # Compute marginals:
        marginal_X = joint.sum(axis=1)
        marginal_Y = joint.sum(axis=0)
        st.write("**Marginal Distribution for X:**")
        st.table(marginal_X.reshape(-1, 1))
        st.write("**Marginal Distribution for Y:**")
        st.table(marginal_Y.reshape(1, -1))
        # Generate a heatmap of the joint distribution
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cax = ax.imshow(joint, cmap="viridis", interpolation="none")
        ax.set_title("Joint Distribution Heatmap")
        ax.set_xlabel("Y values")
        ax.set_ylabel("X values")
        fig.colorbar(cax)
        st.pyplot(fig)
    st.text_area("Your Answer:", height=150, key="ex2_11")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

Suppose the following joint distribution is generated:
$$
\begin{bmatrix}
0.10 & 0.20 \\
0.25 & 0.45 \\
\end{bmatrix}
$$
with total sum = 1. The marginal distribution for X is calculated by summing each row:
$$
P(X=0)=0.10+0.20=0.30,\quad P(X=1)=0.25+0.45=0.70.
$$
Similarly, the marginal for Y is computed by summing each column:
$$
P(Y=0)=0.10+0.25=0.35,\quad P(Y=1)=0.20+0.45=0.65.
$$

$$
P(X = x, Y = y).
$$
Summing over all values of 
$$y$$ 
for a fixed value of $$x$$ gives the total probability for that $$x$$, that is:
$$
P(X = x).
$$
Likewise, summing over all values of $$x$$ for a fixed value of $$y$$ gives:
$$
P(Y = y).
$$
This is why the marginal distributions are obtained by summing along the rows (for 
$$P(X = x)$$) or columns (for 
$$P(Y = y)$$) of the joint distribution table.
        """
        show_sample_answer(sample_md, key_suffix="2_11")

def exercise_2_12():
    st.subheader("Exercise 2.12: Conditional Distribution Calculator")
    st.markdown(r"""
**Question:**  
Given a joint probability distribution for two variables $$X$$ and $$Y$$, calculate the conditional distributions  
$$
P(Y \mid X)
$$  
for each value of $$X$$.  

Provide a brief explanation of why the conditional probabilities are computed in this way.
""")
    st.text_area("Your Answer:", height=150, key="ex2_12")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer:**

Using the joint distribution from Exercise 2.11:
$$
\begin{bmatrix}
0.10 & 0.20 \\
0.25 & 0.45 \\
\end{bmatrix}
$$
The marginal distribution for X is:
$$
P(X=0)=0.10+0.20=0.30,\quad P(X=1)=0.25+0.45=0.70.
$$

Then, the conditional distribution for X = 0 is:
$$
P(Y=0\mid X=0)=\frac{0.10}{0.30}\approx 0.333,\quad
P(Y=1\mid X=0)=\frac{0.20}{0.30}\approx 0.667.
$$

For X = 1:
$$
P(Y=0\mid X=1)=\frac{0.25}{0.70}\approx 0.357,\quad
P(Y=1\mid X=1)=\frac{0.45}{0.70}\approx 0.643.
$$

*Explanation:*  
The conditional probability \(P(Y=y \mid X=x)\) is defined as \( \frac{P(X=x,Y=y)}{P(X=x)} \). By dividing the joint probability by the marginal probability of X, we obtain the probability distribution for Y given that X is fixed.
        """
        show_sample_answer(sample_md, key_suffix="2_12")
import streamlit as st
# from latex_helpers import show_sample_answer  # Make sure to import your helper

def exercise_2_13():
    """
    Exercise 2.13: Law of Iterated Expectations Verifier
    Demonstrates how to verify the law of iterated expectations in a simple discrete setting.
    Uses a helper function to display or download the sample answer, depending on the small-screen setting.
    """

    st.subheader("Exercise 2.13: Law of Iterated Expectations Verifier")
    
    st.markdown(r"""
**Question:**  
For random variables $$X$$ and $$Y$$ with a given joint distribution, verify the law of
iterated expectations:
$$
\mathbb{E}[Y] = \mathbb{E}\bigl[\mathbb{E}[Y \mid X]\bigr].
$$

1. Generate or assume a (discrete or continuous) joint distribution for $$X$$ and $$Y$$.  
2. Compute the conditional expectation:
$$
\mathbb{E}[Y \mid X = x]
$$ 
for each possible value of $$x$$.  
3. Show that summing (in the discrete case) or integrating (in the continuous case) over $$x$$, weighted by the distribution of $$X$$, yields:
$$
\mathbb{E}[Y] = \mathbb{E}\bigl[\mathbb{E}[Y \mid X]\bigr].
$$
4. Provide a brief proof outline.
""")

    st.text_area("Your Answer:", height=150, key="ex2_13_user_answer")

    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer (Illustration & Brief Proof)**

1. **Example Setup**  
   Suppose \(X\) takes values 0 and 1 with
   $$
   P(X=0) = 0.4, \quad P(X=1) = 0.6.
   $$
   and assume
   $$
   \mathbb{E}[Y \mid X=0] = 3, \quad \mathbb{E}[Y \mid X=1] = 5.
   $$
   By the law of iterated expectations,
   $$
   \mathbb{E}[Y]
   = P(X=0) \,\mathbb{E}[Y \mid X=0]
   + P(X=1) \,\mathbb{E}[Y \mid X=1]
   = 0.4 \times 3 + 0.6 \times 5 = 4.2.
   $$

2. **Brief Proof Outline (Discrete Case)**  
   - Start with the definition
     $$
     \mathbb{E}[Y] 
     = \sum_{y} y \, P(Y = y).
     $$
   - Express joint probabilities:
     $$
     \mathbb{E}[Y]
     = \sum_{x} \sum_{y} y \, P(X=x, Y=y).
     $$
   - Factorize:
     $$
     P(X=x, Y=y) = P(Y=y \mid X=x)\,P(X=x),
     $$
     so
     $$
     \mathbb{E}[Y]
     = \sum_{x} \sum_{y} y \, P(Y=y \mid X=x)\,P(X=x).
     $$
   - Recognize that
     $$
     \sum_{y} y \, P(Y=y \mid X=x) 
     = \mathbb{E}[Y \mid X=x].
     $$
   - Hence,
     $$
     \mathbb{E}[Y] 
     = \sum_{x} \mathbb{E}[Y \mid X=x]\;P(X=x),
     $$
     which is precisely
     $$
     \mathbb{E}[Y] = \mathbb{E}\bigl[\mathbb{E}[Y \mid X]\bigr].
     $$
        """

        # Use your helper function. For example:
        show_sample_answer(sample_md, key_suffix="2_13")
def exercise_2_14():
    st.subheader("Exercise 2.14: Normal Distribution Probability Calculator (Interactive)")
    st.markdown(r"""
**Question:**  
Given a normally distributed random variable $$Z$$ with mean $$\mu$$ and standard deviation $$\sigma$$, compute the probability
$$
P(Z \le c)
$$
for a chosen threshold $$c$$. Use the cumulative distribution function (CDF) of the standard normal distribution to find this probability, and discuss how changing $$\mu, \sigma,$$ or $$c$$ affects your result.
""")
    # Interactive inputs
    mu = st.number_input("Mean (μ):", value=0.0, step=0.1, key="ex2_14_mu")
    sigma = st.number_input("Std. Dev (σ):", value=1.0, step=0.1, key="ex2_14_sigma")
    c = st.number_input("Threshold (c):", value=0.0, step=0.1, key="ex2_14_c")
    
    if sigma <= 0:
        st.error("σ must be greater than 0.")
        return
    
    # Probability via standardization
    z_star = (c - mu)/sigma
    prob = norm.cdf(z_star)
    st.markdown(f"**Probability:  P(Z ≤ {c}) ≈ {prob:.4f}**")

    # Optional plot of the (standardized) PDF
    xvals = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    pdf_vals = norm.pdf((xvals - mu)/sigma)
    fig, ax = plt.subplots()
    ax.plot(xvals, pdf_vals, label="Standardized PDF", color="midnightblue")
    ax.axvline(c, color="red", linestyle="--", label=f"Threshold c={c}")
    # Shade region
    ax.fill_between(xvals[xvals <= c], pdf_vals[xvals <= c], color="red", alpha=0.3)
    ax.set_title("Normal Distribution Visualization")
    ax.legend()
    st.pyplot(fig)

    st.text_area("Your Answer:", height=150, key="ex2_14_answer")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer**  

1. **Standardization Step**  
$$
Z^* = \frac{c - \mu}{\sigma}.
$$

2. **CDF Lookup**  
The probability is given by the standard normal CDF:
$$
P(Z \le c) = \Phi\bigl(Z^*\bigr).
$$

3. **Example**  
If $$\mu = 0, \sigma = 1,$$ and $$c = 1.64,$$ then
$$
Z^* = 1.64,
$$
and 
$$
P(Z \le 1.64) \approx 0.9495.
$$

4. **Interpretation**  
- Increasing $$c$$ or decreasing $$\sigma$$ generally **increases** the probability.
- Changing $$\mu$$ shifts the center of the distribution.

Thus, the probability calculation follows from the standard normal tables (or `scipy.stats.norm.cdf`).
        """
        show_sample_answer(sample_md, key_suffix="2_14")

def exercise_2_15():
    st.subheader("Exercise 2.15: Bayes’ Rule Visualizer (Interactive)")
    st.markdown(r"""
**Question:**  
Use sliders to adjust the **prior** probability $$P(A)$$, the **likelihood** $$P(B \mid A)$$, and the **false-alarm rate** $$P(B \mid \neg A)$$. Then observe how the **posterior** probability 
$$
P(A \mid B)
$$
changes. Provide a brief explanation connecting these concepts to Bayes’ rule.
""")

    prior = st.slider("Prior P(A):", 0.0, 1.0, 0.2, 0.01, key="ex2_15_prior")
    likelihood = st.slider("P(B | A):", 0.0, 1.0, 0.8, 0.01, key="ex2_15_like")
    false_alarm = st.slider("P(B | ¬A):", 0.0, 1.0, 0.1, 0.01, key="ex2_15_false")
    
    # Bayes: P(A|B) = [P(B|A)*P(A)] / P(B)
    # P(B) = P(B|A)*P(A) + P(B|¬A)*(1 - P(A))
    p_b = likelihood * prior + false_alarm * (1 - prior)
    if p_b == 0:
        st.error("P(B) = 0. Adjust your sliders to avoid this scenario.")
        return
    posterior = (likelihood * prior) / p_b
    
    st.markdown(f"**Posterior P(A|B):** {posterior:.4f}")
    st.markdown(f"**Overall Probability P(B):** {p_b:.4f}")
    
    # Simple bar chart
    fig, ax = plt.subplots()
    categories = ["P(A)", "P(B|A)", "P(B|¬A)", "P(A|B)"]
    values = [prior, likelihood, false_alarm, posterior]
    ax.bar(categories, values, color=["orange","skyblue","limegreen","tomato"])
    ax.set_ylim([0,1])
    ax.set_title("Bayes’ Rule Visualization")
    for i, val in enumerate(values):
        ax.text(i, val+0.02, f"{val:.2f}", ha="center")
    st.pyplot(fig)

    st.text_area("Your Answer:", height=150, key="ex2_15_answer")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer**  

Bayes’ rule states:
$$
P(A \mid B)=\frac{P(B \mid A)\,P(A)}{P(B)},
\quad \text{where} \quad
P(B)=P(B \mid A)\,P(A) + P(B \mid \neg A)\,[1 - P(A)].
$$

- As we increase the likelihood $$P(B \mid A)$$ or prior $$P(A)$$, the posterior $$P(A \mid B)$$ typically **rises**.  
- A high false-alarm rate $$P(B \mid \neg A)$$ means $$B$$ occurs often even when $$A$$ is false, which **lowers** the posterior.  

**Example**: If $$P(A)=0.20, P(B \mid A)=0.80,$$ and $$P(B \mid \neg A)=0.10,$$ then
$$
P(B)=0.80\times 0.20 + 0.10\times 0.80=0.16 + 0.08=0.24,
$$
and
$$
P(A \mid B)=\frac{0.80\times 0.20}{0.24}\approx 0.6667.
$$
        """
        show_sample_answer(sample_md, key_suffix="2_15")


def exercise_2_16():
    st.subheader("Exercise 2.16: Covariance and Correlation Analyzer (Interactive)")
    st.markdown(r"""
**Question:**  
Generate a 2D dataset for two variables $$X$$ and $$Y$$ under different relationships:
- **Positive** (e.g., $$Y=X + \text{noise}$$),
- **Negative** (e.g., $$Y=-X + \text{noise}$$),
- **Uncorrelated** (both independent).

Compute their covariance 
$$
\mathrm{cov}(X,Y)
$$
and correlation 
$$
\mathrm{corr}(X,Y),
$$
then visualize the scatter plot. Discuss how changing the relationship influences these measures.
""")

    relationship = st.selectbox("Select Relationship:", ["Positive", "Negative", "Uncorrelated"], key="ex2_16_relation")
    sample_size = st.slider("Sample Size:", 50, 2000, 500, 50, key="ex2_16_size")
    
    np.random.seed(0)  # for reproducibility
    if relationship == "Positive":
        X = np.random.normal(0, 1, sample_size)
        Y = X + np.random.normal(0, 0.5, sample_size)
    elif relationship == "Negative":
        X = np.random.normal(0, 1, sample_size)
        Y = -X + np.random.normal(0, 0.5, sample_size)
    else:  # Uncorrelated
        X = np.random.normal(0, 1, sample_size)
        Y = np.random.normal(0, 1, sample_size)
    
    cov_xy = np.cov(X, Y, ddof=1)[0,1]
    corr_xy = np.corrcoef(X, Y)[0,1]
    
    st.markdown(f"**Cov(X,Y):** {cov_xy:.4f} | **Corr(X,Y):** {corr_xy:.4f}")
    fig, ax = plt.subplots()
    ax.scatter(X, Y, alpha=0.5, color="purple", edgecolor="white")
    ax.set_title(f"{relationship} Relationship (n={sample_size})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    st.pyplot(fig)
    
    st.text_area("Your Answer:", height=150, key="ex2_16_answer")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer**  

1. **Covariance**:
$$
\mathrm{cov}(X,Y)
= \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}).
$$
2. **Correlation**:
$$
\mathrm{corr}(X,Y)
= \frac{\mathrm{cov}(X,Y)}{\sqrt{\mathrm{var}(X)\,\mathrm{var}(Y)}}.
$$

- **Positive** relationship: $$\mathrm{cov}(X,Y)>0$$ and $$\mathrm{corr}(X,Y)$$ near +1.  
- **Negative** relationship: $$\mathrm{cov}(X,Y)<0$$ and $$\mathrm{corr}(X,Y)$$ near -1.  
- **Uncorrelated**: $$\mathrm{corr}(X,Y)$$ around 0.  

The scatter plot helps visualize whether \(X\) and \(Y\) move together or in opposite directions.
        """
        show_sample_answer(sample_md, key_suffix="2_16")


def exercise_2_17():
    st.subheader("Exercise 2.17: Mean Squared Error Minimizer (Interactive)")
    st.markdown(r"""
**Question:**  
Simulate data \(\{(X_i,Y_i)\}\) for a simple relationship (e.g., $$Y=2X+\text{noise}$$). Compare:
1. A **constant** predictor $$g(X)=\alpha$$.
2. The **conditional mean** predictor $$g(X)=\mathbb{E}[Y\mid X].$$

Compute their mean squared error (MSE):
$$
\mathrm{MSE}(g) = \mathbb{E}\bigl[(Y-g(X))^2\bigr],
$$
and show that the conditional mean yields a strictly lower MSE than any constant function (unless \(X\) and \(Y\) are uncorrelated).
""")

    func_type = st.selectbox("Model g(X):", ["Constant α", "Conditional Mean"], key="ex2_17_func")
    n_points = st.slider("Number of Points:", 50, 1000, 200, 50, key="ex2_17_n")
    
    # Generate data (X uniform, Y=2X + noise)
    X_data = np.random.uniform(0,1,n_points)
    Y_data = 2*X_data + np.random.normal(0,0.2,n_points)
    
    if func_type == "Constant α":
        alpha_val = st.slider("Choose α:", -1.0, 3.0, 1.0, 0.1, key="ex2_17_alpha")
        pred = alpha_val
    else:
        # 'Conditional Mean' -> we use the known true function
        pred = 2*X_data
        alpha_val = None
    
    # Compute MSE
    mse_val = np.mean((Y_data - pred)**2)
    st.markdown(f"**Mean Squared Error (MSE):** {mse_val:.5f}")
    
    # Plot
    fig, ax = plt.subplots()
    ax.scatter(X_data, Y_data, color="gray", alpha=0.5, label="Data")
    if func_type == "Constant α":
        ax.axhline(pred, color="red", label=f"Constant α={alpha_val:.2f}")
    else:
        # Sort X for a clean line
        xsrt = np.sort(X_data)
        ax.plot(xsrt, 2*xsrt, color="blue", label=r"$g(X)=2X$")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"MSE = {mse_val:.5f}")
    ax.legend()
    st.pyplot(fig)

    st.text_area("Your Answer:", height=150, key="ex2_17_answer")
    with st.expander("Show Sample Answer"):
        sample_md = r"""
**Sample Answer**  

1. **Mean Squared Error**  
$$
\mathrm{MSE}(g)
= \mathbb{E}\bigl[(Y - g(X))^2\bigr].
$$

2. **Constant vs. Conditional Mean**  
- If $$g(X)=\alpha$$ (a constant), the best choice is $$\alpha = \mathbb{E}[Y].$$ Even then, it ignores any dependence of \(Y\) on \(X\).  
- By contrast, if $$g(X)=\mathbb{E}[Y\mid X],$$ then 
$$
\mathrm{MSE}(g)
$$
is minimized among *all possible* functions \(g\). This is a fundamental result in statistics: the conditional mean is the best predictor in the least squares sense.

3. **Example**  
Here, we generated data with 
$$
Y=2X + \text{noise}.
$$
- The constant function's MSE is higher unless \(X\) and \(Y\) are completely unrelated.  
- Using $$g(X)=2X$$ or a similar approximation closely matches the true pattern, yielding a lower MSE.
        """
        show_sample_answer(sample_md, key_suffix="2_17")
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
elif exercise_choice == "2.6: Skewness & Kurtosis Calculator (Interactive)":
    exercise_2_6()
elif exercise_choice == "2.7: Variance and Std Calculator (Interactive)":
    exercise_2_7()
elif exercise_choice == "2.8: Expected Value Calculator (Interactive)":
    exercise_2_8()
elif exercise_choice == "2.9: Discrete Distribution Plotter (Interactive)":
    exercise_2_9()
elif exercise_choice == "2.10: Bernoulli Simulator (Interactive)":
    exercise_2_10()
elif exercise_choice == "2.11: Joint & Marginal Distribution Table Generator":
    exercise_2_11()
elif exercise_choice == "2.12: Conditional Distribution Calculator":
    exercise_2_12()
elif exercise_choice == "2.13: Law of Iterated Expectations Verifier":
    exercise_2_13()    
elif exercise_choice == "2.14: Normal Distribution Probability Calculator (Interactive)":
    exercise_2_14()
elif exercise_choice == "2.15: Bayes’ Rule Visualizer (Interactive)":
    exercise_2_15()
elif exercise_choice == "2.16: Covariance and Correlation Analyzer (Interactive)":
    exercise_2_16()
elif exercise_choice == "2.17: Mean Squared Error Minimizer (Interactive)":
    exercise_2_17()
