#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 2: Review of Probability – Exercises
Created on Sat Apr 12 00:53:06 2025

@author: ThodorisKourtalis
"""

import streamlit as st
import math
import numpy as np
from scipy.stats import norm, skew, kurtosis  # For normal distribution and statistics
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Chapter 2: Exercises - Review of Probability",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Chapter 2: Review of Probability – Exercises")
st.markdown(
    """
This page provides a series of exercises to help you review key concepts from Chapter 2 of *Introduction to Econometrics*.  
Select an exercise from the menu below, try solving it using the interactive inputs provided, and then click "Show Sample Answer" to compare your solution.
    """
)

# Updated Exercise selection including additional interactive exercises
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

# ------------------------------
# Exercise 2.1: Understanding Distributions
def exercise_2_1():
    st.subheader("Exercise 2.1: Understanding Discrete and Continuous Distributions")
    st.markdown(
        """
**Question:**  
Describe one example of a **discrete random variable** and one example of a **continuous random variable** from everyday life. Explain why each example fits its category.

*Hint:* Countable outcomes (e.g., number of phone calls) vs. measurable outcomes (e.g., travel time).
        """
    )
    st.text_area("Your Answer:", height=150, key="ex2_1")
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
- **Discrete Random Variable:**  
  *Example:* The number of emails received in a day.  
  *Explanation:* Only whole-number values (0, 1, 2, …) are possible.  

- **Continuous Random Variable:**  
  *Example:* The time taken to commute to work (in minutes).  
  *Explanation:* It can take any value in a range (e.g., 35.27 minutes) with infinite precision.
            """
        )

# ------------------------------
# Exercise 2.2: Expected Value Calculation
def exercise_2_2():
    st.subheader("Exercise 2.2: Expected Value Calculation")
    st.markdown(r"""
**Question:**  
Consider a random variable \( M \) (the number of times your wireless connection fails) with the following distribution:
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
        st.markdown(r"""
**Sample Answer:**

The expected value is computed as:
$$
E(M) = \sum_{m} m\,P(M=m) = 0 \times 0.80 + 1 \times 0.10 + 2 \times 0.06 + 3 \times 0.03 + 4 \times 0.01 = 0.35.
$$

Thus, \(E(M)=0.35\).
        """)

# ------------------------------
# Exercise 2.3: Joint and Conditional Probabilities
def exercise_2_3():
    st.subheader("Exercise 2.3: Joint and Conditional Probabilities")
    
    # First part: Question and variable definitions.
    st.markdown(r"""
**Question:**  
Suppose we have two binary variables:
- **\(X\)**: Weather (0 = rainy, 1 = clear)
- **\(Y\)**: Commute length (0 = long, 1 = short)
    """)
    
    # Next, display the joint distribution table.
    st.markdown(r"""
Their joint distribution is given by:

|                | \(Y=0\) (Long) | \(Y=1\) (Short) | Total   |
|----------------|----------------|-----------------|---------|
| **\(X=0\)** (Rainy)  | 0.15           | 0.15            | 0.30    |
| **\(X=1\)** (Clear)  | 0.07           | 0.63            | 0.70    |
| **Total**      | 0.22           | 0.78            | 1.00    |
    """)
    
    # Then, in a separate Markdown block, include the "Calculate:" portion.
 # Prompt with inline math expressions
   st.markdown(r"""
**Calculate:**  
a) `$P(Y=1)$`, the marginal probability of a short commute, and  
b) `$P(Y=0 \\mid X=0)$`, the conditional probability of a long commute given that it is rainy.
""")

    st.text_area("Your Answer:", height=200, key="ex2_3")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**

a) **Marginal Probability \(P(Y=1)\):**

To calculate the marginal probability of a short commute, sum the joint probabilities for \(Y=1\):
$$
P(Y=1) = P(X=0, Y=1) + P(X=1, Y=1) = 0.15 + 0.63 = 0.78.
$$

b) **Conditional Probability \(P(Y=0 \mid X=0)\):**

This is the probability of a long commute given that it is rainy. It is calculated by dividing the joint probability \(P(X=0, Y=0)\) by the marginal probability \(P(X=0)\):
$$
P(Y=0 \mid X=0) = \frac{P(X=0, Y=0)}{P(X=0)} = \frac{0.15}{0.30} = 0.50.
$$

Thus, there is a 50% chance of a long commute when the weather is rainy.
            """
        )

# ------------------------------
# Exercise 2.4: Normal Distribution Application
def exercise_2_4():
    st.subheader("Exercise 2.4: Normal Distribution Application")
    st.markdown(
        """
**Question:**  
Assume the daily percentage change in a stock price is normally distributed with a mean of 0% and a standard deviation of 1.2%.  
Calculate the probability that, on a given day, the percentage change is less than -2%.  
Outline your steps using the standard normal transformation.
        """
    )
    st.text_area("Your Answer:", height=180, key="ex2_4")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**  

1. **Standardize the value.**  
   The z-score is:

   $$
   z = \frac{X - \mu}{\sigma} = \frac{-2 - 0}{1.2} \approx -1.67.
   $$

2. **Find the cumulative probability.**  
   Using a standard normal table:

   $$
   P(Z \leq -1.67) \approx 0.0475.
   $$

Thus, the probability is approximately **4.75%**.
            """
        )

# ------------------------------
# Exercise 2.5: Bayes’ Rule Challenge
def exercise_2_5():
    st.subheader("Exercise 2.5: Bayes’ Rule Challenge")
    st.markdown(
        """
**Question:**  
A medical test for a particular disease has a sensitivity of 98% and a specificity of 95%. The disease prevalence in the population is 1%.  
If a person tests positive, calculate the probability that they actually have the disease using Bayes’ rule.  
Show all your calculation steps.
        """
    )
    st.text_area("Your Answer:", height=200, key="ex2_5")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**

Let:

$$
P(\text{Disease}) \;=\; 0.01, \qquad P(\text{No Disease}) \;=\; 0.99,
$$

$$
P(\text{Test Positive}\mid\text{Disease}) \;=\; 0.98, \qquad P(\text{Test Negative}\mid\text{No Disease}) \;=\; 0.95.
$$

Thus, 
$$
P(\text{Test Positive} \mid \text{No Disease})
\;=\; 1 - 0.95
\;=\; 0.05.
$$

Apply **Bayes’ rule**:

$$
P(\text{Disease} \mid \text{Test Positive}) 
= \frac{\,P(\text{Test Positive}\mid\text{Disease}) \,\times\, P(\text{Disease})\,}
       {\,P(\text{Test Positive})\,}.
$$

where

$$
P(\text{Test Positive}) 
= P(\text{Test Positive}\mid\text{Disease}) \,\times\, P(\text{Disease})
\;+\; P(\text{Test Positive}\mid\text{No Disease}) \,\times\, P(\text{No Disease}).
$$

Plug in the values:

$$
P(\text{Test Positive})
= (0.98 \,\times\, 0.01) \;+\; (0.05 \,\times\, 0.99)
= 0.0098 + 0.0495
= 0.0593.
$$

Hence,

$$
P(\text{Disease}\mid\text{Test Positive})
= \frac{\,0.98 \,\times\, 0.01\,}{\,0.0593\,}
\;\approx\; \frac{0.0098}{\,0.0593\,}
\;\approx\; 0.165.
$$

Thus, a person who tests positive has roughly a **16.5% chance** of actually having the disease.
            """
        )

# ------------------------------
# Exercise 2.6: Skewness & Kurtosis Calculator (Interactive)
def exercise_2_6():
    st.subheader("Exercise 2.6: Skewness & Kurtosis Calculator")
    st.markdown(
        """
**Question:**  
Generate a random sample and examine its skewness and kurtosis.  
Select a distribution and sample size.
        """
    )
    dist_type = st.selectbox("Select Distribution:", ["Normal", "Uniform", "Exponential"], key="ex2_6_dist")
    sample_size = st.slider("Sample Size:", min_value=50, max_value=1000, value=200, step=50, key="ex2_6_size")
    
    if dist_type == "Normal":
        data = np.random.normal(loc=0, scale=1, size=sample_size)
    elif dist_type == "Uniform":
        data = np.random.uniform(low=0, high=1, size=sample_size)
    else:  # Exponential
        data = np.random.exponential(scale=1, size=sample_size)
    
    skew_val = skew(data)
    kurt_val = kurtosis(data, fisher=False)  # raw kurtosis; normal = 3
    st.markdown(f"**Skewness:** {skew_val:.4f}  |  **Kurtosis:** {kurt_val:.4f}")
    
    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="mediumseagreen", edgecolor="black")
    ax.set_title(f"{dist_type} Distribution Histogram (n={sample_size})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# ------------------------------
# Exercise 2.7: Variance and Standard Deviation Calculator (Interactive)
def exercise_2_7():
    st.subheader("Exercise 2.7: Variance and Standard Deviation Calculator")
    st.markdown(
        """
**Question:**  
Generate a sample from a chosen distribution and compute its variance and standard deviation.
        """
    )
    dist_type = st.selectbox("Select Distribution:", ["Normal", "Uniform", "Exponential"], key="ex2_7_dist")
    sample_size = st.slider("Sample Size:", min_value=50, max_value=1000, value=200, step=50, key="ex2_7_size")
    
    if dist_type == "Normal":
        data = np.random.normal(loc=0, scale=1, size=sample_size)
    elif dist_type == "Uniform":
        data = np.random.uniform(low=0, high=1, size=sample_size)
    else:
        data = np.random.exponential(scale=1, size=sample_size)
    
    variance = np.var(data, ddof=1)  # sample variance
    std_dev = np.std(data, ddof=1)   # sample std
    st.markdown(f"**Variance:** {variance:.4f}  |  **Standard Deviation:** {std_dev:.4f}")
    
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="cornflowerblue", edgecolor="black")
    ax.set_title(f"{dist_type} Distribution (n={sample_size})")
    st.pyplot(fig)

# ------------------------------
# Exercise 2.8: Expected Value Calculator (Interactive)
def exercise_2_8():
    st.subheader("Exercise 2.8: Expected Value Calculator (Interactive)")
    st.markdown(
        """
**Question:**  
Specify outcomes and their probabilities using sliders and calculate the expected value.
        """
    )
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
        st.warning(f"The total probability is {total_prob:.4f}. Probabilities will be normalized.")
        probs = [p/total_prob for p in probs]
    exp_val = sum(o * p for o, p in zip(outcomes, probs))
    st.markdown(f"**Expected Value:** {exp_val:.4f}")
    
    # Optionally, plot the outcomes with probabilities
    fig, ax = plt.subplots()
    ax.bar(range(n_outcomes), probs, color="mediumpurple", edgecolor="black")
    ax.set_xlabel("Outcome Index")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(n_outcomes))
    ax.set_title("Outcome Probabilities")
    st.pyplot(fig)

# ------------------------------
# Exercise 2.9: Discrete Distribution Plotter (Interactive)
def exercise_2_9():
    st.subheader("Exercise 2.9: Discrete Distribution Plotter")
    st.markdown(
        """
**Question:**  
Use sliders to set probabilities for each outcome of a discrete distribution and plot the distribution.
        """
    )
    n_outcomes = st.slider("Number of Outcomes:", min_value=2, max_value=10, value=4, key="ex2_9_n")
    outcomes = list(range(n_outcomes))
    probs = []
    for i in range(n_outcomes):
        prob = st.slider(f"Probability for Outcome {i}:", min_value=0.0, max_value=1.0, value=1/n_outcomes, step=0.01, key=f"ex2_9_prob_{i}")
        probs.append(prob)
    total_prob = sum(probs)
    if not np.isclose(total_prob, 1):
        st.warning(f"Total probability = {total_prob:.4f}. Probabilities will be normalized.")
        probs = [p/total_prob for p in probs]
    fig, ax = plt.subplots()
    ax.bar(outcomes, probs, color="tomato", edgecolor="black")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Probability")
    ax.set_title("Discrete Probability Distribution")
    st.pyplot(fig)

# ------------------------------
# Exercise 2.10: Bernoulli Simulator (Interactive)
def exercise_2_10():
    st.subheader("Exercise 2.10: Bernoulli Simulator")
    st.markdown(
        """
**Question:**  
Simulate Bernoulli trials interactively. Adjust the probability of success and the number of trials using sliders, then view the sample mean and a histogram of outcomes.
        """
    )
    p = st.slider("Probability of Success:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="ex2_10_p")
    n_trials = st.slider("Number of Trials:", min_value=10, max_value=10000, value=100, step=10, key="ex2_10_n")
    if st.button("Simulate Bernoulli Trials", key="simulate_bernoulli"):
        trials = np.random.binomial(n=1, p=p, size=int(n_trials))
        sample_mean = np.mean(trials)
        st.markdown(f"**Sample Mean:** {sample_mean:.4f}")
        fig, ax = plt.subplots()
        ax.hist(trials, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color="salmon", edgecolor="black")
        ax.set_xticks([0, 1])
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Bernoulli Trials")
        st.pyplot(fig)

# ------------------------------
# Display selected exercise using exact string comparison
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
