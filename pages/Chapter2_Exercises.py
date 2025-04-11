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
    st.markdown(
        """
**Question:**  
Consider a random variable \( M \) (the number of times your wireless connection fails) with the following distribution:
- \( P(M=0) = 0.80 \)
- \( P(M=1) = 0.10 \)
- \( P(M=2) = 0.06 \)
- \( P(M=3) = 0.03 \)
- \( P(M=4) = 0.01 \)

Calculate \( E(M) \) and explain each step.
        """
    )
    st.text_area("Your Answer:", height=150, key="ex2_2")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**  

$$
E(M) = 0 \times 0.80 + 1 \times 0.10 + 2 \times 0.06 + 3 \times 0.03 + 4 \times 0.01 = 0.35.
$$

Thus, the expected number of failures is **0.35**.
            """
        )

# ------------------------------
# Exercise 2.3: Joint and Conditional Probabilities
def exercise_2_3():
    st.subheader("Exercise 2.3: Joint and Conditional Probabilities")
    st.markdown(
        """
**Question:**  
Suppose we have two binary variables:
- \( X \): Weather (0 = rainy, 1 = clear)
- \( Y \): Commute length (0 = long, 1 = short)

Their joint distribution is:

|                | \(Y=0\) (Long) | \(Y=1\) (Short) | Total  |
|----------------|----------------|-----------------|--------|
| \(X=0\) (Rainy)  | 0.15           | 0.15            | 0.30   |
| \(X=1\) (Clear)  | 0.07           | 0.63            | 0.70   |
| **Total**      | 0.22           | 0.78            | 1.00   |

Calculate:  
a) \( P(Y=1) \), the marginal probability of a short commute, and  
b) \( P(Y=0 \mid X=0) \), the conditional probability of a long commute given that it is rainy.
        """
    )
    st.text_area("Your Answer:", height=200, key="ex2_3")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**  

a)  
$$
P(Y = 1) = 0.15 + 0.63 = 0.78.
$$

b)  
$$
P(Y = 0 \mid X = 0) = \frac{0.15}{0.30} = 0.50.
$$

So, there's a 50% chance of a long commute when it is rainy.
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
A medical test for a disease has:
- \( P(\text{Disease}) = 0.01 \)
- \( P(\text{No Disease}) = 0.99 \)
- Sensitivity: \( P(\text{Test Positive} \mid \text{Disease}) = 0.98 \)
- Specificity: \( P(\text{Test Negative} \mid \text{No Disease}) = 0.95 \)

Calculate \( P(\text{Disease} \mid \text{Test Positive}) \) using Bayes’ rule. Show all steps.
        """
    )
    st.text_area("Your Answer:", height=200, key="ex2_5")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**

Given:
- \( P(\text{Disease}) = 0.01 \)
- \( P(\text{No Disease}) = 0.99 \)
- \( P(\text{Test Positive} \mid \text{Disease}) = 0.98 \)
- \( P(\text{Test Negative} \mid \text{No Disease}) = 0.95 \)
  
Thus,  
$$
P(\text{Test Positive} \mid \text{No Disease}) = 1 - 0.95 = 0.05.
$$

Apply Bayes’ rule:

$$
P(\text{Disease} \mid \text{Test Positive}) = \frac{P(\text{Test Positive} \mid \text{Disease}) \times P(\text{Disease})}{P(\text{Test Positive})},
$$

where

$$
P(\text{Test Positive}) = P(\text{Test Positive} \mid \text{Disease}) \times P(\text{Disease}) + P(\text{Test Positive} \mid \text{No Disease}) \times P(\text{No Disease}).
$$

Plugging in:

$$
P(\text{Test Positive}) = (0.98 \times 0.01) + (0.05 \times 0.99) = 0.0098 + 0.0495 = 0.0593.
$$

Thus,

$$
P(\text{Disease} \mid \text{Test Positive}) = \frac{0.98 \times 0.01}{0.0593} \approx 0.165.
$$

A person who tests positive has roughly a **16.5%** chance of having the disease.
            """
        )

# ------------------------------
# Exercise 2.6: Skewness & Kurtosis Calculator
def exercise_2_6():
    st.subheader("Exercise 2.6: Skewness & Kurtosis Calculator")
    st.markdown(
        """
**Question:**  
Enter a series of numbers (comma separated) to calculate their skewness and kurtosis.
        """
    )
    data_str = st.text_input("Enter numbers separated by commas (e.g., 1, 2, 3, 4, 5):", key="ex2_6")
    if st.button("Calculate Skewness & Kurtosis", key="calc_skew"):
        try:
            data = [float(x.strip()) for x in data_str.split(',') if x.strip() != ""]
            if len(data) < 2:
                st.error("Please enter at least two numbers.")
            else:
                skew_val = skew(data)
                kurt_val = kurtosis(data, fisher=False)  # Use Fisher=False to report raw kurtosis (normal=3)
                st.success(f"Skewness: {skew_val:.4f}, Kurtosis: {kurt_val:.4f}")
        except Exception as e:
            st.error("Error parsing data. Please ensure the values are numeric.")

# ------------------------------
# Exercise 2.7: Variance and Std Calculator
def exercise_2_7():
    st.subheader("Exercise 2.7: Variance and Standard Deviation Calculator")
    st.markdown(
        """
**Question:**  
Enter a series of numbers (comma separated) to calculate their variance and standard deviation.
        """
    )
    data_str = st.text_input("Enter numbers separated by commas:", key="ex2_7")
    if st.button("Calculate Variance & Std", key="calc_var"):
        try:
            data = [float(x.strip()) for x in data_str.split(',') if x.strip() != ""]
            if len(data) < 2:
                st.error("Please enter at least two numbers.")
            else:
                variance = np.var(data, ddof=1)  # Sample variance
                std_dev = np.std(data, ddof=1)   # Sample standard deviation
                st.success(f"Variance: {variance:.4f}, Standard Deviation: {std_dev:.4f}")
        except Exception as e:
            st.error("Error parsing data. Please ensure the values are numeric.")

# ------------------------------
# Exercise 2.8: Expected Value Calculator (Interactive)
def exercise_2_8():
    st.subheader("Exercise 2.8: Expected Value Calculator (Interactive)")
    st.markdown(
        """
**Question:**  
Input a list of outcomes and their corresponding probabilities (comma separated).  
Calculate the expected value.
        """
    )
    outcomes_str = st.text_input("Enter outcomes (e.g., 0, 1, 2, 3):", key="ex2_8_outcomes")
    probs_str = st.text_input("Enter probabilities (e.g., 0.8, 0.1, 0.06, 0.04):", key="ex2_8_probs")
    if st.button("Calculate Expected Value", key="calc_exp"):
        try:
            outcomes = [float(x.strip()) for x in outcomes_str.split(',') if x.strip() != ""]
            probs = [float(x.strip()) for x in probs_str.split(',') if x.strip() != ""]
            if len(outcomes) != len(probs) or len(outcomes) == 0:
                st.error("Please ensure you enter the same number of outcomes and probabilities.")
            else:
                if not np.isclose(sum(probs), 1):
                    st.warning("Probabilities do not sum to 1. They will be normalized.")
                    probs = [p / sum(probs) for p in probs]
                exp_val = sum(o * p for o, p in zip(outcomes, probs))
                st.success(f"Expected Value: {exp_val:.4f}")
        except Exception as e:
            st.error("Error parsing input. Ensure values are numeric and separated by commas.")

# ------------------------------
# Exercise 2.9: Discrete Distribution Plotter
def exercise_2_9():
    st.subheader("Exercise 2.9: Discrete Distribution Plotter")
    st.markdown(
        """
**Question:**  
Enter a list of outcomes and their probabilities (comma separated) to plot the discrete probability distribution.
        """
    )
    outcomes_str = st.text_input("Outcomes (e.g., 0, 1, 2, 3):", key="ex2_9_outcomes")
    probs_str = st.text_input("Probabilities (e.g., 0.8, 0.1, 0.06, 0.04):", key="ex2_9_probs")
    if st.button("Plot Distribution", key="plot_dist"):
        try:
            outcomes = [float(x.strip()) for x in outcomes_str.split(',') if x.strip() != ""]
            probs = [float(x.strip()) for x in probs_str.split(',') if x.strip() != ""]
            if len(outcomes) != len(probs) or len(outcomes) == 0:
                st.error("Ensure equal numbers of outcomes and probabilities are entered.")
            else:
                if not np.isclose(sum(probs), 1):
                    st.warning("Probabilities do not sum to 1. They will be normalized.")
                    probs = [p / sum(probs) for p in probs]
                # Plotting the discrete distribution
                fig, ax = plt.subplots()
                ax.bar(outcomes, probs, width=0.4, color="skyblue", edgecolor="black")
                ax.set_xlabel("Outcomes")
                ax.set_ylabel("Probability")
                ax.set_title("Discrete Probability Distribution")
                st.pyplot(fig)
        except Exception as e:
            st.error("Error parsing input. Ensure numeric, comma-separated values.")

# ------------------------------
# Exercise 2.10: Bernoulli Simulator
def exercise_2_10():
    st.subheader("Exercise 2.10: Bernoulli Simulator")
    st.markdown(
        """
**Question:**  
Simulate Bernoulli trials by specifying the probability of success and the number of trials.  
Display the sample mean along with a histogram of outcomes.
        """
    )
    p = st.number_input("Enter probability of success (0 to 1):", min_value=0.0, max_value=1.0, value=0.5, key="ex2_10_p")
    n_trials = st.number_input("Enter the number of trials:", min_value=1, value=100, key="ex2_10_n")
    if st.button("Simulate Bernoulli Trials", key="simulate_bernoulli"):
        trials = np.random.binomial(n=1, p=p, size=int(n_trials))
        sample_mean = np.mean(trials)
        st.success(f"Sample Mean: {sample_mean:.4f}")
        fig, ax = plt.subplots()
        ax.hist(trials, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color="salmon", edgecolor="black")
        ax.set_xticks([0, 1])
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Bernoulli Trials")
        st.pyplot(fig)

# ------------------------------
# Display the selected exercise based on the user's choice
if exercise_choice.startswith("2.1"):
    exercise_2_1()
elif exercise_choice.startswith("2.2"):
    exercise_2_2()
elif exercise_choice.startswith("2.3"):
    exercise_2_3()
elif exercise_choice.startswith("2.4"):
    exercise_2_4()
elif exercise_choice.startswith("2.5"):
    exercise_2_5()
elif exercise_choice.startswith("2.6"):
    exercise_2_6()
elif exercise_choice.startswith("2.7"):
    exercise_2_7()
elif exercise_choice.startswith("2.8"):
    exercise_2_8()
elif exercise_choice.startswith("2.9"):
    exercise_2_9()
elif exercise_choice.startswith("2.10"):
    exercise_2_10()
