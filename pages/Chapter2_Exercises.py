#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 00:53:06 2025

@author: ThodorisKourtalis
"""

import streamlit as st
import math
from scipy.stats import norm  # For normal distribution (optional for sample calculation)

# Configure the chapter page settings
st.set_page_config(
    page_title="Chapter 2: Exercises - Review of Probability",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Chapter 2: Review of Probability – Exercises")
st.markdown(
    """
This page provides a series of exercises to help you review key concepts from Chapter 2 of *Introduction to Econometrics*.  
Choose an exercise below, try to solve it on your own in the text area provided, and then click "Show Sample Answer" to compare your work.
    """
)

# Use radio buttons (or a selectbox) to allow the user to choose an exercise.
exercise_choice = st.radio("Select an Exercise:",
                            [
                              "2.1: Understanding Distributions",
                              "2.2: Expected Value Calculation",
                              "2.3: Joint and Conditional Probabilities",
                              "2.4: Normal Distribution Application",
                              "2.5: Bayes’ Rule Challenge"
                            ])

st.markdown("---")

# Exercise 2.1: Understanding Distributions
def exercise_2_1():
    st.subheader("Exercise 2.1: Understanding Discrete and Continuous Distributions")
    st.markdown(
        """
**Question:**  
In everyday life, describe one example of a **discrete random variable** and one example of a **continuous random variable**. Explain why each example fits its category.
        
*Hint:* Consider events such as the number of phone calls you receive in an hour (discrete) versus the time you spend on a commute (continuous).
        """
    )
    st.text_area("Your Answer:", height=150, key="ex2_1")
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
- **Discrete Random Variable:**  
  *Example:* The number of emails you receive in a day.  
  *Explanation:* The values can only be whole numbers (0, 1, 2, …) and do not take on fractional values.

- **Continuous Random Variable:**  
  *Example:* The amount of time (in minutes) it takes to cook a meal.  
  *Explanation:* Time can be measured to any degree of precision (e.g., 35.27 minutes) and has an infinite number of possible values in any interval.
            """
        )

# Exercise 2.2: Expected Value Calculation
def exercise_2_2():
    st.subheader("Exercise 2.2: Expected Value Calculation")
    st.markdown(
        """
**Question:**  
Consider a random variable M representing the number of times your wireless connection fails while writing a term paper. Its probability distribution is given by:

- Pr(M = 0) = 0.80  
- Pr(M = 1) = 0.10  
- Pr(M = 2) = 0.06  
- Pr(M = 3) = 0.03  
- Pr(M = 4) = 0.01  

Calculate the expected number of failures and explain your calculation step by step.
        """
    )
    st.text_area("Your Answer:", height=150, key="ex2_2")
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
The expected value is computed by taking the sum of each outcome multiplied by its probability:

$$
E(M) = 0 \\times 0.80 + 1 \\times 0.10 + 2 \\times 0.06 + 3 \\times 0.03 + 4 \\times 0.01 
= 0 + 0.10 + 0.12 + 0.09 + 0.04 = 0.35.
$$

Thus, the expected number of connection failures is **0.35**.
            """
        )

# Exercise 2.3: Joint and Conditional Probabilities
def exercise_2_3():
    st.subheader("Exercise 2.3: Joint and Conditional Probabilities")
    st.markdown(
        """
**Question:**  
Imagine two binary random variables:  
- **X** represents the weather (0 = rainy, 1 = clear), and  
- **Y** represents the length of a commute (0 = long, 1 = short).

Their joint probability distribution is given by:  

|        | Y = 0 (Long) | Y = 1 (Short) | Total   |
|--------|--------------|---------------|---------|
| X = 0 (Rainy) | 0.15         | 0.15         | 0.30    |
| X = 1 (Clear) | 0.07         | 0.63         | 0.70    |
| **Total**     | 0.22         | 0.78         | 1.00    |

Calculate:  
a) The marginal probability of having a short commute, and  
b) The conditional probability of having a long commute given that it is raining.
        """
    )
    st.text_area("Your Answer:", height=200, key="ex2_3")
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
a) The marginal probability of a short commute is the sum of the probabilities where \\(Y = 1\\):

$$
P(Y = 1) = 0.15 + 0.63 = 0.78.
$$

b) The conditional probability of a long commute given that it is raining is:

$$
P(Y = 0 \\mid X = 0) 
= \\frac{P(X = 0 \\text{ and } Y = 0)}{P(X = 0)} 
= \\frac{0.15}{0.30} 
= 0.50.
$$

So, there is a 50% chance of a long commute when it is raining.
            """
        )

# Exercise 2.4: Normal Distribution Application
def exercise_2_4():
    st.subheader("Exercise 2.4: Normal Distribution Application")
    st.markdown(
        """
**Question:**  
Assume that the daily percentage change in a stock price follows a normal distribution with a mean of 0% and a standard deviation of 1.2%.  
Calculate the probability that, on a given day, the percentage change is less than -2%.  
Outline your steps using the standard normal transformation.
        """
    )
    st.text_area("Your Answer:", height=180, key="ex2_4")
    with st.expander("Show Sample Answer"):
        st.markdown(
            r"""
**Sample Answer:**  

1. **Standardize the value.** The z-score is:

$$
z \;=\; \frac{X - \mu}{\sigma} \;=\; \frac{-2 \,-\, 0}{1.2} \;\approx\; -1.67.
$$

2. **Find the cumulative probability.** Using a standard normal table (or a calculator):

$$
P\bigl(Z \le -1.67\bigr) \;\approx\; 0.0475.
$$

Thus, the probability that the stock price change is less than -2% is approximately **4.75%**.
            """
        )

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
