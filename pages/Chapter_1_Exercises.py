import streamlit as st

# Configure the chapter page settings
st.set_page_config(
    page_title="Chapter 1: Exercises",
    page_icon="📘",
    layout="wide",
)

st.title("📘 Chapter 1: Economic Questions and Data")
st.markdown(
    """
This page presents review exercises from Chapter 1 of *Introduction to Econometrics*.  
Select an exercise below to view the question, type your answer, and compare it with the sample answer.
"""
)

# Use radio buttons for exercise selection
exercise_choice = st.radio("Select an Exercise:", [
    "1.1 - Effect of Reading on Vocabulary",
    "1.2 - Alcohol Consumption and Memory Loss",
    "1.3 - Training and Worker Productivity"
])

st.markdown("---")

# Define functions for each exercise for clarity and reusability

def exercise_1_1():
    st.subheader("Exercise 1.1: Ideal Experiment on Reading and Vocabulary")
    st.markdown(
        """
**Question:**  
Describe a hypothetical ideal randomized controlled experiment to study the effect of six hours of reading on the improvement of the vocabulary of high school students.  
**Follow-up:** Suggest some impediments to implementing this experiment in practice.
        """
    )
    st.text_area("Your Answer:", height=200, key="q11")
    # Use an expander to hide/reveal the sample answer
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
Randomly assign a large group of high school students to two groups:
- **Treatment Group:** Receives a structured program that mandates six hours of reading per week.
- **Control Group:** Continues with their normal reading routine.

Measure vocabulary improvement using standardized tests administered before and after the reading program.  
**Challenges:**  
- Maintaining strict adherence to the reading schedule.  
- Accounting for external factors such as students’ extracurricular reading habits.  
- Ethical issues related to forcing a regimented reading schedule on students.
            """
        )

def exercise_1_2():
    st.subheader("Exercise 1.2: Ideal Experiment on Alcohol Consumption and Memory Loss")
    st.markdown(
        """
**Question:**  
Describe a hypothetical ideal randomized controlled experiment to study the effect of alcohol consumption on long-term memory loss.  
**Follow-up:** Identify impediments to executing such an experiment.
        """
    )
    st.text_area("Your Answer:", height=200, key="q12")
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
Randomly assign participants into two groups:
- **Treatment Group:** Receives a measured dose of alcohol at controlled intervals.
- **Control Group:** Receives a placebo drink with no alcohol.

Assess memory performance using standardized cognitive tests at the beginning and end of the study period.  
**Challenges:**  
- Ethical issues and potential harm from alcohol consumption.  
- Ensuring participants comply with the study protocol.  
- Dealing with individual differences in alcohol metabolism and baseline cognitive function.
            """
        )

def exercise_1_3():
    st.subheader("Exercise 1.3: Effect of Employee Training on Productivity")
    st.markdown(
        """
**Question:**  
Examine the causal effect of hours spent on employee training (measured as hours per worker per week) on worker productivity (output per worker per hour). Provide:
1. An ideal randomized controlled experiment (RCT) design.
2. A description of an observational cross-sectional data set.
3. A description of an observational time series data set.
4. A description of an observational panel data set.
        """
    )
    st.text_area("Your Answer:", height=300, key="q13")
    with st.expander("Show Sample Answer"):
        st.markdown(
            """
**Sample Answer:**  
1. **Ideal RCT:** Randomly assign workers into two groups where one group receives additional training while the other follows the standard training program. Compare productivity after the intervention.  
2. **Cross-Sectional Data:** Collect data from various firms at a single point in time on average training hours and worker productivity along with control variables such as firm size and sector.  
3. **Time Series Data:** Track productivity and training hours at one firm over many time periods (e.g., monthly or quarterly) to analyze trends and immediate impacts.  
4. **Panel Data:** Gather data from multiple firms over several years, allowing you to control for both time-invariant characteristics and changes over time, offering a richer analysis of training impacts on productivity.
            """
        )

# Render the selected exercise
if exercise_choice.startswith("1.1"):
    exercise_1_1()
elif exercise_choice.startswith("1.2"):
    exercise_1_2()
elif exercise_choice.startswith("1.3"):
    exercise_1_3()
