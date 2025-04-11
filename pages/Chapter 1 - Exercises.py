# pages/Chapter 1 - Exercises.py

import streamlit as st

st.set_page_config(
    page_title="Chapter 1: Exercises",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Chapter 1: Economic Questions and Data")
st.markdown(
    """
This page includes review exercises from Chapter 1 of *Introduction to Econometrics*.  
Use the dropdown below to select an exercise.
"""
)

exercise_choice = st.selectbox("Choose an Exercise:", [
    "1.1 - Effect of Reading on Vocabulary",
    "1.2 - Alcohol & Memory Loss",
    "1.3 - Training and Worker Productivity"
])

def exercise_1_1():
    st.subheader("1.1 Ideal Experiment on Reading and Vocabulary")
    st.markdown("""
**Question:**  
Describe a hypothetical ideal randomized controlled experiment to study the effect of six hours of reading on the improvement of the vocabulary of high school students.  
Suggest some impediments to implementing this experiment in practice.
""")
    st.text_area("Your Answer", height=200, key="q11")
    if st.button("Show Sample Answer", key="btn_sample_1"):
        st.markdown("""
**Sample Answer:**  
Randomly assign high school students to two groups: one reads for 6 hours weekly, the other doesn’t. Use vocabulary tests before and after.  
**Challenges:** Compliance, external factors, and ethical issues.
        """)

def exercise_1_2():
    st.subheader("1.2 Ideal Experiment on Alcohol and Memory")
    st.markdown("""
**Question:**  
Describe a hypothetical ideal randomized controlled experiment to study the effect of the consumption of alcohol on long-term memory loss.  
Suggest some impediments to implementing this experiment in practice.
""")
    st.text_area("Your Answer", height=200, key="q12")
    if st.button("Show Sample Answer", key="btn_sample_2"):
        st.markdown("""
**Sample Answer:**  
Randomly assign participants to treatment (alcohol) and control groups. Test memory before and after over time.  
**Challenges:** Ethics, legality, variability in alcohol response.
        """)

def exercise_1_3():
    st.subheader("1.3 Training and Productivity")
    st.markdown("""
**Question:**  
You are asked to study the causal effect of hours spent on employee training on worker productivity. Describe:  
a. An ideal RCT  
b. Cross-sectional data  
c. Time-series data  
d. Panel data
""")
    st.text_area("Your Answer", height=300, key="q13")
    if st.button("Show Sample Answer", key="btn_sample_3"):
        st.markdown("""
**Sample Answer:**  
- a. RCT: Randomly assign training to some workers. Measure output.  
- b. Cross-sectional: Survey different firms at one time.  
- c. Time series: Track one plant’s training and productivity over time.  
- d. Panel: Track many plants over many years.
        """)

# Display selected exercise
if "1.1" in exercise_choice:
    exercise_1_1()
elif "1.2" in exercise_choice:
    exercise_1_2()
elif "1.3" in exercise_choice:
    exercise_1_3()
