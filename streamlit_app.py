import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Chapter 1 Exercises: Economic Questions and Data",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Chapter 1: Economic Questions and Data - Exercises")
st.markdown(
    """
This section provides exercises inspired by Chapter 1 of *Introduction to Econometrics*.
Please answer the following review questions. When you're ready, click “Show Sample Answer” to compare with a sample response.
    """
)

# Sidebar selectbox for choosing an exercise
exercise_choice = st.sidebar.selectbox("Select an Exercise", [
    "Review 1.1: Effect of Reading on Vocabulary",
    "Review 1.2: Alcohol Consumption and Memory Loss",
    "Review 1.3: Causal Effect of Employee Training"
])

def exercise_1_1():
    st.header("Exercise 1.1: Ideal Experiment on Reading and Vocabulary")
    st.markdown(
        """
**Question:**  
Describe a hypothetical ideal randomized controlled experiment to study the effect of six hours of reading on the improvement of the vocabulary of high school students.  
Suggest some impediments to implementing this experiment in practice.
        """
    )
    answer1 = st.text_area("Your Answer:", height=200, key="q11")
    if st.button("Show Sample Answer", key="btn_sample_1"):
        st.markdown(
            """
**Sample Answer:**  
One approach is to randomly select a large group of high school students and then randomly assign them into two groups.  
- The **treatment group** receives a structured program that requires six hours of reading per week using selected texts.  
- The **control group** continues with their usual reading habits.  

Vocabulary improvement is measured using a standardized vocabulary test administered at the start and end of the study period.  
**Potential impediments:**  
- **Compliance and Measurement:** Ensuring that students adhere to the prescribed six hours might be challenging.  
- **External Factors:** Differences in prior reading habits, quality of reading materials, or additional support outside school might confound the results.  
- **Ethical Concerns:** Mandating or limiting reading hours may face ethical and practical issues in a real school setting.
            """
        )

def exercise_1_2():
    st.header("Exercise 1.2: Ideal Experiment on Alcohol Consumption and Memory Loss")
    st.markdown(
        """
**Question:**  
Describe a hypothetical ideal randomized controlled experiment to study the effect of the consumption of alcohol on long-term memory loss.  
Suggest some impediments to implementing this experiment in practice.
        """
    )
    answer2 = st.text_area("Your Answer:", height=200, key="q12")
    if st.button("Show Sample Answer", key="btn_sample_2"):
        st.markdown(
            """
**Sample Answer:**  
A possible experiment would involve randomly assigning participants into two groups:  
- **Treatment Group:** Receives a controlled dose of alcohol at regular intervals.  
- **Control Group:** Receives a placebo drink with no alcohol content.  

Memory performance is tested using standardized cognitive tests before and after a long study period.  
**Potential impediments:**  
- **Ethical Concerns:** Administering alcohol, especially repeatedly over a long period, could pose health risks and ethical dilemmas.  
- **Compliance and Safety:** Ensuring participants only follow the prescribed regimen and managing adverse health events.  
- **Confounding Variables:** Individual differences in alcohol tolerance, lifestyle factors, and preexisting conditions might interfere with the results.
            """
        )

def exercise_1_3():
    st.header("Exercise 1.3: Causal Effect of Employee Training on Productivity")
    st.markdown(
        """
**Question:**  
You are asked to study the causal effect of hours spent on employee training (measured in hours per worker per week) on the productivity of workers (output per worker per hour).  
Describe:  
a. An ideal randomized controlled experiment to measure this causal effect;  
b. An observational cross-sectional data set with which you could study this effect;  
c. An observational time series data set for studying this effect; and  
d. An observational panel data set for studying this effect.
        """
    )
    answer3 = st.text_area("Your Answer:", height=300, key="q13")
    if st.button("Show Sample Answer", key="btn_sample_3"):
        st.markdown(
            """
**Sample Answer:**  
- **a. Ideal Randomized Controlled Experiment:**  
  Randomly assign workers in a manufacturing plant into two groups. The treatment group receives a structured, additional training program (extra hours of training), while the control group receives the standard training regime. After a fixed period, compare the productivity (output per worker per hour) of the two groups.

- **b. Observational Cross-Sectional Data Set:**  
  Collect data from a variety of manufacturing plants at a single point in time. For each plant, record the average training hours per worker and the corresponding productivity levels. Additional variables (like plant size and industry type) would help control for confounding factors.

- **c. Observational Time Series Data Set:**  
  Use data from one manufacturing plant over multiple time periods (e.g., monthly or quarterly data) that records the number of training hours and productivity metrics. This data set allows for time trend analysis and correlation between training changes and subsequent productivity.

- **d. Observational Panel Data Set:**  
  Gather data from multiple manufacturing plants over several time periods. This panel data can help control for unobserved heterogeneity across plants by comparing within-plant changes over time in training hours and productivity.
            """
        )

# Display the chosen exercise
if exercise_choice == "Review 1.1: Effect of Reading on Vocabulary":
    exercise_1_1()
elif exercise_choice == "Review 1.2: Alcohol Consumption and Memory Loss":
    exercise_1_2()
elif exercise_choice == "Review 1.3: Causal Effect of Employee Training":
    exercise_1_3()
