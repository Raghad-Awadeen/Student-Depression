import streamlit as st
import pickle
import numpy as np

st.markdown(
    """
    <style>
        /* Background Color */
        .stApp {
            background-color: #fff5ee;
        }

        /* Header Styling */
        h1, h2, h3 {
            color: #2C3E50;
            text-align: center;
        }

        /* Label and Text Styling */
        label, p {
            font-size: 18px !important;
            color: #34495E !important;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #ecbca7 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 10px;
        }

        div.stButton > button:hover {
            background-color: #b5caf3 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    with open("depression_classifier.pkl", 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("🌿 AI-Powered Student Depression Detection")
st.write("Enter your details and let an AI-powered model analyze your mental health.")

age = st.number_input("Age", min_value=18, max_value=34, value=20)
Academic_Pressure = st.slider("Academic Pressure", 1, 5)

sleep_hours_options = {"Less than 5 hours": 1, "5-6 hours": 2, "7-8 hours": 3, "More than 8 hours": 4}
sleep_hours_label = st.selectbox("Sleep Duration", list(sleep_hours_options.keys()))
sleep_hours_value = sleep_hours_options[sleep_hours_label]  # Convert to numeric

Dietary_Habits_options = {"Unhealthy": 3, "Moderate": 2, "Healthy": 1}
Dietary_Habits_label = st.selectbox("Dietary Habits", list(Dietary_Habits_options.keys()))
Dietary_Habits_value = Dietary_Habits_options[Dietary_Habits_label]  # Convert to numeric

study_hours = st.slider("Daily Study Hours", 0, 12, 6)

Family_History_options = {"No": 0, "Yes": 1}
Family_History_label = st.selectbox("Family History of Mental Illness", list(Family_History_options.keys()))
Family_History_value = Family_History_options[Family_History_label]  # Convert to numeric

suicidal_thoughts_options = {"No": 0, "Yes": 1}
suicidal_thoughts_label = st.selectbox("Have you ever had suicidal thoughts?", list(suicidal_thoughts_options.keys()))
suicidal_thoughts_value = suicidal_thoughts_options[suicidal_thoughts_label]  # Convert to numeric

Financial_Stress = st.slider("Financial Stress", 1, 5)
Study_Satisfaction = st.slider("Study Satisfaction", 1, 5)

cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=4.0, step=0.1, format="%.2f")

input_data = np.array([[age, Academic_Pressure, study_hours, sleep_hours_value,
                         Dietary_Habits_value, Family_History_value, suicidal_thoughts_value,
                         Financial_Stress, Study_Satisfaction, cgpa]])


if st.button("Check Depression Status"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 The AI suggests the student may be depressed. Please seek support.")
    else:
        st.success("✅ The AI does not detect signs of depression.")
