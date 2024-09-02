import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and preprocessors
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Define the Streamlit app
def main():
    st.title("Salary Prediction App")

    # User inputs
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    #gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ["Sales", "Finance", "Operations", "HR", "IT", "Admin"])
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    education_level = st.selectbox("Education Level", [1, 2, 3, 4])
    performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4, 5])
    #work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4, 5])
    #employee_satisfaction = st.selectbox("Employee Satisfaction", [1, 2, 3, 4, 5])
    #training_hours = st.number_input("Training Hours", min_value=0, max_value=200, value=50)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=40, value=5)
    #years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=40, value=5)
    #years_with_current_manager = st.number_input("Years with Current Manager", min_value=0, max_value=40, value=5)
    #attrition = st.selectbox("Attrition", ["Yes", "No"])
    #project_involvement = st.selectbox("Project Involvement", ["Project A", "Project B", "Project C", "Project D"])
    skills = st.selectbox("Skills", ["Java", "Python", "C++", "Data Analysis"])
    training_effectiveness = st.selectbox("Training Effectiveness", [1, 2, 3, 4, 5])
    certification = st.selectbox("Certification", ["Certified", "Not Certified"])

    # Preprocess the inputs
    input_data = np.array([[age, department, years_of_experience, education_level, performance_rating,
                            years_in_current_role, skills, training_effectiveness, certification]])

    input_df = pd.DataFrame(input_data, columns=['Age', 'Department', 'YearsOfExperience', 'EducationLevel',
                                                 'PerformanceRating', 'YearsInCurrentRole', 'Skills',
                                                 'TrainingEffectiveness', 'Certification'])

    # Apply the same preprocessing as the training data
    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_df = scaler.transform(input_df)

    if st.button('Predict Salary'):
        # Predict the salary
        prediction = model.predict(input_df)
        # Display the prediction
        st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()