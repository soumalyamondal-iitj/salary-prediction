import streamlit as st
import pandas as pd
import joblib

# Load the model from a file
model_path = "best_model.pkl"
model = joblib.load(model_path)

# Create the UI
st.title('Salary Prediction')

# Input fields for the key features
years_experience = st.number_input('Years of Experience', min_value=0, max_value=40, value=5)
education_level = st.number_input('Education Level (1 to 5)', min_value=1, max_value=5, value=3)
performance_rating = st.number_input('Performance Rating', min_value=1, max_value=5, value=3)
department = st.selectbox('Department', ['Operations', 'IT', 'Finance', 'Sales', 'Marketing'])
age = st.number_input('Age', min_value=20, max_value=65, value=30)

# Input field for skills using selectbox
skills = st.selectbox('Skills', ['C++', 'Data Analysis', 'Java', 'Python', 'Machine Learning', 'SQL'])

# Map categorical variables to numerical values
department_map = {'Operations': 0, 'IT': 1, 'Finance': 2, 'Sales': 3, 'Marketing': 4}
department_numerical = department_map[department]

# Process skills into numerical features (one-hot encoding)
#skills_list = ['C++', 'Data Analysis', 'Java', 'Python', 'Machine Learning', 'SQL']
#skills_features = [1 if skills == skill else 0 for skill in skills_list]

# Prepare input data with exactly the same features as the model expects
input_data = pd.DataFrame([[
    years_experience, education_level, performance_rating, age,
    department_numerical, skills
]], columns=[
    'YearsOfExperience', 'EducationLevel', 'PerformanceRating', 'Age',
    'Department', 'Skills'
])

# Predict Salary
if st.button('Predict Salary'):
    # Validation checks
    if not (0 <= years_experience <= 40):
        st.error('Years of Experience must be between 0 and 40.')
    elif not (1 <= performance_rating <= 5):
        st.error('Performance Rating must be between 1 and 5.')
    elif not (20 <= age <= 65):
        st.error('Age must be between 20 and 65.')
    elif not (1 <= education_level <= 5):
        st.error('Education Level must be between 1 and 5.')
    else:
        # Predict salary
        prediction = model.predict(input_data)[0]
        st.write(f'Predicted Salary: ${prediction:,.2f}')