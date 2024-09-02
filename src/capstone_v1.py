
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
import joblib

# Load dataset
df = pd.read_csv("/content/sample_data/HR Dataset.csv")

# Display basic information
df.info()
df.describe()

# Visualize the distribution of salaries
sns.histplot(df['Salary'], kde=True)
plt.title('Distribution of Salaries')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Drop columns that are not needed for prediction
df = df.drop(columns=['Attrition', 'ProjectInvolvement', 'LastTrainingDate'])

# Split the data into training and testing sets
X = df.drop(['Salary'], axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical data
numeric_features = ['YearsOfExperience', 'EducationLevel', 'PerformanceRating', 'Age']
numeric_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_features = ['Department', 'Skills', 'Gender']
categorical_transformer = OneHotEncoder(drop='first')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Track experiments with MLflow
mlflow.set_experiment('data-science-salaries')

best_model = None
best_score = float('inf')
best_model_name = ""

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # Log metrics
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)

        # Log model
        mlflow.sklearn.log_model(pipeline, model_name)

        print(f'{model_name} - RMSE: {rmse}, MAE: {mae}, R2: {r2}')

        # Determine if this is the best model so far
        if rmse < best_score:
            best_score = rmse
            best_model = pipeline
            best_model_name = model_name

print(f"The best model is {best_model_name} with RMSE: {best_score}")

# Example for Random Forest hyperparameter tuning
if best_model_name == "Random Forest":
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30]
    }

    grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Log best parameters and results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Negate because scoring='neg_mean_squared_error'

    with mlflow.start_run(run_name='Random Forest Hyperparameter Tuning'):
        mlflow.log_params(best_params)
        mlflow.log_metric('best_score', best_score)
        mlflow.sklearn.log_model(grid_search.best_estimator_, 'best_random_forest')

    # Save the trained model
    joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
else:
    # Save the best trained model
    joblib.dump(best_model, 'best_model.pkl')

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
skills = st.selectbox('Skill (Select one)', ['None', 'C++', 'Data Analysis', 'Java', 'Python', 'Machine Learning', 'SQL'])

# Map categorical variables to numerical values
department_map = {'Operations': 0, 'IT': 1, 'Finance': 2, 'Sales': 3, 'Marketing': 4}
department_numerical = department_map[department]

# Process skills into numerical features (one-hot encoding)
skills_list = ['C++', 'Data Analysis', 'Java', 'Python', 'Machine Learning', 'SQL']
skills_features = [1 if skills == skill else 0 for skill in skills_list]

# Prepare input data with exactly the same features as the model expects
input_data = pd.DataFrame([[
    years_experience, education_level, performance_rating, age,
    department_numerical, *skills_features
]], columns=[
    'YearsOfExperience', 'EducationLevel', 'PerformanceRating', 'Age',
    'Department', *skills_list
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