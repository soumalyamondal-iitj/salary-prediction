import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

# Load the dataset
file_path = 'HR Dataset.csv'  # Change this to your file path
data = pd.read_csv(file_path)

# Drop the unnamed column and split the data into features and target
data = data.drop(columns=['Unnamed: 0', 'EmployeeID','Gender', 
                          'WorkLifeBalance','EmployeeSatisfaction','TrainingHours',
                           'YearsSinceLastPromotion','YearsWithCurrentManager',
                           'Attrition','ProjectInvolvement','LastTrainingDate'])
X = data.drop(columns=['Salary'])
y = data['Salary']

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns

# Use LabelEncoder for simplicity
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Impute missing values if any
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Normalize/scale the data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Evaluate each model using cross-validation
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    cv_scores[name] = rmse_scores.mean()

# Print the RMSE scores
print("Model Comparison Scores:")
for model_name, score in cv_scores.items():
    print(f"{model_name}: RMSE = {score}")

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_
best_rmse_rf = np.sqrt(-grid_search_rf.best_score_)

print("Best Parameters for Random Forest:", best_params_rf)
print("Best RMSE for Random Forest:", best_rmse_rf)

# Save the model, label encoders, and scaler
joblib.dump(grid_search_rf.best_estimator_, "best_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")