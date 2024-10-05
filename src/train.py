from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
import numpy as np

# Load the data
start_time = time.time()
path = Path("/home/sonujha/rnd/Loan-Approval-Prediction/data/")
df = pd.read_csv(path/'train.csv')
print(f"Loaded the data in {time.time() - start_time:.2f} seconds")

# Convert categorical variables into numerical variables
start_time = time.time()
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
le = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)
print(f"Converted categorical variables into numerical variables in {time.time() - start_time:.2f} seconds")

# Define features and target
start_time = time.time()
X = df.drop(['id', 'loan_status'], axis=1)
y = df['loan_status']
print(f"Defined features and target in {time.time() - start_time:.2f} seconds")

# Split the data into training and testing sets
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Split the data into training and testing sets in {time.time() - start_time:.2f} seconds")

# Scale the data
start_time = time.time()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"Scaled the data in {time.time() - start_time:.2f} seconds")

# Define hyperparameter tuning space for Random Forest Classifier
start_time = time.time()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
print(f"Defined hyperparameter tuning space for Random Forest Classifier in {time.time() - start_time:.2f} seconds")

# Perform hyperparameter tuning for Random Forest Classifier
start_time = time.time()
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Performed hyperparameter tuning for Random Forest Classifier in {time.time() - start_time:.2f} seconds")

# Train the best Random Forest Classifier model
start_time = time.time()
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
print(f"Trained the best Random Forest Classifier model in {time.time() - start_time:.2f} seconds")

# Make predictions
start_time = time.time()
y_pred = best_model.predict_proba(X_test)[:, 1]
print(f"Made predictions in {time.time() - start_time:.2f} seconds")

# Evaluate the model
start_time = time.time()
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)
print("Classification Report:")
print(classification_report(y_test, (y_pred >= 0.5).astype(int)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, (y_pred >= 0.5).astype(int)))
print(f"Evaluating the model in {time.time() - start_time:.2f} seconds")

# Make predictions on the test set
start_time = time.time()
test_df = pd.read_csv(path/'test.csv')
test_df[categorical_cols] = test_df[categorical_cols].apply(le.fit_transform)
test_ids = test_df['id']
test_df = pd.DataFrame(scaler.transform(test_df.drop('id', axis=1)), columns=test_df.drop('id', axis=1).columns)
predictions = best_model.predict_proba(test_df)[:, 1]
print(f"Made predictions on the test set in {time.time() - start_time:.2f} seconds")

# Save the predictions to a submission file
start_time = time.time()
submission_df = pd.DataFrame({'id': test_ids, 'loan_status': predictions})
submission_df.to_csv('submission.csv', index=False)
print(f"Saved the predictions to a submission file in {time.time() - start_time:.2f} seconds")
