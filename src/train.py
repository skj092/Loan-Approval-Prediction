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

# Load the data
path = Path("/home/sonujha/rnd/Loan-Approval-Prediction/data/")
df = pd.read_csv(path/'train.csv')

# Convert categorical variables into numerical variables
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
le = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)

# Define features and target
X = df.drop(['id', 'loan_status'], axis=1)
y = df['loan_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter tuning space for Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform hyperparameter tuning for Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Train the best Random Forest Classifier model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)
print("Classification Report:")
print(classification_report(y_test, (y_pred >= 0.5).astype(int)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, (y_pred >= 0.5).astype(int)))

# Make predictions on the test set
test_df = pd.read_csv(path/'test.csv')
test_df[categorical_cols] = test_df[categorical_cols].apply(le.fit_transform)
# drop id before scaling
test_df = scaler.transform(test_df.drop('id', axis=1))
predictions = best_model.predict_proba(test_df)[:, 1]

# Save the predictions to a submission file
submission_df = pd.DataFrame({'id': range(len(predictions)), 'loan_status': predictions})
submission_df.to_csv('submission.csv', index=False)
