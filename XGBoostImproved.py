import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'categorized_job_postings.csv'
job_data = pd.read_csv(file_path)

# Separate features and target variable
X = job_data.drop('fraudulent', axis=1)
y = job_data['fraudulent']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight value for imbalanced dataset
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Initialize XGBoost classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define a grid of hyperparameters to test
param_grid = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, scale_pos_weight]  # scale_pos_weight calculated from the dataset
}

# Set up GridSearchCV
grid_cv = GridSearchCV(xgb_clf, param_grid, scoring='recall', cv=3, verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_cv.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", grid_cv.best_params_)

# Train a new classifier using the best parameters found by the grid search
best_xgb_clf = grid_cv.best_estimator_

# Predict probabilities for the test set
y_scores = best_xgb_clf.predict_proba(X_test)[:, 1]

# Adjust the classification threshold
threshold = 0.8  # Let's try lowering the threshold to 0.4 and see if recall improves
y_pred_adjusted = (y_scores >= threshold).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_adjusted)
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
class_report = classification_report(y_test, y_pred_adjusted)

# Output model evaluation
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fake", "Fake"], yticklabels=["Not Fake", "Fake"])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()