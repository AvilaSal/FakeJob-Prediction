import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'categorized_job_postings.csv'
job_data = pd.read_csv(file_path)

# Separate features and target variable
X = job_data.drop('fraudulent', axis=1)
y = job_data['fraudulent']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling fake job postings in the training set
fake_indices = y_train[y_train == 1].index
real_indices = y_train[y_train == 0].index

# Number of fake jobs to oversample to
n_fake_jobs = y_train[real_indices].shape[0]

# Randomly sample from the fake jobs
oversampled_fake_indices = np.random.choice(fake_indices, size=n_fake_jobs, replace=True)

# Combine with real job indices
oversampled_indices = np.concatenate([real_indices, oversampled_fake_indices])

# New oversampled training set
X_train_oversampled = X_train.loc[oversampled_indices]
y_train_oversampled = y_train.loc[oversampled_indices]

# Initialize and fit Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_oversampled, y_train_oversampled)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

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
