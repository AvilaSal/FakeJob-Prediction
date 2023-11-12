import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

file_path = 'categorized_job_postings.csv'
job_data = pd.read_csv(file_path)

X = job_data.drop('fraudulent', axis=1)  # 'fraudulent' is the target variable
y = job_data['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

selector = SelectFromModel(rf, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[(selector.get_support())]
print("Selected Features:", selected_features)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# now use the random forest classifier with the selected features (important features)
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the selected features on the training set
rf_selected.fit(X_train_selected, y_train)

# Predict on the test set
y_pred = rf_selected.predict(X_test_selected)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# code to visualize confusion matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict on the test set
y_pred = rf_selected.predict(X_test_selected)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fake", "Fake"], yticklabels=["Not Fake", "Fake"])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()