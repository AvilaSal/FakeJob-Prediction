import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'categorized_job_postings.csv'
job_data = pd.read_csv(file_path)

# Splitting the dataset into features and target variable
X = job_data.drop('fraudulent', axis=1)
y = job_data['fraudulent']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluation
def evaluate_model(model_name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy}\n")
    print(f"{model_name} - Confusion Matrix:\n{conf_matrix}\n")
    print(f"{model_name} - Classification Report:\n{class_report}\n")

evaluate_model("Logistic Regression", y_test, y_pred_log_reg)
evaluate_model("Support Vector Machine", y_test, y_pred_svm)
