import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV

file_path = 'categorized_job_postings.csv'
job_data = pd.read_csv(file_path)

X = job_data.drop('fraudulent', axis=1)
y = job_data['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hyperparameters
gamma_value = 2
min_child_weight_value = 10
scale_pos_weight_value = 10

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# implement ADASYN
adasyn = ADASYN(random_state=42)

# resample training data
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# grid search for best parameters
param_grid = {
    'gamma': [0.1, 0.5, 1, 2],
    'min_child_weight': [1, 5, 10],
    'scale_pos_weight': [1, 2, 5, 10]
}

xgb_clf = xgb.XGBClassifier(
    gamma=gamma_value,
    min_child_weight=min_child_weight_value,
    scale_pos_weight=scale_pos_weight_value,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_train_resampled, y_train_resampled)

# Set up the grid search with cross-validation
grid_cv = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='recall', cv=3, verbose=2)

grid_cv.fit(X_train, y_train)
# print best parameters
print("Best parameters found: ", grid_cv.best_params_)

y_pred = xgb_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# accuracy report
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# confusion matrix vis
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fake", "Fake"], yticklabels=["Not Fake", "Fake"])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
