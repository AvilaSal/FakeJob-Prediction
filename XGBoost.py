import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'categorized_job_postings.csv'
job_data = pd.read_csv(file_path)

X = job_data.drop('fraudulent', axis=1)
y = job_data['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')

xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=["Not Fake", "Fake"], yticklabels=["Not Fake", "Fake"])
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

from sklearn.metrics import roc_curve, auc

# roc curve calculations
fpr, tpr, _ = roc_curve(y_test, xgb_clf.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# Plot feature importances in a horizontal bar chart
plt.figure(figsize=(10, 6))
feature_importance = xgb_clf.feature_importances_
sorted_idx = feature_importance.argsort()

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color='darkslateblue')
plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Top Feature Importances')
plt.gca()

plt.tight_layout()
plt.show()

accuracy_df = pd.DataFrame({"Metric": ["Accuracy"], "Value": [accuracy]})

# export the accuracy assessment as a CSV file
accuracy_df.to_csv('accuracy.csv', index=False)
