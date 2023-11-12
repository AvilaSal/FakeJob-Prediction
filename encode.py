import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'fake_job_postings.csv'
job_data = pd.read_csv(file_path)

# Handle missing values (similar to your code)
numerical_columns = job_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = job_data.select_dtypes(include=['object']).columns.tolist()

for col in numerical_columns:
    job_data[col].fillna(job_data[col].median(), inplace=True)

for col in categorical_columns:
    job_data[col].fillna('Unknown', inplace=True)

# count the number of real and fake job postings
class_counts = job_data['fraudulent'].value_counts()

plt.figure(figsize=(8, 6))
bars = plt.bar(['Real (0)', 'Fake (1)'], class_counts.values, color=['royalblue', 'indianred'])
plt.title('Job Imbalance Visualization', fontsize=16)
plt.xlabel('Job Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.05, bar.get_height() + 100, str(count), fontsize=12, color='black')

plt.show()

# Display summary statistics
summary_stats = job_data[numerical_columns].describe()
print("Summary Statistics for Numerical Columns:")
print(summary_stats)