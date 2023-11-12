import pandas as pd

file_path = 'fake_job_postings.csv'
job_data = pd.read_csv(file_path)

missing_values = job_data.isnull().sum()

missing_percentage = (missing_values / len(job_data)) * 100

numerical_columns = job_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = job_data.select_dtypes(include=['object']).columns.tolist()

for col in numerical_columns:
    job_data[col].fillna(job_data[col].median(), inplace=True)

for col in categorical_columns:
    job_data[col].fillna('Unknown', inplace=True)

encoded_file_path = 'encoded_job_postings.csv'
job_data.to_csv(encoded_file_path, index=False)
