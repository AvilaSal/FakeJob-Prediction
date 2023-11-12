import pandas as pd

file_path = 'encoded_job_postings.csv'
job_data = pd.read_csv(file_path)

categorical_columns = job_data.select_dtypes(include=['object']).columns.tolist()

# for high cardinality features, use label encoding
for col in categorical_columns:
    if job_data[col].nunique() > 10:
        job_data[col] = job_data[col].astype('category').cat.codes
    else:
        # Apply one-hot encoding to other categorical columns
        one_hot = pd.get_dummies(job_data[col], prefix=col, drop_first=True)
        job_data = job_data.drop(col, axis=1)
        job_data = job_data.join(one_hot)

encoded_file_path = 'categorized_job_postings.csv'
job_data.to_csv(encoded_file_path, index=False)
