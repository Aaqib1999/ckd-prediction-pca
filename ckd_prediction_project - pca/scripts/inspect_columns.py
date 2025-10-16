import pandas as pd

# Load your raw dataset
df = pd.read_csv("data/kidney_disease_mmm.csv")

# Print column names
print("Column names:", df.columns.tolist())

# Show first few rows
print("\nFirst few rows:")
print(df.head())

# Check values in target column (guessing it's 'classification')
if 'classification' in df.columns:
    print("\nUnique values in classification column:", df['classification'].unique())
else:
    print("\n'classification' column NOT found! Please confirm the actual target column name.")
