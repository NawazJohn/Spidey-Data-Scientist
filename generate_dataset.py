import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 1000

# Create dummy data
data = {
    'PassengerId': np.arange(1001, 1001 + n_samples),
    'age': np.random.normal(35, 10, n_samples).astype(int),
    'salary': np.random.normal(60000, 15000, n_samples),
    'department': np.random.choice(['IT', 'HR', 'Marketing', 'Sales', 'Finance'], n_samples),
    'years_at_company': np.random.uniform(0, 15, n_samples).astype(int),
    'performance_score': np.random.uniform(1, 10, n_samples),
}

df = pd.DataFrame(data)

# Introduce missing values and dummy sentinel values (?, -999, None)
df.loc[np.random.choice(df.index, size=65, replace=False), 'salary'] = np.nan
df.loc[np.random.choice(df.index, size=40, replace=False), 'department'] = '?'
df.loc[np.random.choice(df.index, size=50, replace=False), 'performance_score'] = np.nan
df.loc[np.random.choice(df.index, size=25, replace=False), 'age'] = -999

# Target variable
df['promoted'] = np.where(
    (df['performance_score'] > 7.0) & (df['years_at_company'] > 2), 1, 0
)

# Introduce 35 explicit duplicate rows (duplicating existing rows)
dup_indices = np.random.choice(df.index, size=35, replace=False)
duplicates = df.loc[dup_indices].copy()
# To make them exact full duplicates, duplicate entire row including or excluding ID
df = pd.concat([df, duplicates], ignore_index=True)

# Save to CSV
output_path = 'dummy_dataset.csv'
df.to_csv(output_path, index=False)
print(f"Successfully generated {output_path} with {len(df)} rows and {35} duplicates!")

