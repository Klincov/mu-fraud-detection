import pandas as pd
import numpy as np

train = pd.read_parquet("train.parquet")

# Proveri NaN
print("=== NaN vrednosti ===")
nan_cols = train.isnull().sum()
print(nan_cols[nan_cols > 0])

# Proveri inf vrednosti (float kolone)
print("\n=== Inf vrednosti ===")
for col in train.select_dtypes(include=[np.floating]).columns:
    inf_count = np.isinf(train[col]).sum()
    if inf_count > 0:
        print(f"{col}: {inf_count} inf vrednosti")

print(train['category'].unique())