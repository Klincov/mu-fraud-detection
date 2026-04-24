import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

sns.set_theme(style="whitegrid")

data = pd.read_parquet("data.parquet")

# 1. Distribucija fraud vs non-fraud
plt.figure(figsize=(6,4))
ax = (data['is_fraud'].value_counts(normalize=True) * 100)\
        .sort_index()\
        .plot(kind='bar', color=['gray', 'red'])
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
plt.title("Distribucija transakcija (fraud vs non-fraud)")
plt.xlabel("Fraud (0 = ne, 1 = da)")
plt.ylabel("Broj uzoraka")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Distribucija iznosa

# Non-fraud
plt.figure(figsize=(6,4))
sns.boxplot(y=data[data['is_fraud'] == 0]['amt'])
plt.title("Distribucija iznosa - Non Fraud")
plt.ylabel("Iznos transakcije")
plt.tight_layout()
plt.show()

# Fraud
plt.figure(figsize=(6,4))
sns.boxplot(y=data[data['is_fraud'] == 1]['amt'])
plt.title("Distribucija iznosa - Fraud")
plt.ylabel("Iznos transakcije")
plt.tight_layout()
plt.show()

# 3. Fraud rate po dobu dana
plt.figure(figsize=(8,4))
fraud_by_hour = data.groupby('trans_time_hrs')['is_fraud'].mean()
fraud_by_hour.plot()
plt.title("Fraud rate po satu u danu")
plt.xlabel("Sat u danu")
plt.ylabel("Prosečan fraud rate")
plt.xticks(range(0,24))
plt.tight_layout()
plt.show()

# 4. Fraud rate vikend vs radni dan
plt.figure(figsize=(6,4))
fraud_weekend = data.groupby('trans_date_is_weekend')['is_fraud'].mean()
fraud_weekend.plot(kind='bar')
plt.title("Fraud rate: vikend vs radni dan")
plt.xlabel("0 = radni dan, 1 = vikend")
plt.ylabel("Prosečan fraud rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 5. Korelaciona matrica
plt.figure(figsize=(10,8))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Korelaciona matrica")
plt.tight_layout()
plt.show()