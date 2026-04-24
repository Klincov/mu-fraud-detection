import pandas as pd

train = pd.read_parquet("train.parquet")
test  = pd.read_parquet("test.parquet")
val = pd.read_parquet("val.parquet")

# Fraud rate po kategoriji u train vs test
train_rate = train.groupby('category')['is_fraud'].mean().rename('train_fraud_rate')
test_rate  = test.groupby('category')['is_fraud'].mean().rename('test_fraud_rate')
val_rate  = val.groupby('category')['is_fraud'].mean().rename('val_fraud_rate')


print(pd.concat([train_rate, test_rate,val_rate], axis=1).round(4))