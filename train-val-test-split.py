import pandas as pd
import numpy as np

drop_always = [
    'ssn', 'cc_num', 'first', 'last', 'acct_num',
    'street', 'trans_num', 'dob', 'profile',
    'trans_date', 'trans_time',  # već ima izvedene kolone
    'trans_time_secs',           # redundantno sa trans_time_hrs
    'unix_time',                 # koristio si ga samo za sortiranje
    'city',                      # previsoko kardinalitet
    'merchant',                  # previsoko kardinalitet, merchant_risk je bolji proxy
]

# Haversine funkcija za udaljenost u km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # poluprecnik Zemlje u km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df = pd.read_parquet("data.parquet")
df = df.sort_values('unix_time').reset_index(drop=True)
df = df.drop(columns=drop_always)

df['distance_km'] = haversine(
    df['lat'], df['long'],
    df['merch_lat'], df['merch_long']
).astype('float32')

df['amt_vs_avg_7d'] = (df['amt'] / (df['customer_avg_amount_7_day'] + 1)).astype('float32')
df['amt_vs_avg_30d'] = (df['amt'] / (df['customer_avg_amount_30_day'] + 1)).astype('float32')

df = df.drop(columns=["merch_lat","merch_long"]) # dropujemo merch lokaciju, nije toliko korisno naspram lokacije korisnika

# 60% train, 20% val, 20% test
n = len(df)
train_end = int(n * 0.60)
val_end   = int(n * 0.80)

train = df.iloc[:train_end].reset_index(drop=True)
val   = df.iloc[train_end:val_end].reset_index(drop=True)
test  = df.iloc[val_end:].reset_index(drop=True)

target = 'is_fraud'

X_train = train.drop(columns=[target])
y_train = train[target]

X_val = val.drop(columns=[target])
y_val = val[target]

X_test = test.drop(columns=[target])
y_test = test[target]

print(f"Train: {len(train):,} uzoraka | Fraud: {y_train.mean()*100:.3f}%")
print(f"Val:   {len(val):,} uzoraka | Fraud: {y_val.mean()*100:.3f}%")
print(f"Test:  {len(test):,} uzoraka | Fraud: {y_test.mean()*100:.3f}%")

train.to_parquet('train.parquet', index=False)
val.to_parquet('val.parquet',     index=False)
test.to_parquet('test.parquet',   index=False)
