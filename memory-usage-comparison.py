import pandas as pd


skip_cols = ['ssn', 'cc_num', 'first', 'last', 'acct_num', 
             'street', 'trans_num', 'dob']

dtypes = {
    # Kategoričke string kolone -- 'category' štedi dosta memorije
    'gender':    'category',
    'city':      'category',
    'state':     'category',
    'job':       'category',
    'profile':   'category',
    'category':  'category',
    'merchant':  'category',
    'trans_date': 'category',
    'trans_time': 'category',

    # Lokacija
    'zip':        'int32',
    'lat':        'float32',
    'long':       'float32',
    'city_pop':   'int32',
    'merch_lat':  'float32',
    'merch_long': 'float32',

    # Transakcija
    'unix_time':  'int64',   # timestamp, mora int64
    'amt':        'float32',
    'is_fraud':   'int8',    # samo 0 i 1

    # Broj transakcija po periodu
    'customer_num_trans_1_day':  'int16',
    'customer_num_trans_7_day':  'int16',
    'customer_num_trans_30_day': 'int16',

    # Prosečni iznosi
    'customer_avg_amout_1_day':   'float32',
    'customer_avg_amount_7_day':  'float32',
    'customer_avg_amount_30_day': 'float32',

    # Merchant transakcije
    'merchant_num_trans_1_day':  'float32',
    'merchant_num_trans_7_day':  'float32',
    'merchant_num_trans_30_day': 'float32',

    # Merchant risk
    'merchant_risk_1_day':  'int16',
    'merchant_risk_7_day':  'int16',
    'merchant_risk_30_day': 'int16',
    'merchant_risk_90_day': 'int16',

    # Temporalne kolone
    'trans_time_secs':        'int32',
    'trans_time_hrs':         'int8',   # 0-23
    'trans_time_is_night':    'int8',   # 0 ili 1
    'trans_time_day':         'int8',   # 1-31
    'trans_date_is_weekend':  'int8',   # 0 ili 1
}


all_cols = pd.read_csv('data.csv', nrows=0).columns.tolist()
use_cols = [c for c in all_cols if c not in skip_cols]

# Učitavanje CSV sa optimizacijom
# df = pd.read_csv('data.csv', usecols=use_cols, dtype=dtypes)

# Parquet (brže):
df = pd.read_parquet('data.parquet', columns=use_cols).astype(dtypes)

df.info(memory_usage='deep')