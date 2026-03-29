import pandas as pd

baseline = [
    'amt',          
    'is_fraud' # target, ne ide u features naravno
]
demographics = [
    'gender',
    'age',
    'city_pop', # da li korisnik zivi u velikom/malom gradu
    'job',
]
geography = [
    'lat', 'long', # lokacija korisnika
    'distance_km' # izracunato rastojanje od korisnika i merchanta
    'state',
    'zip',
]
transaction = [
    'category',     # tip transakcije (food, travel, shopping..)
]
temporal = [
    'trans_time_hrs',
    'trans_time_is_night',
    'trans_time_day',
    'trans_date_is_weekend',
]
customer_behavior = [
    'customer_num_trans_1_day',
    'customer_num_trans_7_day',
    'customer_num_trans_30_day',
    'customer_avg_amout_1_day',
    'customer_avg_amount_7_day',
    'customer_avg_amount_30_day',
    'amt_vs_avg_7d', 'amt_vs_avg_30d' # izracunato
]
merchant_profile = [
    'merchant_num_trans_1_day',
    'merchant_num_trans_7_day',
    'merchant_num_trans_30_day',
]
merchant_risk = [
    'merchant_risk_1_day',
    'merchant_risk_7_day',
    'merchant_risk_30_day',
    'merchant_risk_90_day',
]

feature_groups = {
    'baseline':              baseline,
    'baseline+demo':         baseline + demographics,
    'baseline+geo':          baseline + geography,
    'baseline+temporal':     baseline + temporal,
    'baseline+customer':     baseline + customer_behavior,
    'baseline+merchant':     baseline + merchant_profile + merchant_risk,
    'all':                   baseline + demographics + geography + 
                             transaction + temporal + customer_behavior + 
                             merchant_profile + merchant_risk,
}