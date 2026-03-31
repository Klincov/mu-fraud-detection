import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_recall_curve,roc_auc_score, average_precision_score, precision_recall_fscore_support
import joblib

# ── Feature grupe ────────────────────────────────────────────────────────────
TARGET_COL = "is_fraud"

baseline = ['amt']

demographics = ['gender', 'city_pop', 'job']

geography = ['lat', 'long', 'distance_km', 'state', 'zip']

transaction = ['category'] # ovo je lose u datasetu... , ne koristi nikad

temporal = ['trans_time_hrs', 'trans_time_is_night', 'trans_time_day', 'trans_date_is_weekend']

customer_behavior = [
    'customer_num_trans_1_day', 'customer_num_trans_7_day', 'customer_num_trans_30_day',
    'customer_avg_amout_1_day', 'customer_avg_amount_7_day', 'customer_avg_amount_30_day',
    'amt_vs_avg_7d', 'amt_vs_avg_30d'
]

merchant_profile = ['merchant_num_trans_1_day', 'merchant_num_trans_7_day', 'merchant_num_trans_30_day']

merchant_risk = ['merchant_risk_1_day', 'merchant_risk_7_day', 'merchant_risk_30_day', 'merchant_risk_90_day']

feature_groups = {
    'baseline':          baseline,
    'baseline+demo':     baseline + demographics,
    'baseline+geo':      baseline + geography,
    'baseline+temporal': baseline + temporal,
    'baseline+customer': baseline + customer_behavior,
    'baseline+merchant': baseline + merchant_profile + merchant_risk,
    'baseline+transaction': baseline + transaction,
    'all':               baseline + demographics + geography +
                         temporal + customer_behavior + merchant_profile + merchant_risk,
    'all_bez_merchant_risk': baseline + demographics + geography +
                             temporal + customer_behavior + merchant_profile,
    'samo_merchant_risk':    baseline + merchant_risk,
    'all_bez_geo':      baseline + demographics + 
                        temporal + customer_behavior + 
                        merchant_profile + merchant_risk,
    'all_bez_customer': baseline + demographics + geography + temporal + 
                        merchant_profile + merchant_risk,
}

# samo ovo menjaj
ACTIVE_GROUP = 'baseline'

ALL_CAT_COLS = ['gender', 'state', 'job', 'category']

def find_best_threshold(model, X_val, y_val, cat_indices):
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    proba = model.predict_proba(val_pool)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    
    print(f"Optimalan prag:  {thresholds[best_idx]:.4f}")
    print(f"Precision:       {precisions[best_idx]:.4f}")
    print(f"Recall:          {recalls[best_idx]:.4f}")
    print(f"F1:              {f1_scores[best_idx]:.4f}")
    
    return thresholds[best_idx]

def get_active_features():
    features = feature_groups[ACTIVE_GROUP]
    cat_features = [c for c in ALL_CAT_COLS if c in features]
    return features, cat_features

def load_data(features, cat_cols):
    cols_to_load = features + [TARGET_COL]

    train = pd.read_parquet("train.parquet", columns=cols_to_load)
    val   = pd.read_parquet("val.parquet",   columns=cols_to_load)
    test  = pd.read_parquet("test.parquet",  columns=cols_to_load)

    for col in cat_cols:
        train[col] = train[col].astype("str")
        val[col]   = val[col].astype("str")
        test[col]  = test[col].astype("str")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_val   = val.drop(columns=[TARGET_COL])
    y_val   = val[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]

    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]

    return X_train, y_train, X_val, y_val, X_test, y_test, cat_indices

def train_catboost(X_train, y_train, X_val, y_val, cat_indices):
    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weights = [1.0, neg / pos]

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_indices)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.05,
        depth=6,
        iterations=2000,
        random_seed=42,
        verbose=200,
        class_weights=class_weights,
        task_type="GPU",
        devices="0",
        early_stopping_rounds=100
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model

def evaluate(model, X_val, y_val, cat_indices, threshold=0.5):
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    proba    = model.predict_proba(val_pool)[:, 1]

    roc = roc_auc_score(y_val, proba)
    pr  = average_precision_score(y_val, proba)
    y_pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, pos_label=1, average="binary"
    )
    return {"roc_auc": roc, "pr_auc": pr, "precision": precision,
            "recall": recall, "f1": f1, "threshold": threshold}

def main():
    features, cat_cols = get_active_features()
    print(f"Aktivna grupa: '{ACTIVE_GROUP}' | Features: {len(features)} | Cat: {cat_cols}\n")

    X_train, y_train, X_val, y_val, X_test, y_test, cat_indices = load_data(features, cat_cols)
    model = train_catboost(X_train, y_train, X_val, y_val, cat_indices)

    # Validacioni set -- koristiš tokom razvoja
    val_metrics = evaluate(model, X_val, y_val, cat_indices, threshold=find_best_threshold(model,X_val,y_val,cat_indices))
    print(f"\nCatBoost [{ACTIVE_GROUP}] VALIDATION METRICS:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Test set -- konačna ocena, gledaš ga jednom na kraju
    test_metrics = evaluate(model, X_test, y_test, cat_indices, threshold=find_best_threshold(model, X_test, y_test, cat_indices))
    print(f"\nCatBoost [{ACTIVE_GROUP}] TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Sačuvaj oboje
    joblib.dump(model, f"catboost_{ACTIVE_GROUP}.pkl")
    pd.DataFrame([val_metrics, test_metrics], index=["val", "test"]).to_json(
        f"metrics/catboost_{ACTIVE_GROUP}_metrics.json"
    )
if __name__ == "__main__":
    main()