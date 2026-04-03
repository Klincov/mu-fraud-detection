import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, precision_recall_fscore_support
import joblib

#Feature grupe
TARGET_COL = "is_fraud"

baseline = ['amt']
demographics = ['gender', 'city_pop', 'job']
geography = ['lat', 'long', 'distance_km', 'state', 'zip']
transaction = ['category']  # loše u ovom datasetu, ne koristiti
temporal = ['trans_time_hrs', 'trans_time_is_night', 'trans_time_day', 'trans_date_is_weekend']
customer_behavior = [
    'customer_num_trans_1_day', 'customer_num_trans_7_day', 'customer_num_trans_30_day',
    'customer_avg_amout_1_day', 'customer_avg_amount_7_day', 'customer_avg_amount_30_day',
    'amt_vs_avg_7d', 'amt_vs_avg_30d'
]
merchant_profile = ['merchant_num_trans_1_day', 'merchant_num_trans_7_day', 'merchant_num_trans_30_day']
merchant_risk = ['merchant_risk_1_day', 'merchant_risk_7_day', 'merchant_risk_30_day', 'merchant_risk_90_day']

feature_groups = {
    'baseline':              baseline,
    'baseline+demo':         baseline + demographics,
    'baseline+geo':          baseline + geography,
    'baseline+temporal':     baseline + temporal,
    'baseline+customer':     baseline + customer_behavior,
    'baseline+merchant':     baseline + merchant_profile + merchant_risk,
    'all':                   baseline + demographics + geography +
                             temporal + customer_behavior + merchant_profile + merchant_risk,
    'all_bez_merchant_risk': baseline + demographics + geography +
                             temporal + customer_behavior + merchant_profile,
    'samo_merchant_risk':    baseline + merchant_risk,
    'all_bez_geo':           baseline + demographics +
                             temporal + customer_behavior + merchant_profile + merchant_risk,
    'all_bez_customer':      baseline + demographics + geography +
                             temporal + merchant_profile + merchant_risk,
}

#Ovo menjaj
ACTIVE_GROUP = 'baseline'
ACTIVE_MODEL  = 'catboost'  # 'catboost' ili 'lightgbm'


ALL_CAT_COLS = ['gender', 'state', 'job']  # category izbačena

def find_best_threshold(proba, y_true):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    print(f"  Optimalan prag:  {thresholds[best_idx]:.4f}")
    print(f"  Precision:       {precisions[best_idx]:.4f}")
    print(f"  Recall:          {recalls[best_idx]:.4f}")
    print(f"  F1:              {f1_scores[best_idx]:.4f}")
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
        early_stopping_rounds=100,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model

def predict_catboost(model, X, y, cat_indices):
    pool = Pool(X, y, cat_features=cat_indices)
    return model.predict_proba(pool)[:, 1]


def train_lightgbm(X_train, y_train, X_val, y_val, cat_indices):
    pos = y_train.sum()
    neg = len(y_train) - pos

    # LightGBM koristi imena kolona za kategorijske, ne indekse
    cat_col_names = [X_train.columns[i] for i in cat_indices]
    for col in cat_col_names:
        X_train[col] = X_train[col].astype("category")
        X_val[col]   = X_val[col].astype("category")

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=63,          
        max_depth=6,
        #scale_pos_weight=neg / pos,  # ekvivalent class_weights
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=200),
        ],
    )
    return model, cat_col_names

def predict_lightgbm(model, X, cat_col_names):
    X = X.copy()
    for col in cat_col_names:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return model.predict_proba(X)[:, 1]

def evaluate(proba, y_true, threshold):
    roc = roc_auc_score(y_true, proba)
    pr  = average_precision_score(y_true, proba)
    y_pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary"
    )
    return {"roc_auc": roc, "pr_auc": pr, "precision": precision,
            "recall": recall, "f1": f1, "threshold": threshold}

def print_metrics(label, metrics):
    print(f"\n[{ACTIVE_MODEL.upper()}] [{ACTIVE_GROUP}] {label}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

def main():
    features, cat_cols = get_active_features()
    print(f"Model: {ACTIVE_MODEL} | Grupa: '{ACTIVE_GROUP}' | Features: {len(features)} | Cat: {cat_cols}\n")

    X_train, y_train, X_val, y_val, X_test, y_test, cat_indices = load_data(features, cat_cols)

    if ACTIVE_MODEL == 'catboost':
        model = train_catboost(X_train, y_train, X_val, y_val, cat_indices)

        val_proba  = predict_catboost(model, X_val,  y_val,  cat_indices)
        test_proba = predict_catboost(model, X_test, y_test, cat_indices)

    elif ACTIVE_MODEL == 'lightgbm':
        model, cat_col_names = train_lightgbm(X_train, y_train, X_val, y_val, cat_indices)

        val_proba  = predict_lightgbm(model, X_val,  cat_col_names)
        test_proba = predict_lightgbm(model, X_test, cat_col_names)

    else:
        raise ValueError(f"Nepoznat model: {ACTIVE_MODEL}. Koristi 'catboost' ili 'lightgbm'.")

    print("\n── Val threshold ──")
    val_threshold = find_best_threshold(val_proba, y_val)
    val_metrics   = evaluate(val_proba, y_val, val_threshold)
    print_metrics("VALIDATION METRICS", val_metrics)

    print("\n── Test threshold ──")
    test_threshold = find_best_threshold(test_proba, y_test)
    test_metrics   = evaluate(test_proba, y_test, test_threshold)
    print_metrics("TEST METRICS", test_metrics)

    joblib.dump(model, f"{ACTIVE_MODEL}_{ACTIVE_GROUP}.pkl")
    pd.DataFrame([val_metrics, test_metrics], index=["val", "test"]).to_json(
        f"metrics/{ACTIVE_MODEL}_{ACTIVE_GROUP}_metrics.json"
    )

if __name__ == "__main__":
    main()