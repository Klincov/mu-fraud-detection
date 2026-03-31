# evaluate_test.py
import pandas as pd
import joblib
from catboost import Pool
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

TARGET_COL = "is_fraud"
CAT_COLS = ["gender", "state", "job", "category"]

def load_test():
    test = pd.read_parquet("test.parquet")

    # label encoding kao kod treniranja LightGBM-a
    for col in CAT_COLS:
        if col in test.columns:
            test[col] = test[col].astype("category").cat.codes.astype("int32")

    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    # CatBoost i dalje treba originalne kategorije → za njega ćemo opet napraviti poseban DataFrame
    test_cb = pd.read_parquet("test.parquet")
    for col in CAT_COLS:
        if col in test_cb.columns:
            test_cb[col] = test_cb[col].astype("str")
    X_test_cb = test_cb.drop(columns=[TARGET_COL])

    cat_features = [X_test_cb.columns.get_loc(c) for c in CAT_COLS if c in X_test_cb.columns]

    return X_test, y_test, X_test_cb, cat_features

def eval_binary(y_true, proba, threshold=0.5):
    roc = roc_auc_score(y_true, proba)
    pr  = average_precision_score(y_true, proba)
    y_pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary"
    )
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }

def main():
    X_test_lgb, y_test, X_test_cb, cat_features = load_test()

    lgb_model = joblib.load("lightgbm_model.pkl")
    cb_model  = joblib.load("catboost_model.pkl")

    # LightGBM – koristi enkodirani X_test_lgb
    lgb_proba = lgb_model.predict_proba(X_test_lgb)[:, 1]

    # CatBoost – koristi X_test_cb i cat_features
    cb_pool  = Pool(X_test_cb, cat_features=cat_features)
    cb_proba = cb_model.predict_proba(cb_pool)[:, 1]

    lgb_metrics = eval_binary(y_test, lgb_proba, threshold=0.5)
    cb_metrics  = eval_binary(y_test, cb_proba, threshold=0.5)

    print("=== TEST METRICS ===")
    print("LightGBM:")
    for k, v in lgb_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nCatBoost:")
    for k, v in cb_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    df_res = pd.DataFrame([lgb_metrics, cb_metrics], index=["lightgbm", "catboost"])
    df_res.to_csv("test_metrics_lightgbm_catboost.csv")

if __name__ == "__main__":
    main()