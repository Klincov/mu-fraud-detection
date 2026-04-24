import pandas as pd
import joblib
from catboost import Pool
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt


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
    'customer+merchant_risk': baseline + customer_behavior + merchant_risk,
    'all_bez_geo':           baseline + demographics +
                             temporal + customer_behavior + merchant_profile + merchant_risk,
    'all_bez_customer':      baseline + demographics + geography +
                             temporal + merchant_profile + merchant_risk,
}


TARGET_COL = "is_fraud"
ALL_CAT_COLS = ["gender", "state", "job"]


def get_active_features():
    features = feature_groups[ACTIVE_GROUP]
    cat_features = [c for c in ALL_CAT_COLS if c in features]
    return features, cat_features

def load_test(features, cat_cols):
    cols_to_load = features + [TARGET_COL]
    test = pd.read_parquet("test.parquet", columns=cols_to_load)

    # CatBoost
    X_cb = test.copy()
    for col in cat_cols:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype("str")
    X_cb = X_cb.drop(columns=[TARGET_COL])

    # LightGBM
    X_lgb = test.copy()
    for col in cat_cols:
        if col in X_lgb.columns:
            X_lgb[col] = X_lgb[col].astype("category")
    X_lgb = X_lgb.drop(columns=[TARGET_COL])

    y = test[TARGET_COL]

    cat_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

    return X_lgb, X_cb, y, cat_indices


def find_best_threshold(proba, y_true):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx = f1_scores.argmax()
    best_threshold = thresholds[max(best_idx - 1, 0)]  # jer thresholds ima len-1

    return best_threshold


def eval_binary(y_true, proba, threshold=None):
    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)

    if threshold is None:
        threshold = find_best_threshold(proba, y_true)

    y_pred = (proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary"
    )
    #Confusion matrixx
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # 'true' normalizuje po pravim klasama
    disp = ConfusionMatrixDisplay(cm, display_labels=['Validna', 'Prevara'])
    disp.plot(values_format='.1%')  
    plt.show()

    #PR kriva i ROC kriva
    PrecisionRecallDisplay.from_predictions(y_true,proba).plot()
    plt.show()

    RocCurveDisplay.from_predictions(y_true, proba).plot()
    plt.show()

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


def evaluate_model(name, model, X_lgb, X_cb, y, cat_features):
    try:
        if "catboost" in name.lower():
            pool = Pool(X_cb, cat_features=cat_features)
            proba = model.predict_proba(pool)[:, 1]
            # CatBoost
            feat_imp = pd.Series(model.get_feature_importance(), 
                                index=X_cb.columns).sort_values(ascending=False)
            feat_imp.plot(kind='bar')
            plt.show()
        else:
            proba = model.predict_proba(X_lgb)[:, 1]

        metrics = eval_binary(y, proba)
        return metrics

    except Exception as e:
        print(f"Greška kod modela {name}: {e}")
        return None


ACTIVE_GROUP = 'all_bez_merchant_risk'

def main():
    models = {
        "catboost": "models/catboost_all_bez_merchant_risk.pkl",
        "lightgbm": "models/catboost_all_bez_merchant_risk.pkl",
    }

    features, cat_cols = get_active_features()
    X_test_lgb, X_test_cb, y_test, cat_indices = load_test(features, cat_cols)

    results = []

    for name, path in models.items():
        try:
            model = joblib.load(path)
            metrics = evaluate_model(name, model, X_test_lgb, X_test_cb, y_test, cat_indices)

            if metrics:
                metrics["model"] = name
                results.append(metrics)

        except Exception as e:
            print(f"Ne mogu da učitam model {name}: {e}")

    df_res = pd.DataFrame(results).set_index("model")
    print("\nTEST METRICS:")
    print(df_res)

    df_res.to_csv("test_metrics.csv")


if __name__ == "__main__":
    main()