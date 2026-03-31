import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import joblib

TARGET_COL = "is_fraud"
CAT_COLS = ["gender", "state", "job", "category"]  # proširi ako imate još kategorijskih

def load_data():
    train = pd.read_parquet("train.parquet")
    val   = pd.read_parquet("val.parquet")

    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype("str")
            val[col]   = val[col].astype("str")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    # indeksi kategorijskih kolona u X_train
    cat_features = [X_train.columns.get_loc(c) for c in CAT_COLS if c in X_train.columns]

    return X_train, y_train, X_val, y_val, cat_features

def train_catboost(X_train, y_train, X_val, y_val, cat_features):
    pos = y_train.sum()
    neg = len(y_train) - pos
    class_weights = [1.0, neg / pos]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        learning_rate=0.05,
        depth=6,
        iterations=2000,
        random_seed=42,
        verbose=200,
        class_weights=class_weights,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model

def evaluate(model, X_val, y_val, cat_features, threshold=0.5):
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    proba = model.predict_proba(val_pool)[:, 1]

    roc = roc_auc_score(y_val, proba)
    pr  = average_precision_score(y_val, proba)
    y_pred = (proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, pos_label=1, average="binary"
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
    X_train, y_train, X_val, y_val, cat_features = load_data()
    model = train_catboost(X_train, y_train, X_val, y_val, cat_features)

    metrics = evaluate(model, X_val, y_val, cat_features)
    print("CatBoost VALIDATION METRICS:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    joblib.dump(model, "catboost_model.pkl")
    pd.Series(metrics).to_json("catboost_val_metrics.json")

if __name__ == "__main__":
    main()