import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import joblib

TARGET_COL = "is_fraud"
CAT_COLS = ["gender", "state", "job", "category"]  # možeš dodati još ako treba

def load_data():
    train = pd.read_parquet("train.parquet")
    val   = pd.read_parquet("val.parquet")

    # label encoding za string kolone
    for col in CAT_COLS:
        if col in train.columns:
            # spoji vrednosti iz train i val da bi kodovi bili konzistentni
            all_vals = pd.concat([train[col], val[col]], axis=0).astype("category")
            train[col] = all_vals.iloc[:len(train)].cat.codes.astype("int32")
            val[col]   = all_vals.iloc[len(train):].cat.codes.astype("int32")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]

    X_val = val.drop(columns=[TARGET_COL])
    y_val = val[TARGET_COL]

    return X_train, y_train, X_val, y_val

def train_lightgbm(X_train, y_train, X_val, y_val):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos

    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,   # fiksan broj stabala
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "average_precision"],
    )
    return model

def evaluate(model, X_val, y_val, threshold=0.5):
    proba = model.predict_proba(X_val)[:, 1]
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
    X_train, y_train, X_val, y_val = load_data()
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    metrics = evaluate(model, X_val, y_val)
    print("LightGBM VALIDATION METRICS:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    joblib.dump(model, "lightgbm_model.pkl")
    pd.Series(metrics).to_json("lightgbm_val_metrics.json")

if __name__ == "__main__":
    main()