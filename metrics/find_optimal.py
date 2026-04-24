import json
import pandas as pd
import glob

rows = []

for file in glob.glob("metrics/*.json"):
    with open(file) as f:
        data = json.load(f)

    rows.append({
        "model": file,
        "pr_auc_val": data["pr_auc"]["val"],
        "pr_auc_test": data["pr_auc"]["test"],
        "f1_val": data["f1"]["val"],
        "f1_test": data["f1"]["test"],
        "recall_test": data["recall"]["test"],
        "precision_test": data["precision"]["test"],
        "roc_auc_test": data["roc_auc"]["test"]
    })

df = pd.DataFrame(rows)

# provaj i neku kombinaciju pr auc, recall, i f1 ako stigne
df = df.sort_values(by="pr_auc_test", ascending=False)

print(df)