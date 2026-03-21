import pandas as pd

df = pd.read_csv("data.csv")

df.to_parquet("data.parquet",
    engine="pyarrow",
    compression="snappy",
    index=False)
