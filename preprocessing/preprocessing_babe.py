import os
import pandas as pd

source_mapping = {
    "federalist": "the-federalist",
    "new-york-times": "the-new-york-times",
}
# the-daily-stormer is missing

bias_mapping = {"Biased" : 1, "Non-biased": 0}

df = pd.read_csv("../data/raw/babe.csv", delimiter= ";")

df = df[["text", "outlet", "label_bias"]].rename(columns={"outlet": "source", "label_bias": "label"})

df["source"] = df["source"].str.lower().str.replace(" ", "-")
df["source"] = df["source"].replace(source_mapping)

df["label"] = df["label"].map(bias_mapping)

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype("int64")
#print(len(df[df["source"] == "daily-stormer"]))

df = df[df["source"]!= "daily-stormer"] # filter out since it doesn't have a score (only 1 case)

adfontes = pd.read_csv("../data/adfontes_clean.csv")
df = df.merge(adfontes, on= 'source', how= 'left')

df.to_csv("../data/processed/babe.csv", index=False)
print("Processed and saved dataset")
