import os
import pandas as pd

source_mapping = {
    "federalist": "the-federalist",
    "new-york-times": "the-new-york-times",
}

# the daily stormer is missing

bias_mapping = {"Biased" : 1, "Non-biased": 0}

# get absolute path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
folder_path = os.path.join(project_root, 'data', 'raw', 'babe.csv')

df = pd.read_csv(folder_path, delimiter=";")

# select relevant columns
df = df[["text", "outlet", "label_bias"]].rename(columns={"outlet": "source", "label_bias": "label"})
df["source"] = df["source"].str.lower().str.replace(" ", "-")
df["source"] = df["source"].replace(source_mapping)
df["label"] = df["label"].map(bias_mapping)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype("int64")


print(f"Removing from the daily-stormer: {len(df[df["source"] == "daily-stormer"])} rows")
df = df[df["source"] != "daily-stormer"] # exclude

adfontes_path = os.path.join(project_root, 'data', 'adfontes_clean.csv')
adfontes = pd.read_csv(adfontes_path)

df = df.merge(adfontes, on='source', how='left')

output_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "babe.csv")
df.to_csv(output_path, index=False)

print("Processed and saved dataset")
