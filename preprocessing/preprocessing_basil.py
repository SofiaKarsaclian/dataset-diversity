import pandas as pd
import json
import os

folder_path = "../data/raw/basil"

# List to store all rows across files
all_rows = []

source_mapping = {"fox": "fox-news",
                  "nyt": "the-new-york-times",
                  "hpo": "huffpost"}

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        for entry in data.get("body", []):
            sentence = entry.get("sentence", "")

            if not sentence: #6 empty sentences
                continue
            
            # Positive label = Lexical bias
            # In MAGPIE they take Information label. Could maybe even be multiclass?
            has_lexical_bias = any(annotation.get("bias") == "Lexical" for annotation in entry.get("annotations", []))

            source = data.get("source", "").lower()
            source = source_mapping.get(source, source)
            
            row = {
                "text": sentence,
                #"date": data.get("date", ""),
                "source": source,
                #"article_main_entities": ", ".join(data.get("main-entities", [])),
                "label": int(has_lexical_bias)  
            }

            all_rows.append(row)

df = pd.DataFrame(all_rows)

ad_fontes = pd.read_csv("../data/adfontes_clean.csv")
df = df.merge(ad_fontes, on= 'source', how= 'left')

# 6 missing texts
print(df.loc[df['text'].isna()]) 
df = df.dropna(subset=["text"]).reset_index(drop=True)
print(df.loc[df['text'].isna()]) 

df.to_csv("../data/processed/basil.csv", index=False)
print("Processed and saved dataset")