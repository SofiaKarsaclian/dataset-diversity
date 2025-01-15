import os
import pandas as pd
import json


folder_path = "data/raw/basil"

# List to store all rows across files
all_rows = []

source_mapping = {"fox": "fox-news",
                  "nyt": "the-new-york-times",
                  "hpo": "huffpost"}

if not os.path.exists(folder_path):
    print(f"Error: The folder path {folder_path} does not exist.")
else:
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file_path}")
                continue 
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue  

            for entry in data.get("body", []):
                sentence = entry.get("sentence", "")

                if not sentence:  
                    continue

                # Positive label = Lexical bias (and not information bias)
                has_lexical_bias = any(annotation.get("bias") == "Lexical" for annotation in entry.get("annotations", []))

                source = data.get("source", "").lower()
                source = source_mapping.get(source, source)

                row = {
                    "text": sentence,
                    "source": source,
                    "label": int(has_lexical_bias)
                }

                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Load and merge with AdFontes data for sources
    ad_fontes = pd.read_csv('data/adfontes_clean.csv')
    df = df.merge(ad_fontes, on='source', how='left')

    # Check if there are any missing 'text' values
    print(f"Rows with missing 'text' before drop: {df['text'].isna().sum()}")
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    print(f"Rows with missing 'text' after drop: {df['text'].isna().sum()}")


    # Save the processed dataframe
    df.to_csv("data/processed/basil.csv", index=False)

    print("Processed and saved dataset")
