import re
import pandas as pd
import os

# Severalsources not in adfontes or allsides
# rare guns the-sun rte telegraph recode dallas-news digital-journal deccan-herald tasnim-news-agency-(press-release) the-national-memo-(blog) valuewalk

source_mapping = {
    'nbcnews': 'nbc-news',
    'time': 'time-magazine',
    'sputnik-international': 'sputnik-international-news',
    'new-york-times': 'the-new-york-times',
    'the-hill': 'hill-reporter',
    'breitbart-news': 'breitbart',
    'american-thinker-(blog)': 'american-thinker',
    'dallas-news': 'dallas-morning-news',
    'bbc-news': 'bbc'
}

folder_path = "../data/raw"

df = pd.read_csv(os.path.join(folder_path,"Sora_LREC2020_biasedsentences.csv"))

df['source'] = df['source'].apply(lambda x: '-'.join(re.sub(r'_\d+', '', x.split('_', 1)[1]).lower().split()))
df['source'] = df['source'].replace(source_mapping)

ad_fontes = pd.read_csv("../data/adfontes_clean.csv")
df = df.merge(ad_fontes, on= 'source', how= 'left')

row_list = []
for _, df in df.groupby("id_article"):
    # Extract values that are the same across the group (first row of the group)
    source = df['source'].iloc[0]
    reliability = df['reliability'].iloc[0]
    bias = df['bias'].iloc[0]
    source_bias = df['source_bias'].iloc[0]
    date = df['date_event'].iloc[0]

    # Add title entry
    row_list.append({
        "text": df["doctitle"].iloc[0],
        "label": df["t"].mean(),
        "source": source,
        "reliability": reliability,
        "bias": bias,
        "source_bias": source_bias,
        "date": date
    })

    # Add sentence entries
    for sent in range(20):
        # Check if sentence exists in the article (it might be missing)
        if not df["s" + str(sent)].any():
            continue

        sentence = df["s" + str(sent)].iloc[0]  # sentences are the same within the group, take the first one
        label = df[str(sent)].mean()  # we take the mean of annotations for the sentence

        # Append sentence information with additional columns
        row_list.append({
            "text": sentence,
            "source": source,
            "label": label,
            "reliability": reliability,
            "bias": bias,
            "source_bias": source_bias,
            "date": date
        })

data = pd.DataFrame(row_list)

data["text"] = data["text"].apply(lambda x: re.sub(r"(\[[0-9]*\]:\ )", "", x))
data["label"] = (data["label"] - data["label"].min()) / (data["label"].max() - data["label"].min())

data.to_csv("../data/processed/starbucks.csv", index=False)
print("Processed and saved dataset")
