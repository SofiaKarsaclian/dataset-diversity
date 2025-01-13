import os
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
folder_path = os.path.join(project_root, 'data', 'raw', 'anno-lexical-train.parquet')

source_mapping = {
    "the-advocate": "the-advocate-–-baton-rouge",
    "atlanta-journalconstitution": "atlanta-journal-constitution",
    "chicago-suntimes": "chicago-sun-times",
    "san-diego-uniontribune": "san-diego-union-tribune",
    "pittsburgh-postgazette": "pittsburgh-post-gazette",
    "nj": "nj.com",
    "insider": "business-insider"
}

annomatic = pd.read_parquet(folder_path)
annomatic["source_name"] = annomatic["source_name"].replace(source_mapping)

adfontes_path = os.path.join(project_root, 'data', 'adfontes_clean.csv')
ad_fontes = pd.read_csv(adfontes_path)

annomatic = annomatic.merge(ad_fontes, left_on='source_name', right_on='source', how='left')
annomatic = annomatic.drop(columns=['source_name', 'source_party', 'sentence_id'])

output_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "annomatic.csv")
annomatic.to_csv(output_path, index=False)

print("Processed and saved dataset")
