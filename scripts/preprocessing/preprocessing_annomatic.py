import pandas as pd

source_mapping = {
    "the-advocate": "the-advocate-–-baton-rouge",
    "atlanta-journalconstitution": "atlanta-journal-constitution",
    "chicago-suntimes": "chicago-sun-times",
    "san-diego-uniontribune": "san-diego-union-tribune",
    "pittsburgh-postgazette": "pittsburgh-post-gazette",
    "nj": "nj.com",
    "insider": "business-insider"
}

annomatic = pd.read_parquet("data/raw/anno-lexical-train.parquet")
annomatic["source_name"] = annomatic["source_name"].replace(source_mapping)

ad_fontes = pd.read_csv('data/adfontes_clean.csv')

annomatic = annomatic.merge(ad_fontes, left_on='source_name', right_on='source', how='left')
annomatic = annomatic.drop(columns=['source_name', 'source_party', 'sentence_id'])


output_path = ("data/processed/annomatic.csv")
annomatic.to_csv(output_path, index=False)

print("Processed and saved dataset")
