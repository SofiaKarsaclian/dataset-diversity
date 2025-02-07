
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.topic_modeling import TopicModeling

datasets_file_paths = {
    #"annomatic": "data/processed/annomatic.csv",
    "babe": "data/processed/babe.csv",
    #"basil": "data/processed/basil.csv"
}
output_dir = "data/enriched/"
os.makedirs(output_dir, exist_ok=True)

# Initialize tools
#entity_extractor = EntityExtractor()
topic_modeling = TopicModeling(output_dir)

def enrich_dataset(name, path):
    """
    Process a single dataset with NER and Topic Modeling, merging results on `text`.
    """
    # Load dataset
    print(f"Processing dataset: {name}")
    df = pd.read_csv(path)
    
    # Extract Entities
    # print(f"Extracting entities for {name}...")
    # processed_data = entity_extractor.process_data(df)

    # # Cluster the entities
    # df_entities = entity_extractor.cluster_entities(processed_data)
    
    # Topic Modeling
    print(f"Performing topic modeling for {name}...")
    
    # Process dataset with TopicModeling
    df_topics = topic_modeling.run(df, name)
    return df_topics
    # Merge Results on `text`
    # print(f"Merging results for {name}...")
    # df_enriched = pd.merge(df_entities, df_topics, on="text", how="inner")
    
    # # # Save Enriched Dataset
    # enriched_path = os.path.join(output_dir, f"{name}_full.parquet")
    # df_enriched.to_parquet(enriched_path, index=False)
    # print(f"Enriched dataset saved to: {enriched_path}")
    # return df_enriched

# Process all datasets
if __name__ == "__main__":
    for name, path in datasets_file_paths.items():
        enrich_dataset(name, path)    
