import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.subset_utils import SubsetGenerator
from sklearn.model_selection import train_test_split


datasets = {
    #"annomatic": pd.read_parquet('data/enriched/annomatic_topics.parquet'),
    "babe": pd.read_parquet('data/enriched/babe_topics.parquet'),
    #"basil": pd.read_parquet('data/enriched/basil_topics.parquet')
}

# keep only majority vote 
columns_to_remove = ["entities_gliner", "entities_spaCy", "entities_flair"]
for name, df in datasets.items():
    datasets[name] = df.drop(columns=columns_to_remove, errors='ignore')

# Split train-test 
def split_train_test(df, test_size=500):    
    df['stratify_col'] = df['label'].astype(str) #+ '_' + df['source'] + '_' + df['topic']
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['stratify_col'], random_state= 42)
    return train_df, test_df

babe_train, babe_test = split_train_test(datasets['babe'])
#basil_train, basil_test = split_train_test(datasets['basil'])

datasets['babe'] = babe_train 
#datasets['basil'] = basil_train

output_dir = "data/processed/test_sets"
os.makedirs(output_dir, exist_ok=True)

# Save the parquet files
babe_test.to_parquet(os.path.join(output_dir, "babe_test.parquet"))
#basil_test.to_parquet(os.path.join(output_dir, "basil_test.parquet"))

print("Generated split and saved test sets...")

# Define parameters: 
# 32 subsamples for each dataset-dimension
alpha_values = [0.001, 0.005, 0.01, 0.05, 0.08, 0.1, 0.5, 1, 10, 100, 1000]
rep = 3
subset_size = 500

for dataset_name, df in datasets.items():
    print(f"Processing dataset: {dataset_name}")

    subset_generator = SubsetGenerator(df)

    # Source subsets
    print(f"  Generating source-based subsets for {dataset_name}")
    source_subsamples = subset_generator.create_subsets(
        dimension="source", 
        rep=rep, 
        subset_size=subset_size, 
        alpha_values=alpha_values, 
        score_function=subset_generator.vendi_score_source
    )
    source_scores = subset_generator.process_vendi_scores(source_subsamples)
    subset_generator.save_subsamples(source_subsamples, f"{dataset_name}_source_subsamples")
    source_scores.to_csv(f"data/subsamples/div_{dataset_name}_source.csv", index=False)

    # Topic subsets
    print(f"  Generating topic-based subsets for {dataset_name}")

    # import topic embeddings
    file_path= "data/enriched/topic_embeddings.pkl"
    topic_embeddings = pd.read_pickle(file_path)


    embeddings = []
    for key, embedding in topic_embeddings.items():
        parts = key.split('_')
        dataset_name = parts[0]  
        topic = '_'.join(parts[2:]) 
        embeddings.append({
            'dataset_name': dataset_name,
            'topic': topic,
            'embedding': embedding
        })

    for dataset_name, df in datasets.items():
        dataset_embeddings = [e for e in embeddings if e['dataset_name'] == dataset_name]
        topic_to_embedding = {e['topic']: e['embedding'] for e in dataset_embeddings}
        df['topic_embedding'] = df['topic'].map(topic_to_embedding)

    topic_subsamples = subset_generator.create_subsets(
        dimension="topic", 
        rep=rep, 
        subset_size=subset_size, 
        alpha_values=alpha_values, 
        score_function=subset_generator.vendi_score_topic, 
        embedding_column="topic_embedding" # cos similarity between topics embeddings
    )
    topic_scores = subset_generator.process_vendi_scores(topic_subsamples)
    subset_generator.save_subsamples(topic_subsamples, f"{dataset_name}_topic_subsamples")
    topic_scores.to_csv(f"data/subsamples/div_{dataset_name}_topic.csv", index=False)

    print(f"  Finished processing {dataset_name}\n")

