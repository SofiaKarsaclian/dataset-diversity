import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.subset_utils import SubsetGenerator
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from bertopic import BERTopic


datasets = {
    "annomatic": pd.read_parquet('data/enriched/annomatic_topics.parquet'),
    "babe": pd.read_parquet('data/enriched/babe_topics.parquet'),
    #"basil": pd.read_parquet('data/enriched/basil_topics.parquet')
}

models = {
    "annomatic": BERTopic.load("data/enriched/models/annomatic_model"),
    "babe": BERTopic.load("data/enriched/models/babe_model")
}

# Add c-TF-IDF
def add_c_tf_idf(dataset, topic_model):
    dataset["topic_nr"] = topic_model.topics_
    c_tf_idf = topic_model.c_tf_idf_

    # Determine the maximum number of components for PCA
    n_components = min(c_tf_idf.shape[0], c_tf_idf.shape[1])
    pca = PCA(n_components=n_components)

    c_tf_idf_pca = pca.fit_transform(c_tf_idf.toarray())
    topic_info = topic_model.get_topic_info()

    c_tf_idf_df = pd.DataFrame({
        'topic_nr': topic_info['Topic'],
        'C-TF-IDF': list(c_tf_idf_pca)
    })

    df = dataset.merge(c_tf_idf_df, on='topic_nr', how='left')
    return df

for key in datasets.keys():
    datasets[key] = add_c_tf_idf(datasets[key], models[key])

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
    print(f"Generating topic-based subsets for {dataset_name}")


    topic_subsamples = subset_generator.create_subsets(
        dimension="topic", 
        rep=rep, 
        subset_size=subset_size, 
        alpha_values=alpha_values, 
        score_function=subset_generator.vendi_score_topic
    )
    topic_scores = subset_generator.process_vendi_scores(topic_subsamples)
    subset_generator.save_subsamples(topic_subsamples, f"{dataset_name}_topic_subsamples")
    topic_scores.to_csv(f"data/subsamples/div_{dataset_name}_topic.csv", index=False)

    print(f"  Finished processing {dataset_name}\n")

