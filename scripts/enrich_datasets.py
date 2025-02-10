import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.topic_modeling import TopicModeling

datasets_file_paths = {
    "annomatic": "data/processed/annomatic.csv",
    "babe": "data/processed/babe.csv",
    #"basil": "data/processed/basil.csv"
}
output_dir = "data/enriched/"
os.makedirs(output_dir, exist_ok=True)

def clean_labels(topic_model, df):
    llama2_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["llama2"].values()]
    topic_model.set_topic_labels(llama2_labels)

    # assign new labels to df
    topic_info = topic_model.get_topic_info()
    topic_mapping = {
        row['Name'][1:] if row['Name'].startswith('-1') else row['Name']: row['CustomName']
        for index, row in topic_info.iterrows()
    }
    df['topic'] = df['topic'].map(topic_mapping)

    return df

def enrich_dataset(name, path):
    """
    Process a single dataset with NER and Topic Modeling, merging results on `text`.
    """
    # Initialize tool
    topic_modeling = TopicModeling(output_dir)

    # Load dataset
    print(f"Processing dataset: {name}")
    df = pd.read_csv(path)

    # Topic Modeling
    print(f"Performing topic modeling for {name}...")

    # Preprocess and enrich topics
    texts = df['text'].tolist()
    topics, probs = topic_modeling.topic_model.fit_transform(texts)
    new_topics = topic_modeling.topic_model.reduce_outliers(texts, topics=topics, probabilities=probs, threshold=0.05, strategy="probabilities")
    topic_modeling.topic_model.update_topics(texts)
    df['topic'] = new_topics
    df['topic'] = df['topic'].map(topic_modeling.topic_model.topic_labels_)
    df['topic'] = df['topic'].apply(lambda x: np.nan if '-1' in str(x) else x)

    sentence_embeddings = topic_modeling.topic_model.embedding_model.encode(df['text'].tolist())
    df['sentence_embedding'] = list(sentence_embeddings)

    # Clean labels
    df = clean_labels(topic_modeling.topic_model, df)


    # Visualize topics and save
    fig = topic_modeling.topic_model.visualize_topics(use_ctfidf=True, title=f'<b>{name} Topics</b>')
    fig.write_html(f"data/visualizations/{name}_topics.html")
    
    emb = np.vstack(df["sentence_embedding"].to_numpy())
    
    fig_doc = topic_modeling.topic_model.visualize_topics(texts, embeddings=emb, hide_annotations=True, hide_document_hover=False, custom_labels=True,
                                                          title=f'<b>{name} Documents and Topics</b>')
    fig_doc.write_html(f"data/visualizations/{name}_docs.html")

    df = df.drop(columns=['sentence_embedding'])

    # Save processed DataFrame
    df.to_parquet(f"{output_dir}/{name}_topics.parquet", index=False)

    # Save the model
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_dir = os.path.join(models_dir, f"{name}_model")
    topic_modeling.topic_model.save(model_dir, serialization="pytorch", save_ctfidf=True, save_embedding_model="sentence-transformers/all-mpnet-base-v2")
    print(f"BERTopic model saved at: {models_dir}")

    return df

# Process all datasets
if __name__ == "__main__":
    for name, path in datasets_file_paths.items():
        enrich_dataset(name, path)