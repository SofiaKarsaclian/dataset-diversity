import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer 
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch

class TopicModeling:
    def __init__(self, output_dir, embedding_model_name='all-MiniLM-L6-v2'):  # Check with all-mpnet-base-v2?
        print("Initializing TopicModeling class...")
        # Initialize the class with output paths and models
        self.output_dir = output_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Quantize model for faster processing without CUDA
        transformer_model = self.embedding_model[0]

        quantized_transformer = torch.quantization.quantize_dynamic(
            transformer_model,
            {torch.nn.Linear},  # Quantize only linear layers
            dtype=torch.qint8
        )
        self.embedding_model[0] = quantized_transformer

        self.topic_embeddings = {}

        # Initialize model
        print("Initializing model...")
        self.umap_model = UMAP(
            n_neighbors=8,
            n_components=5,
            min_dist=0.05,
            random_state=42
        )
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=25, #25
            min_samples=20, #20
            gen_min_span_tree=True,
            prediction_data=True,  # For outlier reduction
            cluster_selection_epsilon=0.1
        )

        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            top_n_words=45,
            language='english',
            calculate_probabilities=True,
            verbose=True
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_and_enrich_topics(self, df, threshold=0.05):
        """
        Perform topic modeling and add enriched columns to the DataFrame.
        """
        texts = df['text'].tolist()

        # Fit the model
        topics, probs = self.topic_model.fit_transform(
            tqdm(texts, desc="Fitting BERTopic model", total=len(texts))
        )

        # Reduce outliers
        new_topics = self.topic_model.reduce_outliers(
            texts, topics=topics, probabilities=probs, threshold=threshold, strategy="probabilities"
        )

        # Save topics and representations to the DataFrame
        df['topic'] = new_topics
        df['topic'] = df['topic'].map(self.topic_model.topic_labels_)
        df['topic'] = df['topic'].apply(lambda x: np.nan if '-1' in str(x) else x)

        # Add sentence embeddings for further analysis
        sentence_embeddings = self.embedding_model.encode(df['text'].tolist())
        df['sentence_embedding'] = list(sentence_embeddings)

        return df

    def visualize_topics_and_save(self, name):
        """
        Visualize the topics and save the visualization as an HTML file.
        """
        fig = self.topic_model.visualize_topics()
        fig.write_html(f"data/visualizations/{name}_topics.html")

    def save_topic_embeddings(self):
        """
        Save the topic embeddings to a pickle file.
        """
        with open(f"{self.output_dir}/topic_embeddings.pkl", "wb") as f:
            pickle.dump(self.topic_embeddings, f)

    def save_processed_df(self, df, name):
        """
        Save the processed DataFrame to the specified output path.
        """
        df.to_parquet(f"{self.output_dir}/{name}_topics.parquet", index=False)



    def run(self, df, name, threshold=0.05):
        """
        Process df, perform topic modeling, and save results.
        """
        try:
            # Preprocess and enrich the dataset
            df = self.preprocess_and_enrich_topics(df, threshold)

            # Save the processed DataFrame
            self.save_processed_df(df, name)

            # Visualize and save topics
            self.visualize_topics_and_save(name)

            # Store topic embeddings
            topic_labels = {int(k): v for k, v in self.topic_model.get_topic_info().set_index("Topic").to_dict()["Name"].items()}

            for topic_id_str in set(df['topic']):
                if pd.isna(topic_id_str):
                    continue

                # Find integer topic ID corresponding to this string label
                topic_id_int = next((k for k, v in topic_labels.items() if v == topic_id_str), None)
                if topic_id_int is not None:
                    self.topic_embeddings[f"{name}_topic_{topic_id_str}"] = self.topic_model.topic_embeddings_[topic_id_int]

            # Save all topic embeddings
            self.save_topic_embeddings()

        except Exception as e:
            print(f"Error in topic modeling for {name}: {e}")
        return df

        print("Topic modeling complete")


