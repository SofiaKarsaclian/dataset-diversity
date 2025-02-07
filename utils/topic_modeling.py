import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch
from transformers import AutoTokenizer, pipeline
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess
from bertopic.representation import TextGeneration
from ctransformers import AutoModelForCausalLM





class TopicModeling:
    def __init__(self, output_dir, embedding_model_name='all-mpnet-base-v2'):
        print("Initializing TopicModeling class...")
        self.output_dir = output_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)

        if torch.cuda.is_available():
            print("CUDA is available. Skipping quantization.")
        else:
            print("CUDA is not available. Quantizing the model.")
            transformer_model = self.embedding_model[0]
            quantized_transformer = torch.quantization.quantize_dynamic(
                transformer_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            self.embedding_model[0] = quantized_transformer

        self.topic_embeddings = {}
        self.coherence_scores = {}

        print("Initializing model...")
        self.umap_model = UMAP(
            n_neighbors=8,
            n_components=5,
            min_dist=0.05,
            random_state=42
        )
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=25,
            min_samples=20,
            gen_min_span_tree=True,
            prediction_data=True,
            cluster_selection_epsilon=0.1
        )

        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )

        gpu_available = torch.cuda.is_available()
        gpu_layers = 0  # Default to 0 (no GPU acceleration)
        if gpu_available:          
            gpu_layers = 50

        # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
        model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/zephyr-7B-alpha-GGUF",
            model_file="zephyr-7b-alpha.Q4_K_M.gguf",
            model_type="mistral",
            gpu_layers=gpu_layers,
            hf=True
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")

        # Pipeline
        generator = pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            max_new_tokens=50,
            repetition_penalty=1.1
        )

        prompt = """<|system|>You are a helpful, respectful and honest assistant for labeling topics..</s>
         <|user|>
         I have a topic that contains the following documents:
        [DOCUMENTS]

        The topic is described by the following keywords: '[KEYWORDS]'.

        Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.</s>
        <|assistant|>"""
        zephyr = TextGeneration(generator, prompt=prompt)
        self.representation_model = {"Zephyr": zephyr}

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            top_n_words=45,
            language='english',
            calculate_probabilities=True,
            representation_model=self.representation_model,
            verbose=True
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def save_model(self, model_name):
        models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        if not hasattr(self.topic_model, 'topic_labels_'):
            raise Exception("The BERTopic model is not yet fitted. Please fit the model before saving.")

        model_dir = os.path.join(models_dir, f"{model_name}_model")
        self.topic_model.save(model_dir, serialization="pytorch", save_ctfidf=True, save_embedding_model="sentence-transformers/all-mpnet-base-v2")
        print(f"BERTopic model saved at: {models_dir}")

    def preprocess_and_enrich_topics(self, df, threshold=0.05):
        texts = df['text'].tolist()

        topics, probs = self.topic_model.fit_transform(
            tqdm(texts, desc="Fitting BERTopic model", total=len(texts))
        )

        new_topics = self.topic_model.reduce_outliers(
            texts, topics=topics, probabilities=probs, threshold=threshold, strategy="probabilities"
        )

        self.topic_model.update_topics(texts)
        df['topic'] = new_topics
        df['topic'] = df['topic'].map(self.topic_model.topic_labels_)
        df['topic'] = df['topic'].apply(lambda x: np.nan if '-1' in str(x) else x)

        sentence_embeddings = self.embedding_model.encode(df['text'].tolist())
        df['sentence_embedding'] = list(sentence_embeddings)

        return df

    def compute_coherence_score(self, df):
        print("Computing coherence score...")

        # Get the top words per topic
        topic_words = self.topic_model.get_topics()

        # Format topic words for Gensim CoherenceModel
        topic_list = [[word for word, _ in words] for topic_id, words in topic_words.items() if topic_id != -1]

        # Tokenize text data
        tokenized_texts = [simple_preprocess(doc) for doc in df['text'].dropna().tolist()]

        # Create Gensim Dictionary
        dictionary = Dictionary(tokenized_texts)

        # Create the CoherenceModel
        coherence_model = CoherenceModel(
            topics=topic_list,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'  
        )

        # Compute coherence score
        coherence_score = coherence_model.get_coherence()
        print(f"✅ Coherence Score (c_v): {coherence_score}")
        return coherence_score
    
    def visualize_topics_and_save(self, name):
        """Visualize the topics and save the visualization as an HTML file."""
        fig = self.topic_model.visualize_topics(use_ctfidf=True, title=f'<b>{name} Topics</b>')
        fig.write_html(f"data/visualizations/{name}_topics.html")

    def save_processed_df(self, df, name):
        df.to_parquet(f"{self.output_dir}/{name}_topics.parquet", index=False)

    def run(self, df, name, threshold=0.05):
        try:
            df = self.preprocess_and_enrich_topics(df, threshold)
            self.save_processed_df(df, name)
            self.visualize_topics_and_save(name)
            self.save_model(name)

        except Exception as e:
            print(f"Error in topic modeling for {name}: {e}")

        return df
