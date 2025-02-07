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
from torch import bfloat16
import transformers

from torch import cuda




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
            n_components=2,
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

        model_id = 'meta-llama/Llama-3.1-8B'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        print(device)

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )

                # Llama 2 Tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        # Llama 2 Model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map='auto',
        )
        model.eval()

        # Our text generator
        generator = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1
        )

        # System prompt describes information given to all conversations
        system_prompt = """
            <s>[INST] <<SYS>>
            You are a helpful, respectful and honest assistant for labeling topics.
            <</SYS>>
            """
        
        # Example prompt demonstrating the output we are looking for
        example_prompt = """
            I have a topic that contains the following documents:
            - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
            - Meat, but especially beef, is the word food in terms of emissions.
            - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

            The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

            Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

            [/INST] Environmental impacts of eating meat
            """
        
        # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
        main_prompt = """
            [INST]
            I have a topic that contains the following documents:
            [DOCUMENTS]

            The topic is described by the following keywords: '[KEYWORDS]'.

            Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
            [/INST]
            """

        prompt = system_prompt + example_prompt + main_prompt
        llama2 = TextGeneration(generator, prompt=prompt)

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            top_n_words=45,
            language='english',
            calculate_probabilities=True,
            representation_model=llama2,
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
