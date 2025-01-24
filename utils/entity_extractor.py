import pandas as pd
import torch
from tqdm import tqdm
import spacy
import flair
from flair.models import SequenceTagger
from gliner import GLiNER
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from sklearn.decomposition import TruncatedSVD
import inflect
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


class EntityExtractor:
    def __init__(self):
        # Load pre-trained models
        self.gliner_model = GLiNER.from_pretrained("EmergentMethods/gliner_large_news-v2.1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gliner_model.to(self.device)
        self.gliner_model.eval()

        self.spacy_nlp = spacy.load("en_core_web_lg")
        self.flair_tagger = SequenceTagger.load("flair/ner-english-fast")

        self.p = inflect.engine()  

        self.labels = [
            "person", "location", "geo-political entity", "work of art", "product",
            "group", "religion", "nationality", "organization", "city", "brand",
            "country", "event", "law", "facility", "language", "title", "media"
        ]

        # Entity disambiguation
        self.exceptions = {
            "hamas", "united states", "ron desantis", "united arab emirates", "massachusetts",
            "mcdonalds", "congress", "texas", "ad fontes", "mercedes", "st. louis", "guterres"
        }

        self.entity_resolution_dict = {
            "United States": ["U.S.", "US", "U S", "America", "USA"],
            "Donald Trump": ["Trump", "President Trump", "Mr. Trump", "Mr Trump", "Donald J. Trump"],
            "Joe Biden": ["Biden", "President Biden", "Mr. Biden", "BIDEN", "President Joe"],
            "Ron DeSantis": ["DeSantis"],
            "Kamala Harris": ["Harris"],
            "George Floyd": ["Floyd"],
            "Barack Obama": ["Obama"],
            "Nancy Pelosi": ["Pelosi"],
            "Republican Party": ["GOP", "Republican Parties"],
            "United Kingdom": ["UK", "United Kingdom", "U.K."],
            "Jesus Christ": ["Christ", "Jesus"],
            "Washington": ["D.C.", "Washington D.C."],
            "COVID-19": ["COVID", "Covid-19"],
            "Gaza": ["Gaza Strip"],
            "Vladimir Putin": ["Putin", "Vladimir Putin", "Mr. Putin"],
            "Benjamin Netanyahu": ["Netanyahu"],
            "television": ["TV"],
            "President": ["president"],
            "European Union": ["EU"],
            "Elon Musk": ["Musk", "Mr. Musk", "Elon"],
            "Rishi Sunak": ["Sunak"],
            "Nikki Haley": ["Nikki", "Haley"],
            "Henry Kissinger": ["Kissinger"],
            "United Arab Emirates": ["UAE"],
            "Xi Jinping": ["Xi"],
        }

        self.reverse_lookup = {
            variant: canonical
            for canonical, variants in self.entity_resolution_dict.items()
            for variant in variants
        }


    def extract_entities_gliner(self, data, text_col, batch_size=10):
        all_text = data[text_col].to_list()
        all_predictions = []

        with tqdm(total=len(all_text), desc="Processing GLiNER Batches", unit="docs") as pbar:
            for batch in self.create_batches(all_text, batch_size):
                with torch.no_grad():
                    predictions = self.gliner_model.batch_predict_entities(batch, self.labels)
                all_predictions.extend(predictions)
                pbar.update(len(batch))

        extracted_entities = [
            [entity["text"] for entity in prediction]
            for prediction in all_predictions
        ]
        data["entities_gliner"] = extracted_entities
        return data

    def extract_entities_spacy(self, data, text_col):
        entities = []
        with tqdm(total=len(data), desc="Processing spaCy Entities", unit="docs") as pbar:
            for doc in self.spacy_nlp.pipe(data[text_col], disable=["parser", "tagger"]):
                extracted_entities = []
                for ent in doc.ents:
                    if ent.label_ not in ["QUANTITY", "DATE", "CARDINAL", "PERCENT", "TIME"]:
                        entity_text = ent.text.lstrip("the ") # Sometimes adds this
                        extracted_entities.append(entity_text)
                entities.append(extracted_entities)
                pbar.update(1)
        data["entities_spaCy"] = entities
        return data

    def extract_entities_flair(self, data, text_col):
        entities = []
        with tqdm(total=len(data), desc="Processing Flair Entities", unit="docs") as pbar:
            for text in data[text_col]:
                sentence = flair.data.Sentence(text)
                self.flair_tagger.predict(sentence)
                extracted_entities = [entity.text for entity in sentence.get_spans('ner')]
                entities.append(extracted_entities)
                pbar.update(1)
        data["entities_flair"] = entities
        return data

    def majority_vote(self, entity_lists):
        entity_counts = {}
        for entity_list in entity_lists:
            for entity in set(entity_list):
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        return [entity for entity, count in entity_counts.items() if count >= 2]

    def process_data(self, df):
        df_ner = df.copy()
        df_ner = self.extract_entities_gliner(df_ner, "text", batch_size=10)
        df_ner = self.extract_entities_spacy(df_ner, "text")
        df_ner = self.extract_entities_flair(df_ner, "text")

        df_ner["entities"] = df_ner.apply(
            lambda row: self.majority_vote([row['entities_gliner'], row['entities_spaCy'], row['entities_flair']]),
            axis=1
        )

        return df_ner

    def create_batches(self, data, batch_size=10):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def clean_entities(self, entity_list):

        unique_entities = []
        seen_entities = set()

        for entity in entity_list:
            entity_text = entity.lower() if isinstance(entity, str) else entity.get("text", "").lower()
            if entity_text not in seen_entities:
                unique_entities.append(entity)
                seen_entities.add(entity_text)

        return unique_entities

    def resolve_entity(self, entity, exceptions=None):
        exceptions = exceptions or set()
        if isinstance(entity, str):
            if entity.lower() in exceptions:
                return entity

            singular_form = self.p.singular_noun(entity)
            return singular_form if singular_form else entity
        return entity

    def resolve_disambiguation(self, entities):
        """
        Resolve ambiguities and apply entity resolution where applicable.
        """
        resolved_entities = []
        for entity in entities:
            canonical = self.reverse_lookup.get(entity, entity)
            resolved_entities.append(canonical)
        return resolved_entities

    def process_entities(self, entity_list, exceptions=None):
        """
        Clean and resolve entities with exception handling and disambiguation.
        """
        processed_entities = []
        exceptions = exceptions or self.exceptions

        for entity in entity_list:
            try:
                resolved_entity = self.resolve_entity(entity, exceptions)
                resolved_entity = self.reverse_lookup.get(resolved_entity, resolved_entity)
                processed_entities.append(resolved_entity)
            except Exception as e:
                print(f"Error processing entity '{entity}': {e}")
                processed_entities.append(entity)

        return processed_entities

    def cluster_entities(self, df, min_cluster_size=30, min_samples=10, n_components=50):
        np.random.seed(42)
        
        df['has_entities'] = df['entities'].apply(lambda x: len(x) > 0)
        df_with_entities = df[df['has_entities']].copy()
        df_without_entities = df[~df['has_entities']].copy()

        df_with_entities['entities_str'] = df_with_entities['entities'].apply(lambda x: ' '.join(x))
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
        X_sparse = vectorizer.fit_transform(df_with_entities['entities_str'])

        if n_components and n_components < X_sparse.shape[1]:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = svd.fit_transform(X_sparse)
            df_with_entities['X_reduced'] = list(X_reduced) 
        else:
            X_reduced = X_sparse
            df_with_entities['X_reduced'] = None

        cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                                min_samples=min_samples, 
                                metric='euclidean', 
                                approx_min_span_tree=True)
        
        df_with_entities['entity_cluster'] = cluster_model.fit_predict(X_reduced)

        df_without_entities['entity_cluster'] = None
        df_with_entities['entity_cluster'] = df_with_entities['entity_cluster'].replace(-1, None)

        df_final = pd.concat([df_with_entities, df_without_entities], ignore_index=True)
        df_final.drop(columns=['has_entities', 'entities_str'], inplace=True)

        return df_final

    def create_wordcloud(self, df, output_path):
        # Create a list of entities, avoiding NaN values
        all_entities = [str(entity) for entity in df["entities"].explode() if not pd.isna(entity)]
        text = ' '.join(all_entities)

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path, format='png')
        plt.close()


