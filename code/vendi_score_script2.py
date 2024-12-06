

import os
import pandas as pd
import numpy as np

# Import BERTopic and Top2Vec
from bertopic import BERTopic
from top2vec import Top2Vec

# Load the BASIL dataset
basil = pd.read_csv("/mounts/data/proj/molly/media_bias/basil_dataset.csv")

# Preprocess text to create cleaned_sents
cleaned_sents = []
for text in basil['text']:
    sentences = text.split(',')
    sentences = [s.strip().strip("'") for s in sentences]  # Remove whitespace and single quotes
    cleaned_sents.extend(sentences)

# Create a folder for outputs
os.makedirs("output", exist_ok=True)

# Initialize a dictionary to store features
features = {}





# ---- BERTopic Embeddings ----
print("Generating embeddings using BERTopic...")

# Initialize BERTopic with an embedding model
bertopic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")

# Fit BERTopic and extract embeddings
topics, bertopic_embeddings = bertopic_model.fit_transform(cleaned_sents)

# Check embeddings
if bertopic_embeddings.size == 0:
    raise ValueError("No BERTopic embeddings were generated. Check your input data.")

# Ensure embeddings are 2D
if bertopic_embeddings.ndim == 1:
    print("BERTopic embeddings are 1D; reshaping.")
    bertopic_embeddings = bertopic_embeddings.reshape(1, -1)

# Save embeddings
print(f"Shape of BERTopic embeddings: {bertopic_embeddings.shape}")
np.savetxt("output/bertopic_embeddings.txt", bertopic_embeddings)

# ---- Top2Vec Embeddings ----
print("Generating embeddings using Top2Vec...")

# Initialize and fit Top2Vec
top2vec_model = Top2Vec(documents=cleaned_sents, embedding_model="universal-sentence-encoder")  # Choose your embedding model

# Extract document vectors
top2vec_embeddings = top2vec_model.document_vectors

# Check embedding shape
print(f"Shape of Top2Vec embeddings: {top2vec_embeddings.shape}")

# Save Top2Vec embeddings
features["Top2Vec"] = top2vec_embeddings
np.savetxt("output/top2vec_embeddings.txt", top2vec_embeddings)

print("BERTopic and Top2Vec embeddings saved successfully.")



