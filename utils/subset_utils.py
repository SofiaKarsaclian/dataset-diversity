import os
import re
import numpy as np
import pandas as pd
from vendi_score import vendi
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class SubsetGenerator:
    def __init__(self, df):
        """
        Initialize the SubsetGenerator class with a dataframe.
        
        Parameters:
        - df (pd.DataFrame): The input dataframe containing the data.
        """
        self.df = df
        np.random.seed(42)

    def generate_random_distribution(self, dim, alpha, rep):
        """
        Generate Dirichlet-distributed samples.

        Parameters:
        - dim (int): Number of categories or dimensions for the Dirichlet distribution (unique sources/topics).
        - alpha (float): Concentration parameter of the Dirichlet distribution.
        - rep (int): Number of random samples to generate.

        Returns:
        - np.ndarray: A 2D array of shape (rep, dim), where each row is a Dirichlet-distributed sample.
        """
        return np.random.dirichlet([alpha] * dim, rep)

    def vendi_score_source(self, df):
        """
        Calculate Vendi Score for source dimension using Manhattan distance on reliability and bias.

        Parameters:
        - df (pd.DataFrame): DataFrame containing 'reliability' and 'bias' columns.

        Returns:
        - float: Vendi Score for sources.
        """
        features = df[['reliability', 'bias']].to_numpy()
        #distances = np.sqrt(np.sum((features[:, np.newaxis] - features[np.newaxis, :]) ** 2, axis=-1))
        # manhattan
        distances = np.sum(np.abs(features[:, np.newaxis] - features[np.newaxis, :]), axis=-1)
        similarity_matrix = 1 / (1 + distances)
        return vendi.score_K(similarity_matrix)

    def vendi_score_topic(self, df):
        """
        Calculate Vendi Score for topic using cosine similarity between topic embeddings.
        """
        similarity_matrix= cosine_similarity(np.stack(df["C-TF-IDF"]))
        vendi_score = vendi.score_K(similarity_matrix)

        return vendi_score


    def create_subsets(self, dimension, rep, subset_size, alpha_values, score_function):
        """
        Generate subsets of data using Dirichlet distributions for sampling proportions.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing the data.
        - dimension (str): The column name for the dimension ('source' or 'topic').
        - rep (int): The number of repetitions or subsets to generate.
        - subset_size (int): The total number of samples to draw for each subset.
        - alpha_values (list): List of alpha values for the Dirichlet distribution.
        - score_function (function): Function to calculate Vendi Score (e.g., vendi_score_source or vendi_score_topic).

        Returns:
        - dict: Dictionary where keys are subset identifiers and values are dicts containing data and Vendi scores.
        """
        self.df[dimension] = self.df[dimension].apply(lambda x: 'missing' if pd.isna(x) else x)

        ks = self.df[self.df[dimension] != 'missing'][dimension].unique()  # exclude those not clustered (topic or entities)
        dim = len(ks)
        subsamples = {}

        for alpha in alpha_values:
            distributions = self.generate_random_distribution(dim, alpha, rep)

            for idx, distribution in enumerate(distributions, start=1):
                remaining_samples = subset_size
                sample_sizes = {k: 0 for k in ks}
                max_sample_sizes = {k: len(self.df[self.df[dimension] == k]) for k in ks}

                # First pass: Calculate sample size for each category based on Dirichlet distribution
                for k, sample_proportion in zip(ks, distribution):
                    if pd.isna(sample_proportion) or sample_proportion <= 0:
                        sample_size = 0  
                    else:
                        sample_size = int(subset_size * sample_proportion)
                    sample_sizes[k] = min(sample_size, max_sample_sizes[k])
                    remaining_samples -= sample_sizes[k]

                # Allocate remaining samples to available categories
                available_categories = [k for k in ks if sample_sizes[k] + remaining_samples < max_sample_sizes[k]]
                if remaining_samples > 0 and available_categories:
                    k = np.random.choice(available_categories)
                    sample_sizes[k] += remaining_samples
                    available_categories = [k for k in ks if sample_sizes[k] < max_sample_sizes[k]]

                sampled_data = []
                for k, sample_size in sample_sizes.items():
                    if sample_size > 0:
                        bin_data = self.df[self.df[dimension] == k]
                        sampled_data.append(bin_data.sample(n=sample_size, replace=False))

                if sampled_data:
                    subset_df = pd.concat(sampled_data)
                    vendi_score = score_function(subset_df)
                    alpha_name = str(alpha).replace('.', '_')
                    subsamples[f"alpha_{alpha_name}_idx_{idx}"] = {
                        "data": subset_df,
                        "vs": round(vendi_score, 3)
                    }

        # Generate perfectly uniform subsets (equal proportions)
        ks = self.df[dimension].unique()
        dim = len(ks)

        inf_alpha = np.ones(dim) / dim  # Equal proportions for each category
        for idx in range(rep):
            remaining_samples = subset_size
            max_sample_sizes = {k: len(self.df[self.df[dimension] == k]) for k in ks}
            sample_sizes = {
                k: min(int(subset_size * inf_alpha[i]), max_sample_sizes[k])
                for i, k in enumerate(ks)
            }
            remaining_samples -= sum(sample_sizes.values())

            while remaining_samples > 0:
                available_categories = [k for k in ks if sample_sizes[k] < max_sample_sizes[k]]
                if not available_categories:
                    break

                for k in available_categories:
                    if remaining_samples > 0 and sample_sizes[k] < max_sample_sizes[k]:
                        sample_sizes[k] += 1
                        remaining_samples -= 1

            sampled_data = []
            for k, sample_size in sample_sizes.items():
                if sample_size > 0:
                    bin_data = self.df[self.df[dimension] == k]
                    sampled_data.append(bin_data.sample(n=sample_size, replace=False))

            if sampled_data:
                subset_df = pd.concat(sampled_data)
                vendi_score = score_function(subset_df)
                subsamples[f"alpha_inf_idx_{idx+1}"] = {
                    "data": subset_df,
                    "vs": round(vendi_score, 3)
                }

        return subsamples

    def process_vendi_scores(self, subsamples):
        """
        Normalize Vendi scores and categorize them into quintiles and deciles.
        """
        subset_keys = list(subsamples.keys())
        vendi_scores = [subsamples[key]['vs'] for key in subset_keys]

        max_vendi = max(vendi_scores)
        normalized_scores = [round(score / max_vendi, 5) for score in vendi_scores]

        alpha_pattern = r'_(\d+(?:_\d+)?|inf)_'
        alphas = [
            re.search(alpha_pattern, key).group(1).replace("_", ".") if re.search(alpha_pattern, key) else None
            for key in subset_keys
        ]

        for i, alpha in enumerate(alphas):
            if alpha is not None:
                alphas[i] = float(alpha) if alpha != "inf" else float('inf')

        return pd.DataFrame({
            "subsample": subset_keys,
            "alpha": alphas,
            "vs": vendi_scores,
            "normalized_vs": normalized_scores
        })

    def save_subsamples(self, subsamples, dataset_name, base_path="data/subsamples"):
        """
        Save subsets to pickle files.
        """
        os.makedirs(base_path, exist_ok=True)

        # Define the file path for saving
        pickle_file_path = os.path.join(base_path, f"{dataset_name}.pkl")

        # Save the subsamples dictionary to a pickle file
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(subsamples, f)

        print(f"Subsamples saved to: {pickle_file_path}")

        # for subset_name, value in subsamples.items():
        #     value['data'].to_csv(os.path.join(path, f"{subset_name}.csv"), index=False)
