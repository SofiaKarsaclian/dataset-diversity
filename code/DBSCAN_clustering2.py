import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


def load_embeddings_from_file(filename):
    """Loads embeddings from a file."""
    return np.loadtxt(filename)


# Load BERTopic and Top2Vec embeddings
bertopic_embeddings = load_embeddings_from_file("/mounts/data/proj/molly/media_bias/output/bertopic_embeddings.txt")
top2vec_embeddings = load_embeddings_from_file("/mounts/data/proj/molly/media_bias/output/top2vec_embeddings.txt")

# Store embeddings in a dictionary for clustering
features = {
    "BERTopic": bertopic_embeddings,
    "Top2Vec": top2vec_embeddings
}


def dbscan_clustering(embeddings, eps=0.5, min_samples=5, normalize_data=True):
    """Perform DBSCAN clustering on embeddings."""
    # Ensure embeddings are 2D
    if embeddings.ndim == 1:
        print("Embeddings are 1D; reshaping into a single sample with multiple features.")
        embeddings = embeddings.reshape(1, -1)
    if normalize_data:
        embeddings = normalize(embeddings, axis=1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embeddings)
    return labels


def plot_dbscan_clusters(embeddings, labels, model_name, save_path=None):
    """
    Plot and optionally save DBSCAN clustering results using t-SNE.

    Parameters:
    - embeddings: numpy.ndarray
        High-dimensional embeddings to be visualized.
    - labels: numpy.ndarray or list
        Cluster labels for each point (output of DBSCAN).
    - model_name: str
        Name of the model (e.g., "BERTopic" or "Top2Vec").
    - save_path: str, optional
        File path to save the plot. If None, the plot will only be displayed.

    Returns:
    - None. Displays and optionally saves the plot.
    """
    # Skip plotting if embeddings have only one sample
    if embeddings.shape[0] <= 1:
        print(f"Skipping plot for {model_name}: Not enough samples for clustering visualization.")
        return

    # Dimensionality reduction using t-SNE
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Prepare data for visualization
    cluster_data = {
        "Dimension 1": reduced_embeddings[:, 0],
        "Dimension 2": reduced_embeddings[:, 1],
        "Cluster": labels
    }

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=cluster_data["Dimension 1"],
        y=cluster_data["Dimension 2"],
        hue=cluster_data["Cluster"],
        palette="Set2",
        legend="full",
        s=50  # Size of the points
    )
    plt.title(f"DBSCAN Clustering Visualization for {model_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()

    # Save the plot if a save path is specified
    if save_path:
        file_name = f"{save_path}/{model_name.replace(' ', '_')}_clusters.png"
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Plot saved as {file_name}")

    # Show the plot
    plt.show()


def save_cluster_sizes_to_file(cluster_sizes, file_path):
    """
    Save cluster sizes to a file.

    Parameters:
    - cluster_sizes: dict
        Dictionary containing cluster IDs and their sizes.
    - file_path: str
        Path to the output file.

    Returns:
    - None. Writes cluster sizes to the specified file.
    """
    with open(file_path, "w") as f:
        f.write("Cluster ID,Size\n")
        for cluster_id, size in cluster_sizes.items():
            f.write(f"{cluster_id},{size}\n")


# Improved Hyperparameter Grid for DBSCAN
eps_values = np.linspace(0.1, 1.0, 10)  # Epsilon from 0.1 to 1.0 in 10 steps
min_samples_values = [3, 5, 10, 20]  # Different min_samples values
normalize_data_options = [True]  # Assume normalization is critical for cosine similarity

# Define the output directories
output_plot_dir = "/mounts/data/proj/molly/media_bias/output/plots"
output_cluster_size_dir = "/mounts/data/proj/molly/media_bias/output/cluster_sizes"
os.makedirs(output_cluster_size_dir, exist_ok=True)  # Ensure the directory exists

# Perform DBSCAN clustering and visualization
for normalize_data in normalize_data_options:
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nDBSCAN with eps={eps:.2f}, min_samples={min_samples}, normalize_data={normalize_data}")

            for model_name, embeddings in features.items():
                # Skip clustering if embeddings are invalid
                if embeddings.ndim == 1:
                    print(f"Skipping {model_name}: Embeddings are 1D (single sample).")
                    continue

                # Perform DBSCAN with updated dimensionality handling
                labels = dbscan_clustering(embeddings, eps=eps, min_samples=min_samples, normalize_data=normalize_data)

                # Analyze clustering results
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)
                n_noise = list(labels).count(-1)
                print(f"Model: {model_name}")
                print(f" - Number of clusters: {n_clusters}")
                print(f" - Number of noise points: {n_noise}")

                # Compute cluster sizes
                cluster_sizes = {}
                for cluster_id in set(labels):
                    cluster_sizes[cluster_id] = list(labels).count(cluster_id)

                # Save cluster sizes to a file
                cluster_size_file = f"{output_cluster_size_dir}/{model_name}_eps{eps:.2f}_min{min_samples}_norm{normalize_data}.csv"
                save_cluster_sizes_to_file(cluster_sizes, cluster_size_file)
                print(f"Cluster sizes saved to {cluster_size_file}")

                # Visualize and save the clustering results
                plot_dbscan_clusters(
                    embeddings=embeddings,
                    labels=labels,
                    model_name=f"{model_name}_eps{eps:.2f}_min{min_samples}_norm{normalize_data}",
                    save_path=output_plot_dir
                )
