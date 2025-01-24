import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
# from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns

# def read_results(dataset):

#     file_path = f"../data/output/results_{dataset}.csv"
#     df = pd.read_csv(file_path).drop("Unnamed: 0", axis=1)

#     info_file = f"../data/subsamples/source/div_subsamples_{dataset}.csv"
#     subsamples_info = pd.read_csv(info_file).drop("Unnamed: 0", axis=1)

#     merged_df = pd.merge(df, subsamples_info, on="subsample").sort_values("vs")

#     return merged_df


def plot_metrics(df, name):
    """
    Plots performance metrics against diversity scores.

    Args:
        df (pd.DataFrame): DataFrame containing metrics and diversity scores.
        name (str): Title for the plot.
    """
    colors = sns.color_palette("Set2", len(['mcc', 'f1', 'precision', 'recall', 'roc_auc']))

    plt.figure(figsize=(8, 5))

    # Create a line plot for each metric
    for i, metric in enumerate(['mcc', 'f1', 'precision', 'recall', 'roc_auc']):
        plt.plot(df['vs'], df[metric], color=colors[i], label=metric)

    # Add labels, title, and legend
    plt.xlabel('Diversity Score (VS)', fontsize=12)
    plt.ylabel('Performance Metrics', fontsize=12)
    plt.title(name, fontsize=14)
    plt.legend(title='Metrics', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=10)

    plt.tight_layout()
    plt.show()


def regression_vs_metric(df, metric):
    """
    Performs a linear regression of a performance metric on diversity scores.

    Args:
        df (pd.DataFrame): DataFrame containing metrics and diversity scores.
        metric (str): Metric to be regressed (e.g., 'mcc', 'f1').
    """
    X = df['vs']
    y = df[metric]

    # Add constant for regression intercept
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(f"\nRegression Results for {metric}:\n")
    print(model.summary())


def calculate_correlations(df):
    """
    Calculates Pearson and Spearman correlations between metrics and diversity scores.

    Args:
        df (pd.DataFrame): DataFrame containing metrics and diversity scores.

    Returns:
        pd.DataFrame: Correlation results for each metric.
    """
    results = []
    performance_metrics = ['mcc', 'f1', 'precision', 'recall', 'roc_auc']

    # Calculate correlations for each metric
    for metric in performance_metrics:
        pearson_corr, pearson_p = pearsonr(df['vs'], df[metric])
        spearman_corr, spearman_p = spearmanr(df['vs'], df[metric])

        # Format the results for clarity
        pearson_corr_str = f"{pearson_corr:.3f}"
        spearman_corr_str = f"{spearman_corr:.3f}"

        if pearson_p < 0.01:
            pearson_corr_str += "**"
        elif pearson_p < 0.05:
            pearson_corr_str += "*"

        if spearman_p < 0.01:
            spearman_corr_str += "**"
        elif spearman_p < 0.05:
            spearman_corr_str += "*"

        results.append({
            'Metric': metric,
            'Pearson Corr.': pearson_corr_str,
            'Pearson P-value': pearson_p,
            'Spearman Corr.': spearman_corr_str,
            'Spearman P-value': spearman_p
        })

    # Convert results to a DataFrame for display
    results_df = pd.DataFrame(results)

    return results_df