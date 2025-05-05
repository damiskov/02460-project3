import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from loguru import logger
import seaborn as sns

sns.set_theme()


def compute_graph_statistics(graphs: List[nx.Graph]) -> Dict[str, np.ndarray]:
    """
    Computes degree, clustering coefficient, and eigenvector centrality for a list of NetworkX graphs.

    Returns a dictionary with keys: 'degree', 'clustering', 'eigenvector_centrality'.
    """
    degrees = []
    clusterings = []
    centralities = []

    for g in graphs:
        degrees.extend([d for _, d in g.degree()])
        clusterings.extend(list(nx.clustering(g).values()))

        try:
            if nx.is_connected(g):
                ec = nx.eigenvector_centrality(g, max_iter=1000)
                centralities.extend(list(ec.values()))

        except nx.NetworkXException:
            # Skip graphs where eigenvector centrality fails to converge
            continue

    return {
        "degree": np.array(degrees),
        "clustering": np.array(clusterings),
        "eigenvector_centrality": np.array(centralities),
    }


def plot_statistic_comparisons(
    empirical_stats: Dict[str, np.ndarray],
    er_stats: Optional[Dict[str, np.ndarray]] = None,
    gnn_stats: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """
    Plots histograms of degree, clustering coefficient, and eigenvector centrality
    for empirical, Erdős–Rényi, and generative model graphs.

    Optionally saves the plots to disk.
    """
    stats = ["degree", "clustering", "eigenvector_centrality"]
    titles = ["Node Degree", "Clustering Coefficient", "Eigenvector Centrality"]

    for stat, title in zip(stats, titles):
        plt.figure(figsize=(8, 5))

        if len(empirical_stats[stat]) > 0:
            plt.hist(
                empirical_stats[stat],
                bins=30,
                alpha=0.5,
                label="Empirical (MUTAG)",
                density=True,
            )
        else:
            logger.warning(f"Empirical (MUTAG) {title} is empty")

        if er_stats and len(er_stats[stat]) > 0:
            plt.hist(
                er_stats[stat], bins=30, alpha=0.5, label="Erdős–Rényi", density=True
            )
        else:
            logger.warning(f"Erdős–Rényi {title} is empty")

        if gnn_stats and len(gnn_stats[stat]) > 0:
            plt.hist(
                gnn_stats[stat],
                bins=30,
                alpha=0.5,
                label="Generative Model",
                density=True,
            )
        else:
            logger.warning(f"Generative Model {title} is empty")

        plt.title(f"Distribution of {title}")
        plt.xlabel(title)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            filename = f"{stat}_distribution.png"
            plt.savefig(f"{save_path}/{filename}")
        else:
            plt.show()


def plot_graph_statistics(
    data: Dict[str, np.ndarray], name: str, color: str, save_path: Optional[str] = None
):
    """
    Plots histograms of degree, clustering coefficient, and eigenvector centrality

    Optionally saves the plots to disk.
    """
    stats = ["degree", "clustering", "eigenvector_centrality"]
    titles = ["Node Degree", "Clustering Coefficient", "Eigenvector Centrality"]

    for stat, title in zip(stats, titles):
        plt.figure(figsize=(8, 5))

        if len(data[stat]) > 0:
            plt.hist(
                data[stat], bins=30, alpha=1, label=name, density=True, color=color
            )
        else:
            logger.warning(f"statistic {title} for {name} is empty")

        plt.xlabel(title)
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            filename = f"{stat}_distribution.png"
            plt.savefig(f"{save_path}/{filename}")
        else:
            plt.show()
