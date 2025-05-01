import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

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
        'degree': np.array(degrees),
        'clustering': np.array(clusterings),
        'eigenvector_centrality': np.array(centralities)
    }


def plot_statistic_comparisons(
    empirical_stats: Dict[str, np.ndarray],
    er_stats: Optional[Dict[str, np.ndarray]] = None,
    gnn_stats: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = "./figs"
):
    """
    Plots histograms of degree, clustering coefficient, and eigenvector centrality
    for empirical, Erdős–Rényi, and generative model graphs.
    
    Optionally saves the plots to disk.
    """
    stats = ['degree', 'clustering', 'eigenvector_centrality']
    titles = ['Node Degree', 'Clustering Coefficient', 'Eigenvector Centrality']

    for stat, title in zip(stats, titles):
        plt.figure(figsize=(8, 5))
        plt.hist(empirical_stats[stat], bins=30, alpha=0.5, label='Empirical (MUTAG)', density=True)

        if er_stats is not None:
            plt.hist(er_stats[stat], bins=30, alpha=0.5, label='Erdős–Rényi', density=True)

        if gnn_stats is not None:
            plt.hist(gnn_stats[stat], bins=30, alpha=0.5, label='Generative Model', density=True)

        plt.title(f'Distribution of {title}')
        plt.xlabel(title)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            filename = f"{stat}_distribution.png"
            plt.savefig(f"{save_path}/{filename}")
        else:
            plt.show()