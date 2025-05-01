import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
from utils.dataset_loader import load_dataset

from collections import defaultdict
from loguru import logger
from tqdm import tqdm

# ======= Baseline Model - Erdös-Rényi =====

def sample_erdos_renyi(num_samples=1000):
    
    train_loader, _, _ = load_dataset()
    train_graphs = [to_networkx(data, to_undirected=True) for data in train_loader.dataset]

    plot_er_graph(train_graphs[0])

    # Step 1: Collect node counts and densities
    node_counts = []
    density_by_n = defaultdict(list)

    for g in train_graphs:
        
        N = g.number_of_nodes()
        E = g.number_of_edges()

        if N > 1:  # skip degenerate graphs
            density = 2 * E / (N * (N - 1))
            node_counts.append(N)
            density_by_n[N].append(density)

    # Convert node_counts to tensor for sampling
    node_counts_tensor = torch.tensor(node_counts)
    unique_node_counts, counts = torch.unique(node_counts_tensor, return_counts=True)
    probs = counts.float() / counts.sum()

    er_graphs = []

    for _ in tqdm(range(num_samples), desc="Sampling Erdős–Rényi graphs"):

        # Sample N from empirical distribution
        N = int(unique_node_counts[torch.multinomial(probs, 1)])

        # Estimate average density for this N
        avg_density = np.mean(density_by_n[N])
        p = min(max(avg_density, 0.001), 1.0)  # clip to avoid degenerate graphs

        # Sample Erdos-Renyi graph
        G = nx.erdos_renyi_graph(n=N, p=p)
        er_graphs.append(G)

    logger.success(f"Sampled {len(er_graphs)} Erdős–Rényi graphs.")
    return er_graphs

# ====== Plotting =====

def plot_er_graph(G, title="Sample Erdős–Rényi Graph"):
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.show()



if __name__=="__main__":
	er_graphs = sample_erdos_renyi(num_samples=1000)
	plot_er_graph(er_graphs[10])
	

