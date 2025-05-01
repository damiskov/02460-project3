from torch_geometric.utils import to_networkx
from utils.dataset_loader import load_dataset

from models.erdos_renyi import sample_erdos_renyi

from evaluation.statistics import compute_graph_statistics
from evaluation.statistics import plot_statistic_comparisons

# Load empirical graphs
train_loader, _, _ = load_dataset()
empirical_graphs = [to_networkx(data, to_undirected=True) for data in train_loader.dataset]
empirical_stats = compute_graph_statistics(empirical_graphs)




# TODO: Sampling of baseline

# er_graphs = sample_erdos_renyi(num_samples=1000)
# er_stats = compute_graph_statistics(er_graphs)

# TODO: Sampling of generative graphs

# gnn_graphs = generate_graphs_with_model()
# gnn_stats = compute_graph_statistics(gnn_graphs)

# Plot comparisons
plot_statistic_comparisons(empirical_stats)  # Add other statistics when finished