from loguru import logger
from tqdm import tqdm

import torch
from torch_geometric.utils import to_networkx

from utils.dataset_loader import load_full_dataset

from models.erdos_renyi import sample_erdos_renyi
from models.gnn_vae import GraphVAE, train_vae
from models.gnn_generator import generate_graphs_with_model

from evaluation.statistics import compute_graph_statistics, plot_statistic_comparisons

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the MUTAG dataset
loader = load_full_dataset()

# 2. Load empirical graphs
empirical_graphs = [to_networkx(data, to_undirected=True) for data in loader.dataset]
empirical_stats = compute_graph_statistics(empirical_graphs)

# 3. Generate Erdős–Rényi graphs (baseline)
er_graphs = sample_erdos_renyi(num_samples=1000)
er_stats = compute_graph_statistics(er_graphs)

# 4. Train GraphVAE and generate graphs
logger.info("\nTraining GraphVAE...")
in_channels = loader.dataset[0].num_node_features
max_nodes = max(g.num_nodes for g in loader.dataset)

logger.debug(f"In channels: {in_channels}")
logger.debug(f"max nodes: {max_nodes}")

model = GraphVAE(
    in_channels=in_channels, hidden_channels=64, latent_dim=32, max_nodes=max_nodes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 500
progress_bar = tqdm(range(1, epochs + 1), desc="Training")

for epoch in progress_bar:
    loss = train_vae(model, loader, optimizer, device)
    progress_bar.set_description(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# Generate graphs from the trained VAE
gnn_graphs = generate_graphs_with_model(model, num_samples=1000, device=device)
gnn_stats = compute_graph_statistics(gnn_graphs)

# 5. Plot comparison of graph statistics
plot_statistic_comparisons(
    empirical_stats=empirical_stats, er_stats=er_stats, gnn_stats=gnn_stats
)
