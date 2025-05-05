# misc.
import networkx as nx
import numpy as np
from tqdm import tqdm


# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

# torch_geometric
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

# ======= Generator =======


class GraphGenerator(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=128, num_nodes=28):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.output_dim = (num_nodes * (num_nodes - 1)) // 2  # upper triangle only

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, z):
        """
        z: latent vector of shape [batch_size, latent_dim]
        returns: list of networkx graphs (length = batch_size)
        """
        batch_size = z.size(0)
        logits = self.mlp(z)  # shape: [batch_size, output_dim]
        graphs = []

        for i in range(batch_size):
            upper = torch.sigmoid(logits[i])  # probabilities in [0, 1]
            adj_matrix = self.reconstruct_symmetric_adjacency(upper)
            G = nx.from_numpy_array(adj_matrix)
            graphs.append(G)

        return graphs

    def reconstruct_symmetric_adjacency(self, upper_flat):
        """
        Reconstruct a symmetric adjacency matrix from its upper-triangular entries.
        Thresholds edges at 0.5.
        """
        adj = np.zeros((self.num_nodes, self.num_nodes))
        triu_indices = np.triu_indices(self.num_nodes, k=1)
        adj[triu_indices] = upper_flat.detach().cpu().numpy()
        adj = adj + adj.T  # make symmetric
        adj = (adj > 0.5).astype(np.float32)  # binarize
        return adj


# ======= Discriminator =======


def nx_to_data(graph):
    """
    Converts a NetworkX graph to a PyG Data object with constant node features.
    """
    edge_index = pyg_utils.from_networkx(graph).edge_index
    num_nodes = graph.number_of_nodes()
    x = torch.ones((num_nodes, 1), dtype=torch.float)  # use constant node features
    return Data(x=x, edge_index=edge_index)


class GraphDiscriminator(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, num_node_features=1):
        super().__init__()
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(num_node_features if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        g_emb = global_add_pool(x, batch)  # shape [batch_size, hidden_dim]
        out = self.fc(g_emb)  # shape [batch_size, 1]
        return out.squeeze(-1)  # shape [batch_size]


def generate_graphs_with_gan(
    generator, num_samples=1000, latent_dim=32, batch_size=32, device="cpu"
):
    """
    Samples graphs from a trained GraphGAN generator.

    Args:
        generator: Trained GraphGenerator model
        num_samples: Number of graphs to generate
        latent_dim: Dimensionality of noise vector
        batch_size: Number of samples per forward pass
        device: torch device

    Returns:
        List of networkx.Graph objects
    """
    generator.eval()
    graphs = []

    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, batch_size), desc="Generating GAN samples"):
            z = torch.randn(batch_size, latent_dim, device=device)
            batch_graphs = generator(z)
            graphs.extend(batch_graphs)

    return graphs[:num_samples]
