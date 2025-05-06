import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader


from loguru import logger 
import networkx as nx
from tqdm import tqdm



# ============= Graph VAE Model =============


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):

            in_c = in_channels if i == 0 else hidden_channels
            nn = torch.nn.Sequential(
                torch.nn.Linear(in_c, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
            )
            self.convs.append(GINConv(nn))


        self.lin_mu = torch.nn.Linear(hidden_channels, latent_dim)
        self.lin_logvar = torch.nn.Linear(hidden_channels, latent_dim)

    def forward(self, x, edge_index, batch):
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Global add pooling to get graph-level embedding
        g = global_add_pool(x, batch)
        mu = self.lin_mu(g)
        logvar = self.lin_logvar(g)
        return mu, logvar


class GraphDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_channels, max_nodes):
    
        super().__init__()
        self.lin1 = torch.nn.Linear(latent_dim, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, max_nodes * max_nodes)


        self.max_nodes = max_nodes

    def forward(self, z):
    
        h = F.relu(self.lin1(z))
        adj_logits = self.lin2(h)  # shape [batch, N*N]


        # reshape to [batch, N, N]
        batch_size = z.size(0)
        adj_logits = adj_logits.view(batch_size, self.max_nodes, self.max_nodes)
        # force symmetry
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2
    
        return adj_logits


# |
# |
# |
# |
# |
# |
# v
# Decoder with skip connections + extra layer


# class GraphDecoder(torch.nn.Module):
#     def __init__(self, latent_dim, hidden_channels, max_nodes):
#         super().__init__()
#         self.max_nodes = max_nodes

#         self.fc1 = torch.nn.Linear(latent_dim, hidden_channels)
#         self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
#         self.fc3 = torch.nn.Linear(hidden_channels, max_nodes * max_nodes)

#         self.norm = nn.LayerNorm(hidden_channels)

#     def forward(self, z):
#         h = F.relu(self.fc1(z))
        
#         # Skip connection: h + fc2(h)
#         residual = h
#         h = F.relu(self.fc2(h))
#         h = self.norm(h + residual)

#         adj_logits = self.fc3(h)

#         # Reshape to [batch, N, N]
#         batch_size = z.size(0)
#         adj_logits = adj_logits.view(batch_size, self.max_nodes, self.max_nodes)

#         # Symmetrize
#         adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

#         return adj_logits




class GraphVAE(torch.nn.Module):
    
    def __init__(
        self, in_channels, hidden_channels, latent_dim, max_nodes, num_layers=3
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            in_channels, hidden_channels, latent_dim, num_layers
        )
        self.decoder = GraphDecoder(latent_dim, hidden_channels, max_nodes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        mu, logvar = self.encoder(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        
        adj_logits = self.decoder(z)

        # Build target dense adjacency
        A = to_dense_adj(edge_index, batch, max_num_nodes=adj_logits.size(1))
        
        return adj_logits, A, mu, logvar

    def loss(self, adj_logits, A, mu, logvar):
        # Reconstruction: BCE (with logits)
        recon_loss = F.binary_cross_entropy_with_logits(adj_logits, A, reduction="sum")
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld


# =========== Training Loop ============


def train_vae(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj_logits, A, mu, logvar = model(data)
        loss = model.loss(adj_logits, A, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


# ========== Sampling ============

# def generate_graphs_with_model(model,max_nodes, num_samples=1000, device="cpu", threshold=0.5):
#     """
#     Samples graphs from a trained GraphVAE model.

#     Args:
#         model: Trained GraphVAE model
#         num_samples: Number of graphs to generate
#         device: Device to run sampling on
#         threshold: Sigmoid threshold for binarizing edges

#     Returns:
#         A list of NetworkX Graphs
#     """
#     model.eval()
#     generated_graphs = []

#     attempted = 0
#     progress = tqdm(total=num_samples, desc="Generating graphs")

#     while len(generated_graphs) < num_samples:
#         attempted += 1

#         z = torch.randn(1, model.encoder.lin_mu.out_features).to(device)
#         adj_logits = model.decoder(z)  # shape [1, N, N]
#         adj_probs = torch.sigmoid(adj_logits)[0]  # shape [N, N]

#         # Binarize using threshold
#         adj_bin = (adj_probs > threshold).float()

#         # Remove self-loops
#         adj_bin.fill_diagonal_(0)

#         # Convert to networkx graph
#         G = nx.from_numpy_array(adj_bin.cpu().numpy())

#         # Skip disconnected graphs if needed
#         if nx.is_connected(G):
#             generated_graphs.append(G)
            
#         progress.update(1)


#     logger.info(
#         f"Generated {len(generated_graphs)} connected graphs from {attempted} attempts"
#     )

#     return generated_graphs


def generate_graphs_with_model(model, max_nodes, num_samples=1000, device="cpu", threshold=0.5, batch_size=32):
    """
    Efficiently sample connected graphs from a trained GraphVAE model.
    """
    model.eval()
    generated_graphs = []
    attempted = 0
    progress = tqdm(desc="Generating connected graphs")

    with torch.no_grad():
        while len(generated_graphs) < num_samples:
            z = torch.randn(batch_size, model.encoder.lin_mu.out_features).to(device)
            adj_logits = model.decoder(z)  # shape [B, N, N]
            adj_probs = torch.sigmoid(adj_logits)

            # Binarize and zero diagonal
            adj_bin = (adj_probs > threshold).float()
            adj_bin[:, torch.arange(max_nodes), torch.arange(max_nodes)] = 0  # remove self-loops

            # Convert to networkx and filter
            for i in range(batch_size):
                attempted += 1
                adj_np = adj_bin[i].cpu().numpy()
                G = nx.from_numpy_array(adj_np)

                if nx.is_connected(G):
                    generated_graphs.append(G)
                    progress.update(1)

                progress.set_postfix({
                    "accepted": len(generated_graphs),
                    "attempted": attempted,
                    "accept_rate": f"{len(generated_graphs)/max(attempted,1):.2%}"
                })

                if len(generated_graphs) >= num_samples:
                    break



    logger.info(f"Generated {len(generated_graphs)} connected graphs from {attempted} attempts")
    return generated_graphs
