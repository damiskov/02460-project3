# misc.
from loguru import logger
from tqdm import tqdm
import networkx as nx

# config
from config import TRAIN_VAE
from config import NUM_SAMPLES

from config import IN_CHANNELS
from config import MAX_NODES
from config import HIDDEN_CHANNELS
from config import LATENT_DIM
from config import EPOCHS
from config import NUM_LAYERS

# torch
import torch
from torch_geometric.utils import to_networkx

# models
from models.erdos_renyi import sample_erdos_renyi

from models.gan import GraphGenerator
from models.gan import GraphDiscriminator
from models.gan import nx_to_data
from models.gan import generate_graphs_with_gan

from models.vae import GraphVAE
from models.vae import train_vae
from models.vae import generate_graphs_with_model

# utils
from utils.dataset_loader import load_full_dataset

# evaluation
from evaluation.statistics import compute_graph_statistics
from evaluation.statistics import plot_graph_statistics



# Load mutag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = load_full_dataset()

# ========= Empirical Data =========

empirical_graphs = [to_networkx(data, to_undirected=True) for data in loader.dataset]
empirical_stats = compute_graph_statistics(empirical_graphs)
plot_graph_statistics(
    data=empirical_stats,
    name="Empirical (MUTAG)",
    color="cornflowerblue",
    save_path="figs/mutag",
)

# ========= Baseline - Erdos-Renyi =========

er_graphs = sample_erdos_renyi(num_samples=1000)
er_stats = compute_graph_statistics(er_graphs)
plot_graph_statistics(
    data=er_stats, name="Erdős–Rényi", color="lightcoral", save_path="figs/er"
)

# ========= Generative Model - VAE ==========


if TRAIN_VAE:
    logger.info("Training GraphVAE...")

    model = GraphVAE(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        latent_dim=LATENT_DIM,
        max_nodes=MAX_NODES,
        num_layers=NUM_LAYERS,  # Increased number of layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    progress_bar = tqdm(range(1, EPOCHS + 1), desc="Training")

    for epoch in progress_bar:
        loss = train_vae(model, loader, optimizer, device)
        progress_bar.set_description(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "models/graph_vae.pt")
    logger.success("Saved GraphVAE model to models/graph_vae.pt")

else:
    model = GraphVAE(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        latent_dim=LATENT_DIM,
        max_nodes=MAX_NODES,
        num_layers=NUM_LAYERS,  # Increased number of layers
    )

    try:
        model.load_state_dict(torch.load("models/graph_vae.pt", map_location="cpu"))
    except:
        raise NotImplementedError("Can not find state dict - please train GraphVAE")

    logger.success("Loaded trained GraphVAE model for sampling")


logger.info("Begin sampling...")

# ========= Sampling =========

vae_graphs = generate_graphs_with_model(
    model, max_nodes=MAX_NODES, num_samples=NUM_SAMPLES, device=device
)

# Compute stats

vae_stats = compute_graph_statistics(vae_graphs)

# Plot comparison of graph statistics
plot_graph_statistics(
    data=vae_stats,
    name="Generated Graphs (VAE)",
    color="orchid",
    save_path="figs/gen/vae",
)


# =========== WL tests ===========


logger.success("main.py complete")
