# misc.
from loguru import logger
from tqdm import tqdm
import networkx as nx

# torch
import torch
from torch_geometric.utils import to_networkx
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ExponentialLR

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = load_full_dataset()


# ========= Empirical Data =========

empirical_graphs = [to_networkx(data, to_undirected=True) for data in loader.dataset]
empirical_stats = compute_graph_statistics(empirical_graphs)
plot_graph_statistics(data=empirical_stats, name="Empirical (MUTAG)", color='cornflowerblue', save_path="figs/mutag")

# ========= Baseline - Erdos-Renyi =========

er_graphs = sample_erdos_renyi(num_samples=1000)
er_stats = compute_graph_statistics(er_graphs)
plot_graph_statistics(data=er_stats, name="Erdős–Rényi", color='lightcoral', save_path="figs/er")

# ========= Generative Model - GAN ==========

# patience = 10
# min_delta = 1e-3
# best_g_loss = float('inf')
# patience_counter = 0

# # Hyperparameters
# latent_dim = 32
# num_nodes = 28
# batch_size = 16
# num_epochs = 100
# lr = 1e-4

# # Load real graphs
# loader = load_full_dataset()
# real_graphs = [to_networkx(data, to_undirected=True) for data in loader.dataset]

# # Instantiate models
# generator = GraphGenerator(latent_dim=latent_dim, num_nodes=num_nodes).to(device)
# discriminator = GraphDiscriminator(num_node_features=1).to(device)

# # Optimizers and loss
# g_opt = torch.optim.Adam(generator.parameters(), lr=lr)
# d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
# g_scheduler = ExponentialLR(g_opt, gamma=0.99)
# d_scheduler = ExponentialLR(d_opt, gamma=0.99)

# loss_fn = BCEWithLogitsLoss()

# for epoch in range(1, num_epochs + 1):

#     total_d_loss = 0
#     total_g_loss = 0

#     for _ in range(len(real_graphs) // batch_size):

#         # === Train Discriminator ===
#         discriminator.train()
#         generator.eval()

#         # Sample real graphs
#         real_batch = real_graphs[:batch_size]
#         real_data = [nx_to_data(g) for g in real_batch]
#         real_labels = torch.full((batch_size,), 0.9, device=device)

#         for d in real_data:
#             d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
        
#         real_data = Batch.from_data_list(real_data).to(device)
#         #real_labels = torch.ones(real_data.num_graphs, device=device)
#         # Trying 'label smoothing'
#         real_labels = torch.full((batch_size,), 0.9, device=device)

#         # Generate fake graphs
#         z = torch.randn(batch_size, latent_dim).to(device)
#         fake_graphs = generator(z)
#         fake_data = [nx_to_data(g) for g in fake_graphs]

#         for d in fake_data:
#             d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
        
#         fake_data = Batch.from_data_list(fake_data).to(device)
#         fake_labels = torch.zeros(fake_data.num_graphs, device=device)

#         # Forward pass
#         d_real = discriminator(real_data)
#         d_fake = discriminator(fake_data)

#         # Loss and update
#         d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)
#         d_opt.zero_grad()
#         d_loss.backward()
#         d_opt.step()


#         # === Train Generator ===
#         generator.train()
#         discriminator.eval()

#         z = torch.randn(batch_size, latent_dim).to(device)
#         fake_graphs = generator(z)
#         fake_data = [nx_to_data(g) for g in fake_graphs]

#         for d in fake_data:
#             d.batch = torch.zeros(d.num_nodes, dtype=torch.long)

#         fake_data = Batch.from_data_list(fake_data).to(device)
#         fake_labels = torch.ones(fake_data.num_graphs, device=device)  # trick D

#         d_fake = discriminator(fake_data)
#         g_loss = loss_fn(d_fake, fake_labels)

#         # Some early stopping criteria

#         g_opt.zero_grad()
#         g_loss.backward()
#         g_opt.step()

#         total_d_loss += d_loss.item()
#         total_g_loss += g_loss.item()
    
#     # Learning rate scheduler step
#     g_scheduler.step()
#     d_scheduler.step()

#     logger.info(f"Epoch {epoch:03d} | D Loss: {total_d_loss:.4f} | G Loss: {total_g_loss:.4f}")


# gan_graphs = generate_graphs_with_gan(
#     generator=generator,
#     num_samples=1000,
#     latent_dim=latent_dim,
#     batch_size=32,
#     device=device
# )

# gan_stats = compute_graph_statistics(gan_graphs)
# plot_graph_statistics(data=gan_stats, name="Generated Graphs (GAN)", color='mediumseagreen', save_path="figs/gen/gan")


# ========= Generative Model - VAE ==========

logger.info("Training GraphVAE...")

in_channels = loader.dataset[0].num_node_features
max_nodes = max(g.num_nodes for g in loader.dataset)

logger.debug(f"In channels: {in_channels}")
logger.debug(f"max nodes: {max_nodes}")

# Hyperparameters

hidden_channels = 64
latent_dim = 32


model = GraphVAE(
    in_channels=in_channels, hidden_channels=hidden_channels, latent_dim=latent_dim, max_nodes=max_nodes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 500
progress_bar = tqdm(range(1, epochs + 1), desc="Training")


for epoch in progress_bar:
    loss = train_vae(model, loader, optimizer, device)
    progress_bar.set_description(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

logger.success("Completed training GraphVAE")

logger.info("Begin sampling...")
# Generate graphs from the trained VAE
vae_graphs = generate_graphs_with_model(model, num_samples=100, device=device)
vae_stats = compute_graph_statistics(vae_graphs)

# 5. Plot comparison of graph statistics
plot_graph_statistics(
    data=vae_stats, name="Generated Graphs (VAE)", 
    color='orchid', save_path="figs/gen/vae"
)



