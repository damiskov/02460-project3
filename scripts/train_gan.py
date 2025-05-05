from tqdm import tqdm
from loguru import logger

import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ExponentialLR

from utils.dataset_loader import load_full_dataset


from models.gan import GraphGenerator
from models.gan import GraphDiscriminator
from models.gan import nx_to_data
from models.gan import generate_graphs_with_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


patience = 10
min_delta = 1e-3
best_g_loss = float("inf")
patience_counter = 0

# Hyperparameters
latent_dim = 32
num_nodes = 28
batch_size = 16
num_epochs = 100
lr = 1e-4

# Load real graphs
loader = load_full_dataset()
real_graphs = [to_networkx(data, to_undirected=True) for data in loader.dataset]

# Instantiate models
generator = GraphGenerator(latent_dim=latent_dim, num_nodes=num_nodes).to(device)
discriminator = GraphDiscriminator(num_node_features=1).to(device)

# Optimizers and loss
g_opt = torch.optim.Adam(generator.parameters(), lr=lr)
d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_scheduler = ExponentialLR(g_opt, gamma=0.99)
d_scheduler = ExponentialLR(d_opt, gamma=0.99)

loss_fn = BCEWithLogitsLoss()

for epoch in range(1, num_epochs + 1):
    total_d_loss = 0
    total_g_loss = 0

    for _ in range(len(real_graphs) // batch_size):
        # === Train Discriminator ===
        discriminator.train()
        generator.eval()

        # Sample real graphs
        real_batch = real_graphs[:batch_size]
        real_data = [nx_to_data(g) for g in real_batch]
        real_labels = torch.full((batch_size,), 0.9, device=device)

        for d in real_data:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long)

        real_data = Batch.from_data_list(real_data).to(device)
        # real_labels = torch.ones(real_data.num_graphs, device=device)
        # Trying 'label smoothing'
        real_labels = torch.full((batch_size,), 0.9, device=device)

        # Generate fake graphs
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_graphs = generator(z)
        fake_data = [nx_to_data(g) for g in fake_graphs]

        for d in fake_data:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long)

        fake_data = Batch.from_data_list(fake_data).to(device)
        fake_labels = torch.zeros(fake_data.num_graphs, device=device)

        # Forward pass
        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data)

        # Loss and update
        d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # === Train Generator ===
        generator.train()
        discriminator.eval()

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_graphs = generator(z)
        fake_data = [nx_to_data(g) for g in fake_graphs]

        for d in fake_data:
            d.batch = torch.zeros(d.num_nodes, dtype=torch.long)

        fake_data = Batch.from_data_list(fake_data).to(device)
        fake_labels = torch.ones(fake_data.num_graphs, device=device)  # trick D

        d_fake = discriminator(fake_data)
        g_loss = loss_fn(d_fake, fake_labels)

        # Some early stopping criteria

        if best_g_loss - g_loss.item() > min_delta:
            best_g_loss = g_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.warning(
                f"Early stopping: G loss did not improve for {patience} epochs."
            )
            break

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        total_d_loss += d_loss.item()
        total_g_loss += g_loss.item()

    # Learning rate scheduler step
    g_scheduler.step()
    d_scheduler.step()

    logger.info(
        f"Epoch {epoch:03d} | D Loss: {total_d_loss:.4f} | G Loss: {total_g_loss:.4f}"
    )
