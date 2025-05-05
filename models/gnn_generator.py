import torch
import networkx as nx

from loguru import logger


def generate_graphs_with_model(model, num_samples=1000, device="cpu", threshold=0.5):
    """
    Samples graphs from a trained GraphVAE model.

    Args:
        model: Trained GraphVAE model
        num_samples: Number of graphs to generate
        device: Device to run sampling on
        threshold: Sigmoid threshold for binarizing edges

    Returns:
        A list of NetworkX Graphs
    """
    model.eval()
    generated_graphs = []

    max_nodes = model.decoder.lin2.out_features
    max_nodes = int(max_nodes**0.5)  # since output is flattened N*N

    attempted = 0
    while len(generated_graphs) < num_samples:
        attempted += 1

        z = torch.randn(1, model.encoder.lin_mu.out_features).to(device)
        adj_logits = model.decoder(z)  # shape [1, N, N]
        adj_probs = torch.sigmoid(adj_logits)[0]  # shape [N, N]

        # Binarize using threshold
        adj_bin = (adj_probs > threshold).float()

        # Remove self-loops
        adj_bin.fill_diagonal_(0)

        # Convert to networkx graph
        G = nx.from_numpy_array(adj_bin.cpu().numpy())

        # Skip disconnected graphs if needed
        if nx.is_connected(G):
            generated_graphs.append(G)

            if attempted % 100 == 0:
                logger.info(f"Generated {len(generated_graphs)}")

        # if len(generated_graphs) >= num_samples:
        #     break

    logger.info(
        f"Generated {len(generated_graphs)} connected graphs from {attempted} attempts"
    )

    return generated_graphs
