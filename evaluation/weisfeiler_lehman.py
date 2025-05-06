import networkx as nx
from loguru import logger


def uniqueness(graphs: list[nx.Graph]) -> tuple[float, set]:
    """
    Computes the percentage of unique graphs for a list of NetworkX graphs.

    Returns the percentage as well as the hash table as a set for use in computing the novely.
    """
    hash_table = set()
    for G in graphs:
        hash_table.add(nx.weisfeiler_lehman_graph_hash(G))
    return len(hash_table)/len(graphs), hash_table


def novelty(graphs: list[nx.Graph], graphs_train: list[nx.Graph]):
    """
    Computes the novelty for a list of NetworkX graphs.

    Returns the percentage of novel graphs and a hash table. 
    The hash table contains the hashes for graphs that are both novel and unique.
    """
    _, train_hash = uniqueness(graphs_train)
    n = 0
    hash_table = set()
    for G in graphs:
        h = nx.weisfeiler_lehman_graph_hash(G)
        if h not in train_hash:
            hash_table.add(h)
            n += 1
    return n / len(graphs), hash_table

def novelty_uniqueness(graphs: list[nx.Graph], graphs_train: list[nx.Graph]) -> dict[str, float]:
    unique_per, _ = uniqueness(graphs)
    novel_per, hash_set = novelty(graphs, graphs_train)
    nu_per = len(hash_set)/len(graphs)
    return {
        "novel": novel_per,
        "unique": unique_per,
        "novel_unique": nu_per
    }


def compute_novel_unique_metrics(
        empirical_graphs: list[nx.Graph], 
        er_graphs: list[nx.Graph], 
        gnn_graphs: list[nx.Graph],
        save_path: str|None = None):
    
    if save_path:
        logger.add(f"{save_path}/metrics.log")
    
    er_metrics = novelty_uniqueness(er_graphs, empirical_graphs)
    logger.info("Erdos-Renyi (ER):")
    logger.info(f"Novel: {er_metrics['novel']}")
    logger.info(f"Unique: {er_metrics['unique']}")
    logger.info(f"Novel and unique: {er_metrics['novel_unique']}")

    gnn_metrics = novelty_uniqueness(gnn_graphs, gnn_graphs)
    logger.info("Graph Neural Network (GNN):")
    logger.info(f"Novel: {gnn_metrics['novel']}")
    logger.info(f"Unique: {gnn_metrics['unique']}")
    logger.info(f"Novel and unique: {gnn_metrics['novel_unique']}")

    
if __name__ == '__main__':
    logger.info("beep")