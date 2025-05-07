import networkx as nx

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
