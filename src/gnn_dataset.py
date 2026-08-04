import torch
import networkx as nx
from torch_geometric.data import Data


def _build_node_index(G: nx.DiGraph) -> dict:
    """Създава mapping от IP адрес към числов индекс."""
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    return node_to_idx


def _build_edge_index(G: nx.DiGraph, node_to_idx: dict) -> torch.Tensor:
    """Строи edge_index тензор от ребрата на графа."""
    edges = list(G.edges())
    
    src_indices = [node_to_idx[src] for src, dst in edges]
    dst_indices = [node_to_idx[dst] for src, dst in edges]

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

    return edge_index

def _build_labels(G: nx.DiGraph, node_to_idx: dict) -> torch.Tensor:
    """Строи label тензор (is_botnet) за всеки възел."""
    num_nodes = len(node_to_idx)
    labels = torch.zeros(num_nodes, dtype=torch.long)

    for node, idx in node_to_idx.items():
        labels[idx] = G.nodes[node]['is_botnet']

    return labels

def build_pyg_data(G: nx.DiGraph, features_df) -> Data:
    """
    Конвертира NetworkX граф към PyTorch Geometric Data обект.

    Args:
        G: насочен граф от build_graph()
        features_df: DataFrame от extract_features() с готови metrics

    Returns:
        PyG Data обект готов за GNN модел
    """
    print("Конвертиране към PyG формат...")

    node_to_idx = _build_node_index(G)

    # Използвай ГОТОВИТЕ features от feature_extractor.py
    features_lookup = features_df.set_index('ip')
    
    num_nodes = len(node_to_idx)
    x = torch.zeros((num_nodes, 4))
    
    for node, idx in node_to_idx.items():
        x[idx, 0] = features_lookup.loc[node, 'in_degree']
        x[idx, 1] = features_lookup.loc[node, 'out_degree']
        x[idx, 2] = features_lookup.loc[node, 'pagerank']
        x[idx, 3] = features_lookup.loc[node, 'betweenness']

    edge_index = _build_edge_index(G, node_to_idx)
    y = _build_labels(G, node_to_idx)

    data = Data(x=x, edge_index=edge_index, y=y)

    print(f"Възли: {data.num_nodes:,}")
    print(f"Ребра: {data.num_edges:,}")
    print(f"Features размерност: {data.num_node_features}")

    return data