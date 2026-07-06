import networkx as nx
import pandas as pd

def _degree_features(G: nx.DiGraph) -> dict:
    """Изчислява in_degree и out_degree за всеки възел."""
    in_degree  = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    features = {}
    for node in G.nodes():
        features[node] = {
            'in_degree':  in_degree[node],
            'out_degree': out_degree[node],
        }

    return features

def _pagerank_features(G: nx.DiGraph) -> dict:
    """Изчислява PageRank за всеки възел."""
    pagerank = nx.pagerank(G)

    return {node: {'pagerank': round(score, 8)}
            for node, score in pagerank.items()}