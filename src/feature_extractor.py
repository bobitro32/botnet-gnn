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

def _betweenness_features(G: nx.DiGraph) -> dict:
    """Изчислява betweenness centrality за всеки възел."""
    betweenness = nx.betweenness_centrality(G, k=500)

    return {node: {'betweenness': round(score, 8)}
            for node, score in betweenness.items()}


def extract_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Извлича всички графови метрики за всеки възел.

    Args:
        G: насочен граф от build_graph()

    Returns:
        DataFrame с features за всеки IP възел
    """
    print("Извличане на features...")

    degree      = _degree_features(G)
    pagerank    = _pagerank_features(G)
    betweenness = _betweenness_features(G)

    rows = []
    for node in G.nodes():
        rows.append({
            'ip':          node,
            'in_degree':   degree[node]['in_degree'],
            'out_degree':  degree[node]['out_degree'],
            'pagerank':    pagerank[node]['pagerank'],
            'betweenness': betweenness[node]['betweenness'],
            'is_botnet':   G.nodes[node]['is_botnet'],
        })

    df = pd.DataFrame(rows)
    print(f"Features извлечени за {len(df):,} възела")
    return df