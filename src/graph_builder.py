import networkx as nx
import pandas as pd 
def _build_edges(G: nx.DiGraph, df: pd.DataFrame) -> nx.DiGraph:
    """Добавя ребра от DataFrame към графа — оптимизирана версия."""
    
    # Агрегирай ВСИЧКО наведнъж с pandas (бързо!)
    edges_agg = df.groupby(['SrcAddr', 'DstAddr']).agg(
        flow_count  = ('SrcAddr', 'count'),
        total_bytes = ('TotBytes', 'sum'),
        total_pkts  = ('TotPkts', 'sum'),
        avg_dur     = ('Dur', 'mean'),
    ).reset_index()

    # Добави ребрата от готовата таблица (бързо!)
    for row in edges_agg.itertuples(index=False):
        G.add_edge(row.SrcAddr, row.DstAddr,
            flow_count  = row.flow_count,
            total_bytes = row.total_bytes,
            total_pkts  = row.total_pkts,
            avg_dur     = row.avg_dur,
        )

    return G

def _mark_botnet_nodes(G: nx.DiGraph, df: pd.DataFrame) -> nx.DiGraph:
    """Маркира възлите като botnet/normal."""
    botnet_src = set(df[df['is_botnet'] == 1]['SrcAddr'])
    botnet_dst = set(df[df['is_botnet'] == 1]['DstAddr'])
    botnet_ips = botnet_src | botnet_dst  # обединение на двете множества

    for node in G.nodes():
        G.nodes[node]['is_botnet'] = 1 if node in botnet_ips else 0

    return G

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Строи насочен граф от network flows.

    Args:
        df: почистен DataFrame от load_ctu13()

    Returns:
        nx.DiGraph с възли (IP) и ребра (комуникации)
    """
    print("Строене на граф...")

    G = nx.DiGraph()

    G = _build_edges(G, df)
    G = _mark_botnet_nodes(G, df)

    print(f"Възли: {G.number_of_nodes():,}")
    print(f"Ребра: {G.number_of_edges():,}")

    botnet_nodes = sum(1 for n in G.nodes() if G.nodes[n]['is_botnet'] == 1)
    print(f"Botnet възли: {botnet_nodes:,}")

    return G