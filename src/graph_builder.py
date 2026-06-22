import networkx as nx
import pandas as pd 
def _build_edges(G: nx.DiGraph, df: pd.DataFrame) -> nx.DiGraph:
    """Добавя ребра от DataFrame към графа."""
    for _, row in df.iterrows():
        src = row['SrcAddr']
        dst = row['DstAddr']

        if G.has_edge(src, dst):
            # Реброто вече съществува — агрегирай
            G[src][dst]['flow_count']  += 1
            G[src][dst]['total_bytes'] += row['TotBytes']
            G[src][dst]['total_pkts']  += row['TotPkts']
        else:
            # Ново ребро
            G.add_edge(src, dst,
                flow_count  = 1,
                total_bytes = row['TotBytes'],
                total_pkts  = row['TotPkts'],
                avg_dur     = row['Dur'],
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