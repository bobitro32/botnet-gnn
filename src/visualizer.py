import networkx as nx
from pyvis.network import Network

def _select_subgraph(G: nx.DiGraph, max_nodes: int = 200) -> nx.DiGraph:
    """Избира подозрителни възли + съседите им за визуализация."""
    botnet_nodes = [n for n in G.nodes() if G.nodes[n]['is_botnet'] == 1]

    selected = set(botnet_nodes)

    # Добави съседите на всеки botnet възел
    for node in botnet_nodes:
        selected.update(G.predecessors(node))
        selected.update(G.successors(node))

        if len(selected) >= max_nodes:
            break

    selected = list(selected)[:max_nodes]

    return G.subgraph(selected).copy()

def visualize_graph(G: nx.DiGraph, output_path: str = 'botnet_graph.html'):
    """Създава интерактивна HTML визуализация на подозрителните IP-та."""
    print("Подготовка на визуализацията...")

    subG = _select_subgraph(G)
    print(f"Визуализирани възли: {subG.number_of_nodes()}")
    print(f"Визуализирани ребра: {subG.number_of_edges()}")

    net = Network(height='750px', width='100%', directed=True,
                  bgcolor='#1a1a1a', font_color='white')

    for node in subG.nodes():
        is_botnet = subG.nodes[node]['is_botnet']

        net.add_node(
            node,
            label=node,
            color='#e74c3c' if is_botnet == 1 else '#3498db', # red - botnet, blue - normal
            size=25 if is_botnet == 1 else 15,
            title=f"IP: {node}\nBotnet: {'Да' if is_botnet else 'Не'}"
        )

    for src, dst, data in subG.edges(data=True):
        net.add_edge(src, dst, value=data.get('flow_count', 1))

    net.show_buttons(filter_=['physics'])
    net.save_graph(output_path)

    print(f"Визуализацията е запазена: {output_path}")