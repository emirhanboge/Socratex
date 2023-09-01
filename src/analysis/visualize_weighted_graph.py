import sys
sys.path.append("..")

from create_graph import load_weighted_graph

import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import community

def visualize_weighted_graph(G: nx.Graph) -> None:
    """Visualize a weighted graph.

    Args:
        G: The weighted graph.
    """
    plt.figure(figsize=(24, 18))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos)
    edges = nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=10, width=2)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    plt.axis("off")
    plt.savefig("../../images/weighted_graph.png")
    plt.show()

def visualize_communities(G: nx.Graphs) -> None:
    """Visualize communities in a graph.

    Args:
        G: The graph.
    """
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    sorted(map(sorted, next_level_communities))

    plt.figure(figsize=(24, 18))
    pos = nx.spring_layout(G, seed=42)
    for communities in next_level_communities:
        nx.draw_networkx_nodes(G, pos, nodelist=communities, node_color="r", alpha=0.9, node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    plt.axis("off")
    plt.savefig("../../images/communities.png")
    plt.show()

if __name__ == "__main__":
    G = load_weighted_graph("../../data/graph.yml")
    print("Graph is loaded.")

    visualize_weighted_graph(G)

    visualize_communities(G)
