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
    plt.figure(figsize=(30, 30))

    print("Finding communities...")
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    community_index = {node: i for i, community in enumerate(top_level_communities) for node in community}

    print("Finding positions...")
    pos = {}
    for com in top_level_communities:
        subgraph = G.subgraph(com)
        sub_pos = nx.spring_layout(subgraph, k=0.3, seed=42)
        pos.update(sub_pos)

    print("Drawing nodes...")
    node_colors = [community_index[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.rainbow)

    print("Drawing edges...")
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_weights, edge_cmap=plt.cm.Blues, arrows=True, arrowstyle="->", arrowsize=10)

    print("Drawing labels...")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    print("Saving image...")
    plt.axis("off")
    plt.savefig("../../images/weighted_graph.png")

    print("Showing image...")
    plt.show()


def visualize_communities(G: nx.Graph) -> None:
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
    G = load_weighted_graph("../../data/graph.yml", "../../data/titles.yml")
    print("Graph is loaded.")

    visualize_weighted_graph(G)

