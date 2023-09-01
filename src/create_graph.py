import yaml
import networkx as nx
from math import log


def jaccard_similarity(node1: str, node2: str, G: nx.Graph) -> float:
    """Calculate the Jaccard similarity between two nodes in a graph.

    Args:
        node1: The first node.
        node2: The second node.
        G: The graph.

    Returns:
        The Jaccard similarity between the two nodes.
    """
    neighbors_node1 = set(G.neighbors(node1))
    neighbors_node2 = set(G.neighbors(node2))
    intersection = len(neighbors_node1.intersection(neighbors_node2))
    union = len(neighbors_node1.union(neighbors_node2))
    return intersection / union


def adamic_adar_index(node1: str, node2: str, G: nx.Graph) -> float:
    """Calculate the Adamic-Adar index between two nodes in a graph.

    Args:
        node1: The first node.
        node2: The second node.
        G: The graph.

    Returns:
        The Adamic-Adar index between the two nodes.
    """
    neighbors_node1 = set(G.neighbors(node1))
    neighbors_node2 = set(G.neighbors(node2))
    common_neighbors = neighbors_node1.intersection(neighbors_node2)
    index = sum(1 / log(len(list(G.neighbors(u)))) for u in common_neighbors)
    return index


def preferential_attachment(node1: str, node2: str, G: nx.Graph) -> int:
    """Calculate the preferential attachment between two nodes in a graph.

    Args:
        node1: The first node.
        node2: The second node.
        G: The graph.

    Returns:
        The preferential attachment between the two nodes.
    """
    return len(list(G.neighbors(node1))) * len(list(G.neighbors(node2)))


def load_graph_from_yaml(filename: str, titlesfile: str) -> nx.Graph:
    """Load a graph from a YAML file.

    Args:
        filename: The name of the file to load the graph from.
        titlesfile: The name of the file to load the titles from.

    Returns:
        The graph loaded from the YAML file.
    """
    with open(filename, "r") as file:
        graph_dict = yaml.safe_load(file)
    with open(titlesfile, "r") as file:
        titles_dict = yaml.safe_load(file)

    G = nx.Graph()
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(titles_dict[node], titles_dict[neighbor])
    return G


def create_weighted_graph(G: nx.Graph) -> nx.Graph:
    """Create a weighted graph from an unweighted graph using the
    Jaccard similarity, Adamic-Adar index, and preferential attachment.

    Args:
        G: The unweighted graph.

    Returns:
        The weighted graph.
    """
    weighted_G = nx.Graph()
    for node1, node2 in G.edges():
        weight1 = jaccard_similarity(node1, node2, G)
        weight2 = adamic_adar_index(node1, node2, G)
        weight3 = preferential_attachment(node1, node2, G)
        combined_weight = weight1 + weight2 + weight3

        weighted_G.add_edge(node1, node2, weight=combined_weight)
    return weighted_G


def load_weighted_graph(filename: str, titlesfile: str) -> nx.Graph:
    """Load the graph, create weights.

    Args:
        filename: The name of the file to load the graph from.
        titlesfile: The name of the file to load the titles from.

    Returns:
        The graph loaded from the YAML file.
    """
    G = load_graph_from_yaml(filename, titlesfile)
    weighted_G = create_weighted_graph(G)
    return weighted_G
