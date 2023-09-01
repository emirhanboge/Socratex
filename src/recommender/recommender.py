import networkx as nx
import sys

sys.path.append("../..")

from create_graph import load_weighted_graph

G = load_weighted_graph("../../../data/graph.yml")

def get_recommended_concepts(label: str, n: int = 5) -> list:
    """
    Given a label, find 'n' closest related concepts based on the graph.

    Parameters:
        label (str): Label of the concept selected by the user.
        n (int): Number of recommendations to make.

    Returns:
        list: List of 'n' recommended concepts.
    """
    if label not in G.nodes:
        return {"error": "Label not found in graph"}

    lengths = nx.single_source_dijkstra_path_length(G, label)

    recommended_labels = sorted(lengths, key=lengths.get)[1:n+1]

    return recommended_labels

if __name__ == "__main__":
    print("Getting recommended concepts for 'technology'...")
    print()
    print(get_recommended_concepts("technology"))
