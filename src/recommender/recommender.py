from train import Transformer

import argparse
import json
import yaml
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

def int_to_node_seq(int_seq: list, int_to_node: dict) -> list:
    """
    Converts a list of integers to a list of nodes.

    Args:
        int_seq (list): List of integers.
        int_to_node (dict): Dictionary mapping integers to nodes.

    Returns:
        list: List of nodes.
    """
    return [int_to_node[str(i)] for i in int_seq]

def similar(a: str, b: str) -> float:
    """
    Calculates the similarity between two strings.

    Args:
        a (str): First string.
        b (str): Second string.

    Returns:
        float: Similarity between the two strings.
    """
    return SequenceMatcher(None, a, b).ratio()

def get_most_similar(input_node: str, valid_nodes: list, threshold: int=0.5) -> str:
    """
    Finds the most similar node to the input node.

    Args:
        input_node (str): Input node.
        valid_nodes (list): List of valid nodes.
        threshold (int, optional): Threshold for similarity. Defaults to 0.8.

    Returns:
        str: Most similar node.
    """
    similarities = [(node, similar(input_node, node)) for node in valid_nodes]
    most_similar_node, highest_similarity = max(similarities, key=lambda x: x[1])

    if highest_similarity > threshold:
        return most_similar_node
    else:
        print(f"Could not find the concept: {input_node}, please check title.yml for valid concepts.")
        exit(1)

def load_data() -> tuple:
    """
    Loads the data from the data directory.

    Returns:
        tuple: Tuple containing node_to_int, int_to_node, and valid_nodes.
    """
    with open("data/node_to_int.json", "r") as f:
        node_to_int = json.load(f)
    with open("data/int_to_node.json", "r") as f:
        int_to_node = json.load(f)
    with open("../../data/titles.yml", "r") as f:
        valid_nodes = yaml.load(f, Loader=yaml.FullLoader)
        valid_nodes = list(valid_nodes.values())

    return node_to_int, int_to_node, valid_nodes

def get_recommendations(model: torch.nn.Module, input_sequence: torch.Tensor, n_recommendations: int) -> list:
    """
    Gets the recommendations for the input sequence.

    Args:
        model (torch.nn.Module): Model to use for recommendations.
        input_sequence (torch.Tensor): Input sequence.
        n_recommendations (int): Number of recommendations.

    Returns:
        list: List of recommended nodes.
    """
    with torch.no_grad():
        output = model(input_sequence)
        output = F.softmax(output, dim=-1)
        predicted_indices = torch.multinomial(output, n_recommendations)
    return predicted_indices.tolist()[0]

def remove_duplicates(predicted_nodes: list, input_nodes: list) -> list:
    """
    Removes duplicates from the predicted nodes.

    Args:
        predicted_nodes (list): List of predicted nodes.
        input_nodes (list): List of input nodes.

    Returns:
        list: List of predicted nodes without duplicates.
    """
    return [node for node in predicted_nodes if node not in input_nodes]

def main(args: argparse.Namespace) -> None:
    """
    Main function for the CLI.

    Args:
        args (argparse.Namespace): Arguments for the CLI.
    """
    model = torch.load("data/model.pt")
    model.eval()

    node_to_int, int_to_node, valid_nodes = load_data()

    input_sequence = [
        node_to_int[get_most_similar(node, valid_nodes)] for node in args.input
    ]

    print(f"Input Concepts: {int_to_node_seq(input_sequence, int_to_node)}")

    input_tensor = torch.tensor(input_sequence).unsqueeze(0).long()
    predicted_indices = get_recommendations(model, input_tensor, args.n)
    predicted_nodes = int_to_node_seq(predicted_indices, int_to_node)

    recommended_nodes = remove_duplicates(predicted_nodes, args.input)

    print(f"Recommended Concepts: {recommended_nodes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concept Recommender CLI')
    parser.add_argument('--input', type=str, nargs='+', help='Input nodes', required=True)
    parser.add_argument('--n', type=int, default=3, help='Number of recommendations. Default is 3.')

    args = parser.parse_args()

    if len(args.input) > 10:
        print("Maximum number of input nodes is 10.")
        exit(1)

    n_recommendations = min(args.n, 10)
    main(args)

