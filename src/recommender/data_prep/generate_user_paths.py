import sys
import random

sys.path.append("..")

from create_graph import load_weighted_graph

import networkx as nx
import community as community_louvain
import csv

def generate_user_paths(G, num_paths=10000, min_path_length=5, max_path_length=15):
    partition = community_louvain.best_partition(G)

    paths = []

    for _ in range(num_paths):
        path_length = random.randint(min_path_length, max_path_length)

        start_node = random.choice(list(G.nodes()))
        current_community = partition[start_node]

        path = [start_node]
        dont_repeat = [start_node]
        for _ in range(path_length - 1):
            current_node = path[-1]
            neighbors = list(G.neighbors(current_node))
            neighbors = [n for n in neighbors if n not in dont_repeat]

            if not neighbors:
                break

            in_community_neighbors = [n for n in neighbors if partition[n] == current_community]
            if in_community_neighbors and random.random() < 0.7:
                neighbors = in_community_neighbors

            edge_weights = [G[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
            total_weight = sum(edge_weights)
            probabilities = [weight / total_weight for weight in edge_weights]

            if random.random() < 0.1:
                next_node = random.choice(neighbors)
            else:
                next_node = random.choices(neighbors, probabilities)[0]

            current_community = partition[next_node]

            path.append(next_node)
            dont_repeat.append(next_node)

        paths.append(path)

    return paths

if __name__ == "__main__":
    G = load_weighted_graph("../../../data/graph.yml", "../../../data/titles.yml")
    user_paths = generate_user_paths(G)

    with open('../../../data/user_paths.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path_{}".format(i+1) for i in range(len(user_paths[0]))])
        for path in user_paths:
            writer.writerow(path)
