import pandas as pd
import json

def read_paths_from_csv(filepath: str) -> list:
    """
    Load paths data from a CSV file.

    Args:
        filepath: The name of the file to load the paths from.

    Returns:
        The paths data.
    """
    df = pd.read_csv(filepath)
    print(df.head())
    return df.values.tolist()

def tokenize_and_pad(paths: list, pad_length: int) -> tuple:
    """
    Tokenize and pad the paths.

    Args:
        paths: The paths to tokenize and pad.
        pad_length: The length to pad the paths to.

    Returns:
        The tokenized and padded paths, the node to integer mapping, and the integer to node mapping.
    """
    node_to_int = {}
    int_to_node = {}
    token_count = 0

    tokenized_paths = []
    for path in paths:
        tokenized_path = []
        for node in path:
            if pd.isna(node):
                continue
            if node not in node_to_int:
                node_to_int[node] = token_count
                int_to_node[token_count] = node
                token_count += 1
            tokenized_path.append(node_to_int[node])
        tokenized_paths.append(tokenized_path)

    for path in tokenized_paths:
        if len(path) < pad_length:
            padding = [100] * (pad_length - len(path))
            path.extend(padding)

    print("Number of unique nodes: {}".format(len(node_to_int)))

    return tokenized_paths, node_to_int, int_to_node

def save_tokenized_paths(tokenized_paths: list, filepath: str) -> None:
    """
    Save tokenized paths to a CSV file.

    Args:
        tokenized_paths: The tokenized paths to save.
        filepath: The filepath to save the tokenized paths to.
    """
    df = pd.DataFrame(tokenized_paths)
    df.to_csv(filepath, index=False)

def save_dict(d: dict, filepath: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        d: The dictionary to save.
        filepath: The filepath to save the dictionary to.
    """
    with open(filepath, 'w') as f:
        json.dump(d, f)

if __name__ == "__main__":
    paths = read_paths_from_csv('../../../data/user_paths.csv')
    del_dups = lambda l: list(dict.fromkeys(l))
    paths = [del_dups(path) for path in paths]
    paths = [path for path in paths if len(path) >= 5] # Remove paths shorter than 5 nodes
    max_len = max([len(path) for path in paths])
    print("Max path len: {}".format(max_len))
    pad_length = max_len + 1
    tokenized_paths, node_to_int, int_to_node = tokenize_and_pad(paths, pad_length)

    save_tokenized_paths(tokenized_paths, '../data/tokenized_paths.csv')
    save_dict(node_to_int, '../data/node_to_int.json')
    save_dict(int_to_node, '../data/int_to_node.json')
