import csv
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

class SocratexDataset(Dataset):
    def __init__(self, data: list, targets: list) -> None:
        """
        Args:
            data: The data.
            targets: The targets.
        """
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple:
        """
        Args:
            index: The index of the item to retrieve.

        Returns:
            The data and target at the given index.
        """
        return torch.tensor(self.data[index], dtype=torch.long), torch.tensor(self.targets[index], dtype=torch.long)

    def __len__(self) -> int:
        """
        Returns:
            The length of the dataset.
        """
        return len(self.data)

def generate_subsequences(sequence: list) -> tuple:
    """
    Generate subsequences from a sequence.

    Args:
        sequence: The sequence.

    Returns:
        The subsequences and labels.
    """
    subsequences = []
    labels = []
    for i in range(1, len(sequence)):
        subsequences.append(sequence[:i])
        labels.append(sequence[i])
    max_length = max([len(subsequence) for subsequence in subsequences])
    for i in range(len(subsequences)):
        if len(subsequences[i]) < max_length:
            padding = [100] * (max_length - len(subsequences[i]))
            subsequences[i].extend(padding)
    if len(labels) < max_length:
        padding = [100] * (max_length - len(labels))
        labels.extend(padding)
    return subsequences, labels

def load_data_from_csv(file_path: str) -> tuple:
    """
    Load data from a CSV file.

    Args:
        file_path: The path to the CSV file.

    Returns:
        The data and labels.
    """
    data = []
    labels = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            subsequences, lbls = generate_subsequences(list(map(int, row)))
            data.extend(subsequences)
            labels.extend(lbls)
    return data, labels


def split_dataset(tensor_data: torch.Tensor, train_ratio: float, val_ratio: float) -> tuple:
    """
    Split a dataset into training, validation, and test sets.

    Args:
        tensor_data: The dataset to split.
        train_ratio: The ratio of the dataset to use for training.
        val_ratio: The ratio of the dataset to use for validation.

    Returns:
        The training, validation, and test sets.
    """
    total_size = len(tensor_data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    return random_split(tensor_data, [train_size, val_size, test_size])

def create_data_loaders(
    train_set: SocratexDataset,
    val_set: SocratexDataset,
    test_set: SocratexDataset,
    batch_size: int
) -> tuple:
    """
    Create data loaders for the training, validation, and test sets.

    Args:
        train_set: The training set.
        val_set: The validation set.
        test_set: The test set.
        batch_size: The batch size.

    Returns:
        The data loaders.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

def create_datasets(
    train_data: list,
    val_data: list,
    test_data: list,
    train_labels: list,
    val_labels: list,
    test_labels: list
) -> tuple:
    """
    Create datasets for the training, validation, and test sets.

    Args:
        train_data: The training data.
        val_data: The validation data.
        test_data: The test data.
        train_labels: The training labels.
        val_labels: The validation labels.
        test_labels: The test labels.

    Returns:
        The datasets.
    """
    train_set = SocratexDataset(train_data, train_labels)
    val_set = SocratexDataset(val_data, val_labels)
    test_set = SocratexDataset(test_data, test_labels)
    return train_set, val_set, test_set

def create_data_loaders_from_csv(
    file_path: str,
    train_ratio: float,
    val_ratio: float,
    batch_size: int
) -> tuple:
    """
    Create data loaders from a CSV file.

    Args:
        file_path: The path to the CSV file.
        train_ratio: The ratio of the dataset to use for training.
        val_ratio: The ratio of the dataset to use for validation.
        batch_size: The batch size.

    Returns:
        The data loaders.
    """
    data, labels = load_data_from_csv(file_path)
    all_data = list(zip(data, labels))
    train_set, val_set, test_set = split_dataset(all_data, train_ratio, val_ratio)
    train_data, train_labels = zip(*train_set)
    val_data, val_labels = zip(*val_set)
    test_data, test_labels = zip(*test_set)
    train_set, val_set, test_set = create_datasets(train_data, val_data, test_data, train_labels, val_labels, test_labels)
    train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set, batch_size)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_data_loaders_from_csv("../data/tokenized_paths.csv", 0.90, 0.05, 32)
    torch.save(train_loader, "../data/train_loader.pt")
    torch.save(val_loader, "../data/val_loader.pt")
    torch.save(test_loader, "../data/test_loader.pt")
    print(f"Train size: {len(train_loader) * 32}")
    print(f"Val size: {len(val_loader) * 32}")
    print(f"Test size: {len(test_loader) * 32}")
    print(f"Total size: {len(train_loader) * 32 + len(val_loader) * 32 + len(test_loader) * 32}")

