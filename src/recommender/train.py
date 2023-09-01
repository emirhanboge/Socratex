from data_prep.dataloader import SocratexDataset

import csv
import json

import torch
import torch.nn as nn
import torch.optim as optim

from nltk.translate.bleu_score import sentence_bleu


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size : int,
        max_len: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        dropout: float
    ) -> None:
        """
        Args:
            input_size: The size of the input.
            output_size: The size of the output.
            max_len: The maximum length of the sequence.
            num_layers: The number of layers in the encoder.
            num_heads: The number of heads in the multi-head attention.
            hidden_size: The size of the hidden layer.
            dropout: The dropout probability.
        """
        super(Transformer, self).__init__()

        self.token_embedding = nn.Embedding(input_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size, dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: The input sequence.

        Returns:
            The output logits.
        """
        sequence_length = sequence.size(1)

        position = torch.arange(sequence_length, device=sequence.device).unsqueeze(0)

        token_embedded = self.token_embedding(sequence)
        positional_embedded = self.positional_embedding(position)

        embedded = token_embedded + positional_embedded
        encoded = self.encoder(embedded)

        pooled = encoded.mean(dim=1)
        logits = self.fc(pooled)

        return logits

def load_dataloaders() -> tuple:
    """
    Load the dataloaders.

    Returns:
        The training, validation, and test dataloaders.
    """
    train_loader = torch.load("data/train_loader.pt")
    val_loader = torch.load("data/val_loader.pt")
    test_loader = torch.load("data/test_loader.pt")
    return train_loader, val_loader, test_loader

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> nn.Module:
    """
    Train the model.

    Args:
        model: The model to train.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        criterion: The loss function.
        optimizer: The optimizer.

    Returns:
        The trained model.
    """
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        avg_bleu_score = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.long())
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=1).cpu().numpy()
                reference = labels.cpu().numpy()

                candidate = int_to_node_seq(predicted)
                reference = int_to_node_seq(reference)

                bleu_score = sentence_bleu([reference], candidate)
                avg_bleu_score += bleu_score

        avg_bleu_score /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}, Val BLEU: {avg_bleu_score}")

    return model

def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module
) -> None:
    """
    Test the model.

    Args:
        model: The model to test.
        test_loader: The test data loader.
        criterion: The loss function.
    """
    model.eval()
    avg_bleu_score = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.long())
            predicted = torch.argmax(output, dim=1).cpu().numpy()
            reference = target.cpu().numpy()
            candidate = int_to_node_seq(predicted)
            reference = int_to_node_seq(reference)
            bleu_score = sentence_bleu([reference], candidate)
            avg_bleu_score += bleu_score
    avg_bleu_score /= len(test_loader)
    print(f"Average BLEU Score: {avg_bleu_score}")

def int_to_node_seq(int_seq: list) -> list:
    """
    Convert an integer sequence to a node sequence.

    Args:
        int_seq: The integer sequence.

    Returns:
        The node sequence.
    """
    return [int_to_node[str(i)] for i in int_seq]

if __name__ == '__main__':
    with open("data/node_to_int.json", "r") as f:
        node_to_int = json.load(f)

    with open("data/int_to_node.json", "r") as f:
        int_to_node = json.load(f)

    vocab_size = len(node_to_int)
    hidden_size = 64
    num_heads = 8
    num_layers = 3
    dropout = 0.1
    max_len = 15
    learning_rate = 0.01

    model = Transformer(
        input_size=vocab_size,
        output_size=vocab_size,
        max_len=max_len,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        dropout=dropout
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader, test_loader = load_dataloaders()
    model = train_model(model, train_loader, val_loader, criterion, optimizer)
    test_model(model, test_loader, criterion)
    torch.save(model, "data/model.pt")

