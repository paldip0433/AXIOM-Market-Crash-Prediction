"""Three small neural networks. Input shape: (batch, days, features)."""
import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 32, batch_first=True)
        self.final = nn.Linear(32, 1)

    def forward(self, x):
        states, _ = self.lstm(x)
        return self.final(states[:, -1]).squeeze(-1)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Both directions only see the past 30 days supplied as input.
        self.lstm = nn.LSTM(input_size, 32, batch_first=True, bidirectional=True)
        self.final = nn.Linear(64, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        both_directions = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.final(both_directions).squeeze(-1)

class TransformerModel(nn.Module):
    def __init__(self, input_size, sequence_length=30):
        super().__init__()
        self.project = nn.Linear(input_size, 32)
        self.positions = nn.Parameter(torch.zeros(1, sequence_length, 32))
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64,
            dropout=0.15, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.final = nn.Linear(32, 1)

    def forward(self, x):
        x = self.project(x) + self.positions
        x = self.encoder(x)
        return self.final(x[:, -1]).squeeze(-1)


def make_model(name, input_size, sequence_length=30):
    if name == "LSTM":
        return LSTMModel(input_size)
    if name == "BiLSTM":
        return BiLSTMModel(input_size)
    if name == "Transformer":
        return TransformerModel(input_size, sequence_length)
    raise ValueError("Unknown model: " + name)
